# Cache-aware scheduling for llama-server

Status: design approved, pending implementation plan
Branch: `feat/cache-aware-sched` (off `wip-all`)

> This document is local design material. It must be dropped from any upstream-bound
> branch; keep it as a standalone first commit so it is trivial to omit.

## Problem

`llama-server` schedules deferred tasks FIFO and evicts prompt-cache entries by LRU.
Neither decision consults the one piece of information that determines the cost of the
choice: how much of a queued request's prompt is already resident in KV.

Measured over 24h on a shared server (Qwen3.8-27B Q4, `-np 1`, `--cache-ram` 32 GiB,
712 requests, 22.0h busy):

| | value |
|---|---|
| queue wait, arrival -> slot launch | median 42s, p75 199s, p90 605s, max 1823s |
| aggregate time queued | 36.5 h |
| requests reusing >=95% of context | 591 / 712 |
| requests with >=30k context reusing <5% | 51, costing 1.79 h of prefill |
| of those, evicted **during their own queue wait** | 73% (vs 9% of healthy requests) |
| median queue wait, lost-cache vs healthy | 9.9 min vs 0.6 min |

**1.45 h of the 1.79 h of wasted prefill is requests whose cache was evicted while they
were already sitting in the queue.** LRU had definitive evidence those entries would be
needed within minutes — their owners had queued tasks — and discarded them anyway.

The load pattern that produces this is an agentic tool-call loop: a session finishes a
tool call and resubmits immediately (median client-side gap 0.0 min), so its prefix is
warm at arrival and dead by launch.

## Goals

- Do not evict a cache entry that a queued task needs.
- Serve the queued task with the most resident prefix first.
- Work for any `-np`; `-np 1` is the degenerate case, not the design target.
- Default-off, byte-identical behaviour when off.
- Keep `server_queue` ignorant of slots and caches, so the change is upstreamable.

## Non-goals

- **Fairness. Explicitly none.** No aging bound, no starvation ceiling. Under sustained
  warm load a cold request can be deferred indefinitely. This is intended: an operator
  enabling this flag has decided throughput outranks latency fairness, and would not want
  a bound that reintroduces the cost the flag exists to remove.
- Fixing #27148 (see Interaction below). Containment only.
- Any change to the router. Nothing here touches `server-models.cpp`.

## Why the router is not involved

Everything the scheduler needs is already co-located in one `server_context`:

- `server_queue::queue_tasks_deferred` is a `std::deque<server_task>`, and `server_task`
  carries `server_tokens tokens` **already tokenized** — deferred prompts are available
  with no re-tokenization.
- `server_prompt_cache::states` is a `std::list<server_prompt_cache_state>`, each holding
  `server_prompt::tokens`. Residency is directly inspectable.
- `server_tokens::get_common_prefix()` (`server-common.h:231`) is the scoring primitive.
- `server_queue::pop_deferred_task(int id_slot)` already implements a priority rule
  ("prioritize tasks that use the specified slot, otherwise pop the first deferred task").

In router mode each model has its own backend process, so per-backend scheduling composes
with no router involvement.

## Design

### Gate

One flag, default off, enabling all three behaviours below:

```
--cache-aware-sched          (env: LLAMA_ARG_CACHE_AWARE_SCHED)
```

Name is bikesheddable; it should sit with the existing `--cache-ram` /
`--cache-reuse` / `--cache-idle-slots` family. Requires `--cache-ram != 0`; warn and
disable otherwise, matching how `--cache-idle-slots` already self-disables
(`server-context.cpp:1327-1341`).

### Layering: callback inversion

`server_queue` must not learn about slots or caches. Invert the dependency — the queue
gains a settable scoring hook that `server_context` installs at startup:

```cpp
// server-queue.h, alongside the existing callback_new_task / callback_update_slots
// Returns the number of already-resident prompt tokens this task could reuse if it were
// launched on id_slot now. Absolute token count. Higher is better.
std::function<size_t(const server_task &, int id_slot)> callback_score_task;
```

`pop_deferred_task(int id_slot)` then becomes: existing explicit-slot precedence first,
then (if `callback_score_task` is installed) argmax over `queue_tasks_deferred` by score
with FIFO tie-break, else current first-deferred behaviour.

Symmetrically, eviction needs demand information without `server_prompt_cache` learning
about the queue:

```cpp
// server-task.h
// True if some queued task's best candidate is this entry, i.e. evicting it would
// force a re-prefill that is already scheduled to happen.
using demand_fn = std::function<bool(const server_prompt_cache_state &)>;
server_prompt_cache_state * alloc(const server_prompt & prompt,
                                  size_t state_size_main,
                                  size_t state_size_drft,
                                  const demand_fn & has_demand);  // may be nullptr-equivalent
```

`server_context` owns all cross-cutting knowledge; neither the queue nor the cache gains a
dependency on the other.

### Scoring

For a deferred task `T` about to be considered for the slot that just freed:

```
score(T, id_slot) = max over candidates C of get_common_prefix(T.tokens, C)
  candidates = { prompt currently held by id_slot } union { prompt of each cache state }
```

The candidate set is scoped to `id_slot`, matching the callback signature: the task would
be launched on *that* slot, so the only reachable residency is that slot's own residual
prompt or a cache entry restorable into it. Other slots' prompts are not candidates —
they are not reachable without a restore, and their own occupants may still need them.
With `-np N`, `pop_deferred_task` is invoked once per slot release, so each slot's
candidate set is evaluated when that slot becomes available.

**Absolute token count, not a ratio.** The quantity being minimised is tokens-to-prefill,
which is absolute. Ratios are the wrong currency, and are the root of defect 2 below.

Cost: `O(|deferred| x (|idle slots| + |states|))` prefix walks, evaluated only on
slot-release (when `pop_deferred_task` is already called), not per batch iteration. With
10 deferred tasks, 6 states and 200k-token prompts that is ~12M token comparisons, single
-digit milliseconds against a median 110s service time. Acceptable, but the implementation
should measure it rather than assume.

### Demand-aware eviction

**There are two strict-LRU eviction paths, not one, and both must honour demand.**
`server_prompt_cache::alloc()` trims to `limit_size` before allocating, and
`server_prompt_cache::update()` — called immediately after every `prompt_save` — trims again
via its own `pop_front()` loops. `update()`'s token loop is always live, because the cache is
constructed as `server_prompt_cache(cache_ram_mib, n_ctx)` so `limit_tokens` is never zero,
and its dynamic limit `max(limit_tokens, limit_size/size_per_token)` derives from the same
budget `alloc()` enforces. Protecting only `alloc()` therefore lets `update()` evict a
protected entry microseconds later, which would leave the feature's central claim unmet.

Change both: walk victims in LRU order and skip any for which `has_demand()` is true.

**The demand predicate is computed once per eviction call**, not once per caller loop and not
per candidate. Per-call is the only safe granularity: the protected set is keyed on element
addresses, and no insertion happens inside a single call's victim loop, so no address can be
reused while the set is live. Hoisting it across several `alloc()`/`update()` calls lets a
`push_back` land on a freed node's address and be misread as protected.

**Protection is advisory.** If every candidate victim is protected, fall back to strict
LRU and emit a warning — otherwise the new entry cannot be cached at all, which would
deny the *running* slot its own state save (`prompt_save` at `server-context.cpp:2314`)
and simply move the loss elsewhere. Capacity still binds; this reorders which loss occurs,
it does not create room.

### Multi-slot

`pop_deferred_task` is already slot-parameterised and is invoked once per slot release, so
`-np N` needs no special case — each release scores the deferred tasks against that slot's
candidate set. Explicit `task.id_slot` binding keeps hard precedence above the score: it is
a user request, not a heuristic.

## Interaction with #27148

[#27148](https://github.com/ggml-org/llama.cpp/issues/27148) (open) reports that the
RAM-backed prompt cache can restore an unrelated finished conversation into a slot serving
a new request, with no client-visible signal (`cached_tokens` reads 0).

`server_prompt_cache::load()` (`server-task.cpp:1793-1823`) is demonstrably loose:

1. **Fresh-slot bootstrap admits nearly everything.** Line 1796 sets `f_keep_best = -1.0f`
   for an empty slot; `f_sim_best` is 0 because the lcp against an empty prompt is 0. The
   test `f_keep_best < f_keep_cur && f_sim_best < f_sim_cur` collapses to
   `f_keep_cur >= 0.25 && lcp_cur > 0`.
2. **The 0.25 guard scales with the entry, not the match.** `f_keep_cur = lcp_cur /
   entry_size`, so 20k of shared boilerplate plus 5k of unrelated content scores 0.8.
   There is no absolute floor on `lcp` and no test that the match extends past boilerplate.
3. **The conjunction is not a total order.** Requiring both metrics to improve makes the
   winner depend on `states` iteration order; no well-defined maximum is found.
4. **No conversation identity exists.** Matching is purely token-prefix.

Not established: the exact step where stale KV past the common prefix survives.
`server-context.cpp:3092` does recompute `n_past` from the common prefix after restore,
which is correct in principle. Selection looseness is proven; the corruption step is not.

**Our work exacerbates it.** Demand-aware eviction keeps entries resident longer, so
`states` holds more and older entries, giving strictly more candidates for a spurious
match on every `load()`. We would raise the incidence without touching the cause.

**Containment, gated by the same flag** (not a fix, and not claimed as one):

- Compute `boilerplate_len` = common prefix shared across all resident states plus the
  incoming tokens. This is the deployment's system-prompt-plus-tool-schema preamble, and
  it is cheap from data already in hand.
- Require `lcp_cur >= boilerplate_len + distinctive_margin` where `distinctive_margin`
  defaults to **512 tokens**, and `lcp_cur >= absolute_floor` where `absolute_floor`
  defaults to **256 tokens**. A match must be *distinctive*, not merely template-deep.
  The floor is not redundant with the margin: `boilerplate_len` is derived from whatever
  entries happen to be resident, so with one or zero states it can be near-zero and the
  margin alone would admit a 600-token coincidental match. Both are constants in this
  revision, not tunables — adding knobs before there is evidence they need tuning is
  unwarranted, and the flag is already opt-in.
- Replace the conjunction with a single well-defined key: maximise `lcp_cur` subject to
  `f_keep_cur >= 0.25`.

This is exactly the reuse of the scoring machinery the feature already needs, and it
directly attacks defect 2. Behaviour when the flag is off is unchanged, so it does not
pre-empt whatever fix #27148 eventually receives.

## Testability: the algorithm must be pure

The three decisions this feature adds are all **pure selections over token sequences**.
They are testable with no model, no context, and no server, provided they are separated
from the KV side effects they currently sit next to. That separation is a requirement of
the design, not an afterthought — it is what makes the algorithm verifiable and what makes
the diff reviewable upstream.

Feasibility confirmed:

- `server_tokens(const llama_tokens & tokens, bool has_mtmd)` (`server-common.h:175`)
  constructs from a plain `std::vector<llama_token>` with no vocab. `get_common_prefix()`
  is a const method. So test fixtures are just integer vectors.
  Caveat: `server_tokens` is **move-only** (copy ctor deleted, `server-common.h:163`), so
  fixtures must be constructed in place or moved.
- `tests/CMakeLists.txt:164` already does `target_link_libraries(test-chat PRIVATE
  server-context)`, and `server-context` is a `STATIC` library
  (`tools/server/CMakeLists.txt:5-27`). Precedent exists; no build-architecture change,
  just `llama_build_and_test(test-server-sched.cpp)` plus that link line.

Three functions to extract, each pure:

```cpp
// 1. how many resident tokens could this task reuse on this slot
size_t sched_score(const server_tokens & task,
                   const server_tokens & slot_prompt,
                   const std::list<server_prompt_cache_state> & states);

// 2. which entry to evict, honouring demand; states.end() => no unprotected victim
std::list<server_prompt_cache_state>::const_iterator
sched_pick_victim(const std::list<server_prompt_cache_state> & states,
                  const demand_fn & has_demand);

// 3. which entry to restore; states.end() => none acceptable
std::list<server_prompt_cache_state>::const_iterator
sched_pick_restore(const std::list<server_prompt_cache_state> & states,
                   const server_tokens & tokens_new,
                   const server_tokens & slot_prompt,
                   const sched_thresholds & th);
```

`server_prompt_cache::load()` keeps the `llama_state_seq_set_data_ext` calls and delegates
its choice to (3). `alloc()` delegates to (2). `pop_deferred_task` uses (1) via the
installed callback.

### Tier 1 — pure unit tests (`tests/test-server-sched.cpp`)

Fast, no model. This is where the algorithm is validated.

1. `sched_score` returns the max over candidates, as absolute tokens.
2. `sched_score` returns 0 when nothing is shared.
3. Empty slot prompt and empty `states` yield 0 rather than a spurious win.
4. Ranking picks the highest score; equal scores fall back to FIFO order.
5. `sched_pick_victim` returns LRU order when no demand.
6. `sched_pick_victim` skips a protected entry and takes the next.
7. `sched_pick_victim` returns `end()` when every entry is protected — caller decides.
8. `sched_pick_restore` accepts a genuine deep continuation.
9. `sched_pick_restore` **rejects a boilerplate-only match** — the #27148 shape: entry is
   20k shared preamble plus 5k unrelated, incoming request shares only the preamble.
10. `sched_pick_restore` respects `absolute_floor` when `boilerplate_len` collapses to ~0
    (zero or one resident state).
11. `sched_pick_restore` maximises `lcp` subject to `f_keep >= 0.25`, and is
    order-independent — shuffling `states` yields the same winner. This is the direct
    regression guard for defect 3.
12. Characterization: the **baseline** predicate accepts case 9. Written against current
    behaviour so the improvement is demonstrated rather than asserted, and so it stands as
    a reproducible artifact for #27148 independent of whether we fix it.

### Tier 2 — integration tests (`tools/server/tests`)

Wiring only, not algorithm. The existing suite drives assertions off debug markers such as
`__TEST_TAG_CACHE_IDLE_SLOT__` / `__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__`
(`server-context.cpp:1339`, `:2315`); add markers for victim-protected,
protection-fell-back-to-LRU, task-chosen-by-score, restore-rejected-as-boilerplate.

1. Flag off: ordering and eviction identical to baseline (regression guard).
2. Two conversations, cache sized for one: the warm queued request is served before the
   cold one, and its entry survives its queue wait.
3. All victims protected: falls back to LRU, warns, still caches.
4. `-np 2`: correct slot-plus-entry pair chosen; explicit `id_slot` binding still beats a
   higher score.

Note the ordering consequence for TDD: tier 1 tests 1-11 must be red before any of
`sched_*` exists, which means extraction happens first as a pure no-behaviour-change
refactor with test 12 (characterization) already green against the baseline.

## Validation: simulation over the real selection functions

Tier 1/2 tests prove the algorithm is correct. They do not show it is *worth having*. That
needs a throughput comparison — obtained by **simulation**, not by running a server.

Prefill and generation rates are parameters, so completion times are computable. A
discrete-event simulator with a fixed-rate cost model runs a 24h workload in milliseconds,
exactly and with no timing noise, which means thresholds can be pinned tightly and swept
across scenarios instead of measured once.

Critically the simulator **links the same `sched_score` / `sched_pick_task` /
`sched_pick_victim` / `sched_pick_restore` functions the server calls**, so it exercises the
real algorithm rather than a reimplementation of it. This is only possible because those
functions are pure — the testability decision and the validation strategy are the same
decision.

`tests/test-server-sched-sim.cpp` holds both the simulator and its tests. No trace capture,
no replay driver, no log parsing, no server-side prompt logging, no privacy exposure on a
shared multi-user box.

### Workload model

Every prompt in every session begins with an identical preamble (modelling the system prompt
plus tool schemas); each session then diverges, and turn `k` strictly extends turn `k-1`.
That shared-preamble-plus-divergent-body shape *is* the pathology, so it must be exact.
Sessions are closed-loop: a session's next request becomes ready `think_s` after its own
previous response completes, so finishing sooner pulls the next request forward and the
compounding effect of a better schedule is captured. Defaults are the measured production
values — 6 sessions, preamble 20k, first turn 40k, growth 600 tokens/turn, 527 generated
tokens/turn, think time 0s, prefill 900 t/s, generation 15.7 t/s, KV 60 KiB/token,
`--cache-ram` 32 GiB.

### History rewriting is part of the workload model

Whether turn k+1's prompt strictly extends turn k's is decided by the chat template and the
client, not by the server, so it is a first-class dimension of the workload: it determines
whether prefix reuse exists at all. The scheduler is template-agnostic, so the simulator
models rewriting **generically** — as a policy over which previously-rendered positions get
re-rendered and when — rather than encoding one model's behaviour.

| policy | divergence | notes |
|---|---|---|
| `APPEND_ONLY` | end of previous render | best case; non-reasoning templates, and Qwen with `preserve_thinking=true` — what the measured 24h window ran |
| `DROP_REASONING_PRIOR_TURNS` | start of previous human turn, on each new human prompt | tool calls within a turn still extend exactly |
| `DROP_REASONING_ALWAYS` | previous assistant message, every request | hostile |
| `TRUNCATE_FRONT` | position 0 once the window slides | maximally hostile; included to prove no harm where the feature cannot help |
| `COMPACT_AT_THRESHOLD` | summary insertion point | what omp's own compaction does |

Qwen3 is one instance, not the model. Its templates gate reasoning on `preserve_thinking or
loop.index0 > last_query_index`
([froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates)
`chat_template.jinja:298`), where `last_query_index` is the last **genuine** user message —
user messages rendered as `<tool_response>` are skipped when scanning back (lines 196-205).
That yields `APPEND_ONLY` when `preserve_thinking` is true and
`DROP_REASONING_PRIOR_TURNS` when false.

**Tokens are derived from a rewrite generation, not from arithmetic.** Each position is
tagged with the turn at which it was last re-rendered, and its token is a function of
(session, position, generation). So `get_common_prefix` between consecutive renders equals
the policy's declared divergence position *by construction*, and a pinned test asserts that
identity for every policy. Getting the geometry right once, in `sim_render_at`, is enough.

**Expected outcome differs by policy, and the tests say so.** Where reuse exists the feature
must reduce prefill tokens and makespan. Where it does not — `TRUNCATE_FRONT` — the only
requirement is **no harm**, asserted within 1%. A reviewer running a truncating client must
not see a regression, and that negative result is as important to have as the positive ones.

Two parameters here are **not measured** and must not be presented as though they were: tool
calls per human prompt (the server log cannot distinguish a tool response from a human
prompt) and the share of generated tokens that is reasoning. Both are swept.

### The simulator demonstrates mechanism; it does not predict production

An earlier revision of this spec required the simulator's baseline arm to reproduce the
measured 24h window, calling that its acceptance test. **That requirement was a mistake and is
withdrawn.** It conflated two different jobs:

- **Demonstrating the mechanism** — "under a workload with cache pressure, ranking by resident
  prefix re-prefills fewer tokens than FIFO, and protecting demanded entries reduces losses."
  That is a claim about the *algorithm*, true of any pressured workload. It requires the
  simulator to be **correct**, not to resemble any particular server.
- **Predicting production benefit** — that would require calibration. The PR has no need for
  it, because the measured 24h window already answers it directly, and a measurement is
  stronger evidence than a simulation of that measurement.

Chasing calibration was also actively harmful: asserting that the model reproduces production
invites a reviewer to test that claim, and the model has more free parameters
(`per_session_think_s`, active-session selection, `n_turns`, `tools_per_prompt`,
`reasoning_frac`, `preamble_tokens`) than the single observable (0.83 reuse) constrains. Two
knobs fitted against one target is interpolation, not calibration.

So the simulator carries exactly two responsibilities, neither needing calibration:

1. **Correctness of its own model.** The render self-consistency test — measured
   `get_common_prefix` between consecutive turns must equal the divergence position the policy
   declares, for every policy and session size. This is a real gate and must not be weakened.
2. **Mechanism and regression gating.** Treatment beats baseline under pressure; no harm where
   nothing is reusable; and pinned thresholds that detect any future behaviour change. These
   need **determinism and exactness**, which the simulator has, not resemblance to production.

Heterogeneous session sizes are retained — not to match a distribution, but because a mixed
workload exercises the policy more thoroughly than a uniform one.

### Sweep and pinned thresholds

Slots x cache pressure x rewrite policy: `{1,2,4,8}` slots against tight and roomy caches, across the policy table above. Per scenario, pin
minimum prefill-token reduction and minimum makespan reduction at the observed value less a
10% margin. The simulator has no timing dependence, so these are exact and host-independent
— which is what makes tight thresholds legitimate here where they would not be for a
wall-clock benchmark. Verify the gate has teeth by making `sched_pick_victim` ignore demand
and confirming the tight-cache scenarios fail.

Being algorithm-agnostic about cache behaviour, this doubles as a regression gate for any
future prompt-cache or eviction work, not just this flag.

### What each layer of evidence establishes

Be explicit in the PR, because a reviewer will be:

- **Motivation** is measured: 24h of production logs, 712 requests, 73% of lost-cache
  requests evicted during their own queue wait, 1.45h of 1.79h of wasted prefill. Observed,
  not modelled.
- **Correctness** is the tier-1 unit tests.
- **Integration** is the tier-2 server tests.
- **Mechanism** is *simulated* under a fixed-rate cost model on a synthetic pressured
  workload. Simulated deltas are **not** production estimates and must not be presented as
  such; the measured window is what speaks to real-world benefit.
- **Known limit:** constant prefill and generation rates. Real generation degrades with
  context length (measured 30.9 t/s below 25k, 13.2 t/s above 180k), but the scheduler does
  not change context lengths, so that term largely cancels between arms. State this rather
  than leaving it to be found.
- Report the cold-request p99 wait from the sweep alongside the throughput gain.

## Risks

- **Starvation** under sustained warm load. Accepted by decision; document in the flag's
  help text and in `README.md` so the semantics are explicit at the point of opt-in.
- **Upstream pushback on the absence of a fairness bound.** The defence is that the flag
  is off by default and the trade is documented. A reviewer may still require a bound.
- **Increased #27148 exposure**, partially contained above, not fixed.
- **Scoring overhead** on very deep queues with very long prompts. Bounded and measured,
  evaluated only on slot-release.

## Out of scope

- Router changes.
- Any change to `--cache-ram` sizing defaults.
- Cross-process or cross-host coordination. Ranking requires the tokenized prompts and
  cache residency, both of which exist only inside a backend; a client-side or router-side
  gate cannot see them. (Also measured: 92% of inter-request gaps are under 5s, so an
  "wait for idle" gate would never fire.)

---

## Revision 2026-08-24: scope reduced to residency ordering only

An independent Go simulator (`/home/claude-aiven-4/code/kvsched-sim`, see
`docs/PORTING-TO-LLAMA-CPP.md`) rebuilt the baseline cache model from source and measured each
component separately across 7 environments with a mutation harness. It overturns parts of this
spec. The corrections are load-bearing:

**This spec's baseline model was wrong in five ways.** Eviction is FIFO by insertion order, not
LRU. A cache hit **erases** the entry rather than leaving it readable. Restore selection must beat
*two* ratios simultaneously, so it is order-dependent. Selection is two-stage, and the cache is
bypassed entirely when the slot already retains >=50%. And there is **no cross-session prefix
sharing** — N diverging sessions cost N full blobs — which invalidates the "preamble-only restores
save ~20k tokens" reasoning.

**Eviction protection is structurally inert.** Under 1% on every metric bar one corner. This spec
warned that a *threshold* predicate would protect every entry and prescribed argmax to avoid it —
but argmax protects everything too, once waiters outnumber entries. At one slot with six sessions
the cache holds ~3 entries against 5-6 waiters, so every entry is protected and the advisory
fallback reproduces baseline exactly. **Removed.**

**The distinctive restore guard is not justified and costs throughput** (+24.3% prefill, −18.9%
high-reuse). The 512-token margin has no basis: KV beyond the matched prefix is truncated
unconditionally (`keep_first(n_past)` then `seq_rm(p0, -1)`), so a preamble-only restore is
provably safe. **Removed**, and `load()` returns to baseline selection.

**Residency ordering is the whole effect** — busy −26.7%, prefill −73.4%, evictions −82.7% geomean.
The port site this spec chose (`pop_deferred_task`'s FIFO branch, leaving slot-pinning precedence
alone) is confirmed correct. **Kept.**

**Recalibrate the claim.** The gain is capped by baseline prefill share, which was 10.8% in the
reference window. Quote **single-digit percent** for a production-like deployment, not −26.7%.

**Cost this spec missed entirely:** prompt-cache save/load runs at ~4-4.5 GiB/s serially on the
scheduler thread — ~2.2 s per displacement, ~23 min/day, and it stalls the whole task loop. A
separate shippable optimisation.

**Next:** change A (linger ~200 ms on release, then serve the newest in-window arrival) targets the
disjoint case ordering cannot reach — the incumbent that has not resubmitted yet. Median gap
between a response completing and its successor arriving is 44 ms over 50,072 measured tool-loop
iterations. Its 200 ms bound is also its own anti-starvation guard.
