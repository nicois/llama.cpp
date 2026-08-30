# Cache-aware scheduling: PR readiness

Triage of the `feat/cache-aware-sched` work, written 2026-08-25 against
`server : keep the slot linger deadline per slot`. This file lives in `docs/superpowers/`
and must be dropped from any upstream-bound branch, like the plan and the spec.

The branch ships two independent, default-off features:

- `--cache-aware-sched`: rank deferred tasks by resident prefix instead of FIFO.
- `--slot-linger-ms`: on slot release, hold the slot briefly so an in-window follow-up
  is served ahead of the older backlog.

## Fixed, and safe to carry into a PR

| Fix | Where | Evidence |
|---|---|---|
| Linger deadline is per slot, not one global record | `server-queue.{h,cpp}` | `test_linger_does_not_drop_concurrent_slot_releases`. Fails at 4.07s against a 2.6s bound before the fix. |
| No linger when a deferred task is already bound to the releasing slot | `server-queue.cpp` | `test_no_linger_when_a_deferred_task_is_bound_to_the_slot`. Before the fix the window fires and the bound task waits it out. |
| `pop_deferred_task()` uses `sched_pick_task()` instead of its own copy of the argmax | `server-queue.cpp` | The unit test in `test-server-sched.cpp` now guards the shipping path. Behaviour was already equivalent; this removes the divergence risk. |
| Unused `has_demand` lambda deleted | `test-server-sched-sim.cpp` | `-DLLAMA_FATAL_WARNINGS=ON` builds again. It is on by default in `ci/run.sh` and explicit in build-cpu, build-cuda-ubuntu, build-apple, build-ibm, build-android and release.yml. |
| Scheduling test scaffolding shared via `utils.py` | `tools/server/tests/` | Three tests were duplicated verbatim across the two files. |
| `--slot-linger-ms` help text states the real starvation bound | `common/arg.cpp` | See "Starvation" below. |

### First-bug detail: per-slot linger

`linger_until_ms` and `linger_slot_id` were single scalars. A release from slot B while slot A
was lingering overwrote A's record, so A's deferred promotion was lost: one deferred task was
promoted per window regardless of how many slots freed. With `-np > 1` the extra slots idled
while tasks were still deferred. All the original tests used `n_slots=1`, so nothing caught it.

### Second-bug detail: bound task waits out the window

`can_linger` only checked that the deferred queue was non-empty. A task pinned via `id_slot` to
the slot that just released is already the best possible use of that slot, and `pop_deferred_task()`
promotes it on its first branch, so the window added pure latency with no possible cache benefit.

## Open, and blocking for upstream

1. **`__TEST_TAG_*` log markers are compiled into `llama-server`.**
   `server-context.cpp` and `server-queue.cpp` emit four of them at debug level purely so the
   pytest suite can scrape them. The adjacent `QUE_DBG`/`SRV_DBG` lines already carry the same
   information. Assert on those and drop the markers before proposing this upstream.

2. **The plan document is stale.**
   `docs/superpowers/plans/2026-08-23-cache-aware-scheduling.md` still specifies
   `sched_pick_victim`, `sched_pick_restore`, `sched_demand_fn`, `sched_thresholds`,
   `absolute_floor = 256` and `distinctive_margin = 512`. All of those were removed by
   `97d90eaec` when the scope was cut to residency ordering. The reduction was recorded in the
   spec only, so the plan now contradicts the code. Either amend it or drop it.

3. **Starvation under `--slot-linger-ms` is unbounded in arrivals, not in `N`.**
   Measured with `-np 1 --slot-linger-ms 400`, one deferred victim and a stream of follow-ups
   each arriving inside the fresh window:

   | `--slot-linger-ms` | victim latency | follow-ups served first |
   |---|---|---|
   | 0 | 0.49s | 0 |
   | 400 | 2.82s | 8 (all of them) |

   The victim only completed because the stream stopped at 8. Mechanism: a hit calls
   `cancel_linger_if_active()`, which clears the window without promoting anything from the
   deferred queue; when the new task releases, the window re-arms. The backlog is therefore
   passed over once per arrival.

   Not fixed on purpose. The spec declares fairness an explicit non-goal ("no aging bound, no
   starvation ceiling, do not add one"), so capping consecutive hits is a policy change that
   belongs to whoever owns the design, and it would cut the tool-loop benefit the flag exists
   for. The help text now states the behaviour. **This is the one open decision.**

4. **`sched_score()` over-estimates reusable prefix.**
   It returns the best common prefix across the slot prompt and every prompt-cache state, but
   `server_prompt_cache::load()` only restores a state when `f_keep >= 0.25` and both `f_keep`
   and `f_sim` improve on the baseline. So a task can be ranked first on a state that `load()`
   will then refuse to restore, and it re-prefills from scratch. The slot-prompt half of the
   score is exact; only the cache-state half is optimistic. Read from the code, not measured -
   worth a targeted test before claiming a throughput number in a PR.

5. **Scoring cost is paid under `mutex_tasks`.**
   `pop_deferred_task()` calls the score callback for every unpinned deferred task, and each
   call walks every cache state doing `get_common_prefix()`. That is
   O(n_deferred x n_states x prompt_len) with the queue mutex held, on every slot release.
   Harmless at the tested scale (1 MiB cache-ram, a handful of states); the default is
   `--cache-ram 8192`, where the state list is much longer. Not measured.

## Verification state

- `-DLLAMA_FATAL_WARNINGS=ON` build of `llama-server`, `test-server-sched`,
  `test-server-sched-sim`: clean.
- `test-server-sched`: 8 assertions. `test-server-sched-sim`: 344 assertions. No failures.
- `test_cache_aware_sched.py` + `test_slot_linger.py`: 8 passed.
- Baseline server suites (basic, completion, chat_completion, slot_save, ctx_shift,
  kv_keep_only_active): 113 passed, 1 skipped. The one failure,
  `test_completion_stream_with_openai_library_stops`, needs to download Phi-3.5-mini and fails
  the same way on `wip-all`.
- Note for reruns on this host: pass `PORT=8123`. Port 8080 holds a long-running router, and
  the suite will silently talk to it and report `400 model name is missing`.
