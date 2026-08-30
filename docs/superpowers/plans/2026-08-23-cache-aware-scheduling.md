# Cache-Aware Scheduling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `llama-server` stop evicting prompt-cache entries whose owners have queued requests, and serve the queued request with the most resident prefix first, behind one opt-in flag.

**Architecture:** Extract the three scheduling decisions (score a task, pick an eviction victim, pick a restore candidate) as pure functions over token sequences in a new `server-sched.{h,cpp}`, separated from the KV side effects they currently sit beside. `server_queue` gains a settable scoring callback so it stays ignorant of slots and caches; `server_prompt_cache::alloc()` gains a demand predicate so it stays ignorant of the queue. `server_context` owns all cross-cutting knowledge and installs both.

**Tech Stack:** C++17, CMake, the `tests/testing.h` harness (`struct testing`, `t.test(name, fn)`, `t.assert_equal(msg, expected, actual)`, `t.assert_true(msg, cond)`, `t.summary()`), pytest for server integration tests. No Python, no model and no server are needed for the throughput evidence — it is a C++ simulator over the same pure functions.

## Global Constraints

- Worktree: `/home/claude-aiven-4/code/llama.cpp/.claude/worktrees/cache-aware-sched`, branch `feat/cache-aware-sched`. Do not touch the main checkout or `wip-all`.
- Spec: `docs/superpowers/specs/2026-08-23-cache-aware-scheduling-design.md`. It is authoritative.
- Flag: `--cache-aware-sched`, env `LLAMA_ARG_CACHE_AWARE_SCHED`, default **off**. With the flag off, behaviour must be byte-identical to baseline.
- Thresholds are constants, not tunables: `absolute_floor = 256`, `distinctive_margin = 512`, `f_keep_min = 0.25f`.
- Scoring is in **absolute tokens**, never a ratio.
- **No existing diagnostics may be deleted.** The three `SRV_TRC` lines in
  `server_prompt_cache::load()` (pre-loop baseline, per-candidate `f_keep`/`f_sim`,
  post-selection winner) must survive the refactor. The selection functions stay pure — they
  emit nothing — and instead accept an optional `sched_restore_trace *` out-param that
  `load()` logs from. Pass `nullptr` where logging is not wanted (the simulator does).
- **Fairness is an explicit non-goal.** No aging bound, no starvation ceiling. Do not add one.
- Do not add a starvation metric to the server. Tail latency is reported by the simulator (Task 11) instead.
- `server_tokens` is **move-only** (copy ctor deleted, `tools/server/server-common.h:163`). Construct in place or `std::move`.
- Do not modify `tools/server/server-models.cpp` (the router). Out of scope.
- The design doc and this plan live in `docs/superpowers/` and must be dropped from any upstream-bound branch. Keep them in their own commits.

## File Structure

| File | Responsibility |
|---|---|
| `tools/server/server-sched.h` (create) | Declarations of the three pure selection functions, `sched_thresholds`, `sched_demand_fn`, type aliases. |
| `tools/server/server-sched.cpp` (create) | Implementations. No llama API calls, no I/O, no locks. |
| `tools/server/server-task.h` (modify) | `alloc()` gains a demand predicate; `server_prompt_cache` gains a `cache_aware_sched` flag. |
| `tools/server/server-task.cpp` (modify) | `alloc()` delegates victim choice; `load()` delegates restore choice. |
| `tools/server/server-queue.h` (modify) | `callback_score_task` hook; `for_each_deferred()` accessor. |
| `tools/server/server-queue.cpp` (modify) | `pop_deferred_task()` ranks by score after explicit-slot precedence. |
| `tools/server/server-context.h` (modify) | `prompt_save()` passes the demand predicate. |
| `tools/server/server-context.cpp` (modify) | Builds the protected set and installs both callbacks. |
| `tools/server/CMakeLists.txt` (modify) | Add `server-sched.cpp/.h` to the `server-context` library. |
| `common/common.h` (modify) | `cache_aware_sched` param. |
| `common/arg.cpp` (modify) | Flag registration. |
| `tests/test-server-sched.cpp` (create) | 12 pure unit tests. The algorithm is validated here. |
| `tests/CMakeLists.txt` (modify) | Register the test, link `server-context`. |
| `tools/server/tests/unit/test_cache_aware_sched.py` (create) | 4 integration tests for wiring only. |
| `tests/test-server-sched-sim.cpp` (create) | Discrete-event simulator over the real `sched_*` functions; throughput evidence and the pinned regression gate. |
| `tools/server/tests/utils.py` (modify) | `cache_aware_sched` server knob for the integration tests. |

## Critical Pitfall — read before Task 3

Demand must be **argmax**, not a threshold. Every conversation in this workload shares a
multi-thousand-token system-prompt-plus-tool-schema preamble, so *every* cache entry shares
more than `absolute_floor` tokens with *every* queued task. A predicate like
`lcp >= absolute_floor` would therefore protect every entry, every alloc would find no
unprotected victim, and the feature would silently degrade to permanent LRU fallback while
appearing to work.

An entry is protected **iff it is the argmax candidate for at least one deferred task.**

---

### Task 1: Extract the baseline restore selection, unchanged

Pure refactor plus a characterization test. No behaviour change. This lands the new
translation unit and the test binary, and captures current `load()` selection behaviour —
including the #27148 defect — as a reproducible artifact.

**Files:**
- Create: `tools/server/server-sched.h`
- Create: `tools/server/server-sched.cpp`
- Create: `tests/test-server-sched.cpp`
- Modify: `tools/server/CMakeLists.txt:7-27` (add sources to `server-context`)
- Modify: `tests/CMakeLists.txt` (register test, after line 164)
- Modify: `tools/server/server-task.cpp:1793-1823` (`load()` delegates selection)

**Interfaces:**
- Consumes: `server_tokens::get_common_prefix()` (`tools/server/server-common.h:231`), `server_prompt_cache_state` (`tools/server/server-task.h:597`).
- Produces: `sched_states`, `sched_state_it`, `sched_pick_restore_baseline()`.

- [ ] **Step 1: Write the failing test**

Create `tests/test-server-sched.cpp`:

```cpp
#include "testing.h"

#include "server-sched.h"

#include <list>
#include <vector>

// Build a cache state holding the given tokens. Sizes are irrelevant to selection.
static server_prompt_cache_state make_state(const std::vector<llama_token> & toks) {
    server_prompt_cache_state st;
    st.prompt.tokens = server_tokens(toks, false);
    return st;
}

// A run of n tokens starting at `base`, so prefixes are easy to construct.
static std::vector<llama_token> run(llama_token base, size_t n) {
    std::vector<llama_token> v;
    v.reserve(n);
    for (size_t i = 0; i < n; i++) {
        v.push_back(base + (llama_token) i);
    }
    return v;
}

// Concatenate two token runs.
static std::vector<llama_token> cat(const std::vector<llama_token> & a,
                                   const std::vector<llama_token> & b) {
    std::vector<llama_token> v = a;
    v.insert(v.end(), b.begin(), b.end());
    return v;
}

// Test 12 (characterization): baseline selection accepts a boilerplate-only match.
// This encodes the #27148 defect deliberately. When #27148 is fixed upstream this test
// SHOULD fail and should then be deleted.
static void test_baseline_characterization(testing & t) {
    t.test("baseline_accepts_boilerplate_only_match", [&](testing & t) {
        const auto boiler = run(1000, 20000);          // shared system prompt + tool schemas
        const auto stale  = run(90000, 5000);          // unrelated conversation content
        const auto fresh  = run(50000, 30000);         // this request's real content

        sched_states states;
        states.push_back(make_state(cat(boiler, stale)));

        server_tokens tokens_new(cat(boiler, fresh), false);
        server_tokens slot_prompt;                     // fresh slot: empty

        const auto it = sched_pick_restore_baseline(states, tokens_new, slot_prompt);

        t.assert_true("baseline selects the unrelated entry on a boilerplate-only match",
                      it != states.end());
    });
}

int main(int argc, char ** argv) {
    testing t;

    const char * verbose = getenv("LLAMA_TEST_VERBOSE");
    if (verbose) {
        t.verbose = std::string(verbose) == "1";
    }

    if (argc > 1) {
        t.set_filter(argv[1]);
    }

    t.test("baseline", test_baseline_characterization);

    return t.summary();
}
```

- [ ] **Step 2: Register the test and verify it fails to compile**

Add to `tests/CMakeLists.txt` immediately after the existing `test-chat` block that ends at line 164:

```cmake
    llama_build_and_test(test-server-sched.cpp)
    target_link_libraries(test-server-sched PRIVATE server-context)
    target_include_directories(test-server-sched PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
```

Run:
```bash
cd /home/claude-aiven-4/code/llama.cpp/.claude/worktrees/cache-aware-sched
cmake -B build -DLLAMA_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target test-server-sched -j"$(nproc)"
```
Expected: **FAIL** — `fatal error: server-sched.h: No such file or directory`.

- [ ] **Step 3: Create the header**

Create `tools/server/server-sched.h`:

```cpp
#pragma once

#include "server-task.h"

#include <functional>
#include <list>
#include <vector>

using sched_states   = std::list<server_prompt_cache_state>;
using sched_state_it = sched_states::const_iterator;

// Acceptance thresholds for restore selection. Constants, not tunables.
struct sched_thresholds {
    size_t absolute_floor     = 256;   // a match below this is coincidental
    size_t distinctive_margin = 512;   // a match must exceed shared boilerplate by this
    float  f_keep_min         = 0.25f; // do not trash large cached prompts
};

// True if evicting this entry would force a re-prefill that is already queued.
using sched_demand_fn = std::function<bool(const server_prompt_cache_state &)>;

// Optional diagnostic capture. The selection functions are pure and log nothing; a caller
// that wants the SRV_TRC output `load()` has always emitted passes one of these and logs
// from it. Pass nullptr to skip capture entirely (the simulator does).
struct sched_candidate_score {
    size_t lcp    = 0;
    float  f_keep = 0.0f;
    float  f_sim  = 0.0f;
};

struct sched_restore_trace {
    float base_f_keep = 0.0f;                        // pre-loop baseline
    float base_f_sim  = 0.0f;
    std::vector<sched_candidate_score> candidates;    // in states order, one per entry
    bool  found       = false;                        // post-selection
    float best_f_keep = 0.0f;
    float best_f_sim  = 0.0f;
};

// Current upstream selection, extracted verbatim. Retained only so the characterization
// test can demonstrate the #27148 boilerplate-only match. Do not call from new code.
sched_state_it sched_pick_restore_baseline(const sched_states  & states,
                                          const server_tokens & tokens_new,
                                          const server_tokens & slot_prompt,
                                          sched_restore_trace * trace = nullptr);
```

- [ ] **Step 4: Create the implementation, extracted verbatim**

Create `tools/server/server-sched.cpp`. This is `server_prompt_cache::load()` lines
1794-1823 moved with no semantic change:

```cpp
#include "server-sched.h"

sched_state_it sched_pick_restore_baseline(const sched_states  & states,
                                           const server_tokens & tokens_new,
                                           const server_tokens & slot_prompt) {
    const int lcp_best = slot_prompt.get_common_prefix(tokens_new);

    float f_keep_best = slot_prompt.size() > 0 ? float(lcp_best) / slot_prompt.size() : -1.0f;
    float f_sim_best  = float(lcp_best) / tokens_new.size();

    auto it_best = states.end();

    for (auto it = states.begin(); it != states.end(); ++it) {
        const int lcp_cur = it->prompt.tokens.get_common_prefix(tokens_new);

        const float f_keep_cur = float(lcp_cur) / it->prompt.tokens.size();
        const float f_sim_cur  = float(lcp_cur) / tokens_new.size();

        // don't trash large prompts
        if (f_keep_cur < 0.25f) {
            continue;
        }

        if (f_keep_best < f_keep_cur && f_sim_best < f_sim_cur) {
            f_keep_best = f_keep_cur;
            f_sim_best  = f_sim_cur;

            it_best = it;
        }
    }

    return it_best;
}
```

Add both files to the `server-context` library in `tools/server/CMakeLists.txt`, inside the
`add_library(${TARGET} STATIC ...)` list that currently spans lines 7-27:

```cmake
    server-sched.cpp
    server-sched.h
```

- [ ] **Step 5: Run the test to verify it passes**

Run:
```bash
cmake --build build --target test-server-sched -j"$(nproc)"
./build/bin/test-server-sched
```
Expected: **PASS**, 1 test, 1 assertion, 0 failures. It passes because it asserts current
behaviour — that is the point of a characterization test.

- [ ] **Step 6: Make `load()` delegate, verify no behaviour change**

In `tools/server/server-task.cpp`, replace the selection block in
`server_prompt_cache::load()` (lines 1794-1823, from `const int lcp_best` through the
closing brace of the `for` loop) with:

```cpp
    auto it_best_const = sched_pick_restore_baseline(states, tokens_new, prompt.tokens);

    // states is non-const here; convert the const_iterator for the mutating path below
    auto it_best = states.end();
    if (it_best_const != states.end()) {
        it_best = std::next(states.begin(), std::distance(states.cbegin(), it_best_const));
    }
```

Add `#include "server-sched.h"` to the top of `tools/server/server-task.cpp`.

Run the existing server suite to confirm nothing regressed:
```bash
cmake --build build --target llama-server -j"$(nproc)"
cd tools/server/tests && ./tests.sh unit/test_completion.py -v ; cd -
```
Expected: PASS, same as before the change.

- [ ] **Step 7: Commit**

```bash
git add tools/server/server-sched.h tools/server/server-sched.cpp \
        tools/server/server-task.cpp tools/server/CMakeLists.txt \
        tests/test-server-sched.cpp tests/CMakeLists.txt
git commit -m "server: extract prompt-cache restore selection as a pure function

No behaviour change. Adds a characterization test showing the current selection
accepts a boilerplate-only match, which is the #27148 shape.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `sched_score`

**Files:**
- Modify: `tools/server/server-sched.h`
- Modify: `tools/server/server-sched.cpp`
- Modify: `tests/test-server-sched.cpp`

**Interfaces:**
- Consumes: `sched_states`, `server_tokens` (Task 1).
- Produces: `size_t sched_score(const server_tokens & task_tokens, const server_tokens & slot_prompt, const sched_states & states)` — absolute count of already-resident tokens the task could reuse on that slot.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test-server-sched.cpp`, above `main`:

```cpp
// Tests 1-3: sched_score
static void test_score(testing & t) {
    t.test("returns_max_over_candidates_in_absolute_tokens", [&](testing & t) {
        const auto shared = run(1, 5000);

        sched_states states;
        states.push_back(make_state(cat(shared, run(60000, 10))));            // lcp 5000
        states.push_back(make_state(cat(run(1, 9000), run(70000, 10))));      // lcp 9000

        server_tokens task(cat(run(1, 12000), run(80000, 100)), false);
        server_tokens slot_prompt(cat(shared, run(90000, 10)), false);        // lcp 5000

        t.assert_equal("picks the deepest candidate", (size_t) 9000,
                       sched_score(task, slot_prompt, states));
    });

    t.test("returns_zero_when_nothing_is_shared", [&](testing & t) {
        sched_states states;
        states.push_back(make_state(run(500000, 1000)));

        server_tokens task(run(1, 1000), false);
        server_tokens slot_prompt(run(700000, 1000), false);

        t.assert_equal("no shared prefix", (size_t) 0,
                       sched_score(task, slot_prompt, states));
    });

    t.test("empty_slot_and_empty_states_score_zero", [&](testing & t) {
        sched_states states;
        server_tokens task(run(1, 1000), false);
        server_tokens slot_prompt;

        t.assert_equal("nothing resident", (size_t) 0,
                       sched_score(task, slot_prompt, states));
    });
}
```

Also add the ranking test (spec tier-1 test 4). Ranking must be a pure function so the
FIFO tie-break is testable without a queue:

```cpp
// Test 4: sched_pick_task
static void test_pick_task(testing & t) {
    t.test("picks_highest_score_with_fifo_tie_break", [&](testing & t) {
        t.assert_equal("highest score wins", (size_t) 2,
                       sched_pick_task(std::vector<size_t>{10, 30, 90, 40}));
        t.assert_equal("earliest wins among equals", (size_t) 1,
                       sched_pick_task(std::vector<size_t>{10, 90, 90, 90}));
        t.assert_equal("all zero falls back to first", (size_t) 0,
                       sched_pick_task(std::vector<size_t>{0, 0, 0}));
        t.assert_equal("empty yields SIZE_MAX", SIZE_MAX,
                       sched_pick_task(std::vector<size_t>{}));
    });
}
```

Register both in `main` before `t.test("baseline", ...)`:

```cpp
    t.test("score", test_score);
    t.test("pick_task", test_pick_task);
```

Add `#include <cstdint>` and `#include <iterator>` to `tests/test-server-sched.cpp`.

- [ ] **Step 2: Run to verify it fails**

Run:
```bash
cmake --build build --target test-server-sched -j"$(nproc)"
```
Expected: **FAIL** — `error: 'sched_score' was not declared in this scope`.

- [ ] **Step 3: Implement**

Add to `tools/server/server-sched.h`:

```cpp
// How many already-resident prompt tokens this task could reuse if launched on the slot
// whose current prompt is `slot_prompt`. Absolute token count; higher is better.
size_t sched_score(const server_tokens & task_tokens,
                   const server_tokens & slot_prompt,
                   const sched_states  & states);

// Index of the deferred task to run next, given one score per task in queue order:
// highest score wins, earliest wins on ties (FIFO). Returns SIZE_MAX for an empty list.
size_t sched_pick_task(const std::vector<size_t> & scores);
```

Add to `tools/server/server-sched.cpp`:

```cpp
size_t sched_score(const server_tokens & task_tokens,
                   const server_tokens & slot_prompt,
                   const sched_states  & states) {
    size_t best = slot_prompt.get_common_prefix(task_tokens);

    for (const auto & st : states) {
        best = std::max(best, (size_t) st.prompt.tokens.get_common_prefix(task_tokens));
    }

    return best;
}
```

and:

```cpp
size_t sched_pick_task(const std::vector<size_t> & scores) {
    if (scores.empty()) {
        return SIZE_MAX;
    }

    size_t best = 0;
    for (size_t i = 1; i < scores.size(); i++) {
        // strict > keeps the earliest task among equal scores, i.e. FIFO on ties
        if (scores[i] > scores[best]) {
            best = i;
        }
    }

    return best;
}
```

Add `#include <algorithm>` and `#include <cstdint>` to `tools/server/server-sched.cpp`.

- [ ] **Step 4: Run to verify it passes**

Run:
```bash
cmake --build build --target test-server-sched -j"$(nproc)" && ./build/bin/test-server-sched
```
Expected: **PASS**, 4 tests, 4 assertions, 0 failures.

- [ ] **Step 5: Commit**

```bash
git add tools/server/server-sched.h tools/server/server-sched.cpp tests/test-server-sched.cpp
git commit -m "server: add sched_score for cache-aware task ranking

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `sched_pick_victim`

Read the Critical Pitfall section above first. This function takes the predicate; computing
it correctly as an argmax is Task 6's job, but the tests here encode the contract.

**Files:**
- Modify: `tools/server/server-sched.h`
- Modify: `tools/server/server-sched.cpp`
- Modify: `tests/test-server-sched.cpp`

**Interfaces:**
- Consumes: `sched_states`, `sched_demand_fn` (Task 1).
- Produces: `sched_state_it sched_pick_victim(const sched_states & states, const sched_demand_fn & has_demand)`. Returns `states.end()` when every entry is protected — the **caller** decides the fallback, this function does not.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test-server-sched.cpp`:

```cpp
// Tests 5-7: sched_pick_victim
static void test_pick_victim(testing & t) {
    t.test("returns_lru_front_when_no_demand", [&](testing & t) {
        sched_states states;
        states.push_back(make_state(run(1, 100)));      // oldest
        states.push_back(make_state(run(2000, 100)));

        const auto it = sched_pick_victim(states, [](const server_prompt_cache_state &) {
            return false;
        });

        t.assert_true("victim is the front entry", it == states.cbegin());
    });

    t.test("skips_a_protected_entry_and_takes_the_next", [&](testing & t) {
        sched_states states;
        states.push_back(make_state(run(1, 100)));      // oldest, protected
        states.push_back(make_state(run(2000, 100)));   // expected victim

        const auto * protected_entry = &states.front();

        const auto it = sched_pick_victim(states, [&](const server_prompt_cache_state & st) {
            return &st == protected_entry;
        });

        t.assert_true("victim is the second entry", it == std::next(states.cbegin()));
    });

    t.test("returns_end_when_every_entry_is_protected", [&](testing & t) {
        sched_states states;
        states.push_back(make_state(run(1, 100)));
        states.push_back(make_state(run(2000, 100)));

        const auto it = sched_pick_victim(states, [](const server_prompt_cache_state &) {
            return true;
        });

        t.assert_true("no unprotected victim", it == states.cend());
    });
}
```

Register in `main`:

```cpp
    t.test("pick_victim", test_pick_victim);
```

- [ ] **Step 2: Run to verify it fails**

Run:
```bash
cmake --build build --target test-server-sched -j"$(nproc)"
```
Expected: **FAIL** — `error: 'sched_pick_victim' was not declared in this scope`.

- [ ] **Step 3: Implement**

Add to `tools/server/server-sched.h`:

```cpp
// Choose an eviction victim in LRU order (front = oldest), skipping entries that a queued
// task needs. Returns states.end() when every entry is protected; the caller decides what
// to do about that.
sched_state_it sched_pick_victim(const sched_states & states,
                                 const sched_demand_fn & has_demand);
```

Add to `tools/server/server-sched.cpp`:

```cpp
sched_state_it sched_pick_victim(const sched_states & states,
                                 const sched_demand_fn & has_demand) {
    for (auto it = states.cbegin(); it != states.cend(); ++it) {
        if (has_demand && has_demand(*it)) {
            continue;
        }

        return it;
    }

    return states.cend();
}
```

- [ ] **Step 4: Run to verify it passes**

Run:
```bash
cmake --build build --target test-server-sched -j"$(nproc)" && ./build/bin/test-server-sched
```
Expected: **PASS**, 7 tests, 7 assertions, 0 failures.

- [ ] **Step 5: Commit**

```bash
git add tools/server/server-sched.h tools/server/server-sched.cpp tests/test-server-sched.cpp
git commit -m "server: add sched_pick_victim honouring queued demand

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `sched_boilerplate_len` and `sched_pick_restore`

**Files:**
- Modify: `tools/server/server-sched.h`
- Modify: `tools/server/server-sched.cpp`
- Modify: `tests/test-server-sched.cpp`

**Interfaces:**
- Consumes: `sched_states`, `sched_thresholds` (Task 1).
- Produces:
  - `size_t sched_boilerplate_len(const sched_states & states, const server_tokens & tokens_new)`
  - `sched_state_it sched_pick_restore(const sched_states & states, const server_tokens & tokens_new, const server_tokens & slot_prompt, const sched_thresholds & th)`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test-server-sched.cpp`:

```cpp
// Tests 8-11: sched_pick_restore
static void test_pick_restore(testing & t) {
    const sched_thresholds th;

    t.test("accepts_a_genuine_deep_continuation", [&](testing & t) {
        const auto boiler = run(1000, 20000);
        const auto convo  = run(50000, 30000);

        sched_states states;
        states.push_back(make_state(cat(boiler, convo)));                 // the real history
        states.push_back(make_state(cat(boiler, run(90000, 5000))));      // unrelated

        // next turn: same history plus a tool result
        server_tokens tokens_new(cat(cat(boiler, convo), run(200000, 400)), false);
        server_tokens slot_prompt;

        const auto it = sched_pick_restore(states, tokens_new, slot_prompt, th);

        t.assert_true("selects the matching history", it == states.cbegin());
    });

    t.test("rejects_a_boilerplate_only_match", [&](testing & t) {
        const auto boiler = run(1000, 20000);

        sched_states states;
        states.push_back(make_state(cat(boiler, run(90000, 5000))));      // unrelated tail
        states.push_back(make_state(cat(boiler, run(95000, 6000))));      // unrelated tail

        server_tokens tokens_new(cat(boiler, run(50000, 30000)), false);
        server_tokens slot_prompt;

        const auto it = sched_pick_restore(states, tokens_new, slot_prompt, th);

        t.assert_true("nothing is distinctive enough to restore", it == states.cend());
    });

    t.test("respects_absolute_floor_when_boilerplate_len_collapses", [&](testing & t) {
        // Single state, so boilerplate_len is derived from one sample and is near zero.
        // A short coincidental match must still be rejected by absolute_floor.
        sched_states states;
        states.push_back(make_state(cat(run(1, 100), run(90000, 300))));

        server_tokens tokens_new(cat(run(1, 100), run(50000, 30000)), false);
        server_tokens slot_prompt;

        const auto it = sched_pick_restore(states, tokens_new, slot_prompt, th);

        t.assert_true("100-token match is below absolute_floor", it == states.cend());
    });

    t.test("maximises_lcp_and_is_order_independent", [&](testing & t) {
        const auto boiler = run(1000, 20000);
        const auto deep   = cat(boiler, run(50000, 30000));   // lcp 50000 with tokens_new
        const auto mid    = cat(boiler, run(50000, 8000));    // lcp 28000 with tokens_new

        server_tokens tokens_new(cat(deep, run(200000, 400)), false);

        // forward order
        sched_states a;
        a.push_back(make_state(mid));
        a.push_back(make_state(deep));
        server_tokens slot_a;
        const auto it_a = sched_pick_restore(a, tokens_new, slot_a, th);
        const size_t lcp_a = it_a == a.cend() ? 0
                           : (size_t) it_a->prompt.tokens.get_common_prefix(tokens_new);

        // reversed order
        sched_states b;
        b.push_back(make_state(deep));
        b.push_back(make_state(mid));
        server_tokens slot_b;
        const auto it_b = sched_pick_restore(b, tokens_new, slot_b, th);
        const size_t lcp_b = it_b == b.cend() ? 0
                           : (size_t) it_b->prompt.tokens.get_common_prefix(tokens_new);

        t.assert_equal("picks the deepest match", (size_t) 50000, lcp_a);
        t.assert_equal("order does not change the winner", lcp_a, lcp_b);
    });
}
```

Register in `main`:

```cpp
    t.test("pick_restore", test_pick_restore);
```

- [ ] **Step 2: Run to verify it fails**

Run:
```bash
cmake --build build --target test-server-sched -j"$(nproc)"
```
Expected: **FAIL** — `error: 'sched_pick_restore' was not declared in this scope`.

- [ ] **Step 3: Implement**

Add to `tools/server/server-sched.h`:

```cpp
// Length of the prefix shared by every resident entry and the incoming prompt. In practice
// this is the deployment's system-prompt-plus-tool-schema preamble. Returns 0 for fewer
// than two states, since one sample cannot establish what is common.
size_t sched_boilerplate_len(const sched_states & states, const server_tokens & tokens_new);

// Choose an entry to restore. Requires the match to be distinctive: deeper than shared
// boilerplate by `distinctive_margin`, at least `absolute_floor` tokens, and keeping at
// least `f_keep_min` of the cached prompt. Maximises absolute lcp, so the result does not
// depend on iteration order. Returns states.end() when nothing qualifies.
sched_state_it sched_pick_restore(const sched_states  & states,
                                 const server_tokens & tokens_new,
                                 const server_tokens & slot_prompt,
                                 const sched_thresholds & th,
                                 sched_restore_trace * trace = nullptr);
```

Add to `tools/server/server-sched.cpp`:

```cpp
size_t sched_boilerplate_len(const sched_states & states, const server_tokens & tokens_new) {
    if (states.size() < 2) {
        return 0;
    }

    size_t common = SIZE_MAX;

    for (const auto & st : states) {
        common = std::min(common, (size_t) st.prompt.tokens.get_common_prefix(tokens_new));
    }

    return common == SIZE_MAX ? 0 : common;
}

sched_state_it sched_pick_restore(const sched_states  & states,
                                  const server_tokens & tokens_new,
                                  const server_tokens & slot_prompt,
                                  const sched_thresholds & th,
                                  sched_restore_trace * trace) {
    const size_t boiler = sched_boilerplate_len(states, tokens_new);

    // a candidate must beat what the slot already holds
    size_t best_lcp = slot_prompt.get_common_prefix(tokens_new);
    auto   it_best  = states.cend();

    if (trace) {
        // same baseline semantics load() has always logged
        trace->base_f_keep = slot_prompt.size() > 0
            ? float(best_lcp) / slot_prompt.size() : -1.0f;
        trace->base_f_sim  = tokens_new.size() > 0
            ? float(best_lcp) / tokens_new.size() : 0.0f;
        trace->candidates.clear();
        trace->found = false;
    }

    for (auto it = states.cbegin(); it != states.cend(); ++it) {
        const size_t lcp         = it->prompt.tokens.get_common_prefix(tokens_new);
        const size_t cached_size = it->prompt.tokens.size();

        const float f_keep = cached_size   > 0 ? float(lcp) / cached_size   : 0.0f;
        const float f_sim  = tokens_new.size() > 0 ? float(lcp) / tokens_new.size() : 0.0f;

        // capture before any guard, so a skipped candidate still produces a line
        if (trace) {
            trace->candidates.push_back({ lcp, f_keep, f_sim });
        }

        // don't trash large cached prompts
        if (cached_size == 0 || f_keep < th.f_keep_min) {
            continue;
        }

        // the match must be distinctive, not merely template-deep
        if (lcp < th.absolute_floor) {
            continue;
        }
        if (lcp < boiler + th.distinctive_margin) {
            continue;
        }

        // single well-defined key: maximise absolute tokens saved
        if (lcp > best_lcp) {
            best_lcp = lcp;
            it_best  = it;

            if (trace) {
                trace->found       = true;
                trace->best_f_keep = f_keep;
                trace->best_f_sim  = f_sim;
            }
        }
    }

    return it_best;
}
```

Add `#include <cstdint>` to `tools/server/server-sched.cpp` for `SIZE_MAX`.

- [ ] **Step 4: Run to verify it passes**

Run:
```bash
cmake --build build --target test-server-sched -j"$(nproc)" && ./build/bin/test-server-sched
```
Expected: **PASS**, 11 tests, 12 assertions, 0 failures.

- [ ] **Step 5: Commit**

```bash
git add tools/server/server-sched.h tools/server/server-sched.cpp tests/test-server-sched.cpp
git commit -m "server: add sched_pick_restore requiring a distinctive prefix match

Replaces the baseline's non-total-order conjunction with a single key (maximise
absolute lcp), and requires a match to exceed shared boilerplate. Order-independent.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Register the flag

**Files:**
- Modify: `common/common.h:613` (beside `cache_idle_slots`)
- Modify: `common/arg.cpp:1721-1727` (beside `--cache-idle-slots`)

**Interfaces:**
- Produces: `common_params::cache_aware_sched` (bool, default `false`).

- [ ] **Step 1: Add the param**

In `common/common.h`, immediately after line 613 (`bool cache_idle_slots = true;`):

```cpp
    bool    cache_aware_sched   = false; // rank queued tasks by resident prefix, protect needed cache entries
```

- [ ] **Step 2: Register the flag**

In `common/arg.cpp`, immediately after the `--cache-idle-slots` block ending at line 1727:

```cpp
    add_opt(common_arg(
        {"--cache-aware-sched"},
        {"--no-cache-aware-sched"},
        "serve the queued request with the most already-resident prompt first, and avoid "
        "evicting prompt cache entries that a queued request needs. Raises throughput under "
        "concurrent multi-turn load at the cost of tail latency: a request with no resident "
        "prefix may be deferred indefinitely while warmer requests keep arriving. "
        "(default: disabled, requires cache-ram)",
        [](common_params & params, bool value) {
            params.cache_aware_sched = value;
        }
    ).set_env("LLAMA_ARG_CACHE_AWARE_SCHED").set_examples({LLAMA_EXAMPLE_SERVER}));
```

- [ ] **Step 3: Verify the flag parses and is documented**

Run:
```bash
cmake --build build --target llama-server -j"$(nproc)"
./build/bin/llama-server --help 2>&1 | grep -A6 'cache-aware-sched'
```
Expected: the flag and its help text appear, including the starvation warning.

- [ ] **Step 4: Commit**

```bash
git add common/common.h common/arg.cpp
git commit -m "server: add --cache-aware-sched flag, default off

Help text states the tail-latency trade explicitly at the point of opt-in.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Wire demand-aware eviction and install both callbacks

**Files:**
- Modify: `tools/server/server-task.h:612-635` (`server_prompt_cache`)
- Modify: `tools/server/server-task.cpp:1750-1760` (the eviction loop)
- Modify: `tools/server/server-context.h` (`prompt_save`, around line 250)
- Modify: `tools/server/server-queue.h` (add `callback_score_task`, `set_score_task`, `for_each_deferred`)
- Modify: `tools/server/server-queue.cpp:90-113` (`pop_deferred_task`)
- Modify: `tools/server/server-context.cpp:1543`, `:2314` (call sites), and startup wiring

**Interfaces:**
- Consumes: `sched_score` (Task 2), `sched_pick_victim` (Task 3), `sched_demand_fn` (Task 1), `params_base.cache_aware_sched` (Task 5).
- Produces: `server_prompt_cache::alloc(const server_prompt &, size_t, size_t, const sched_demand_fn &)`; `server_prompt_cache::cache_aware_sched` (bool member); `server_slot::prompt_save(server_prompt_cache &, const sched_demand_fn &)`; `server_queue::set_score_task(...)`; `server_queue::for_each_deferred(...)`.

The signature changes and their call sites must land together — a tree that does not compile
cannot be reviewed, so this is one task with one commit.

- [ ] **Step 1: Change the signatures**

In `tools/server/server-task.h`, add to `struct server_prompt_cache` after `limit_tokens`:

```cpp
    // set from params_base.cache_aware_sched at construction
    bool cache_aware_sched = false;
```

and change the `alloc` declaration to:

```cpp
    server_prompt_cache_state * alloc(const server_prompt & prompt,
                                      size_t state_size_main,
                                      size_t state_size_drft,
                                      const sched_demand_fn & has_demand);
```

Add `#include "server-sched.h"` — **not** to `server-task.h` (that would be circular, since
`server-sched.h` includes `server-task.h`). Instead declare the alias locally in
`server-task.h` above `server_prompt_cache`:

```cpp
struct server_prompt_cache_state;
using sched_demand_fn = std::function<bool(const server_prompt_cache_state &)>;
```

and **delete** the duplicate `using sched_demand_fn = ...` from `server-sched.h`, keeping
only the one in `server-task.h`. Add `#include <functional>` to `server-task.h`.

- [ ] **Step 2: Replace the eviction loop**

In `tools/server/server-task.cpp`, replace lines 1750-1760 (the `if (limit_size > 0) { while (...) { ... states.pop_front(); } }` block) with:

```cpp
    if (limit_size > 0) {
        // make room before allocating the new vectors to avoid breaching the limit
        while (!states.empty() && size() + state_size_new > limit_size) {
            auto victim = states.cbegin();

            if (cache_aware_sched) {
                const auto pick = sched_pick_victim(states, has_demand);

                if (pick == states.cend()) {
                    // Every entry is needed by a queued task. Protection is advisory:
                    // fall back to LRU, because refusing to evict would deny the running
                    // slot its own state save and merely move the loss elsewhere.
                    SRV_WRN("%s", " - every prompt cache entry is needed by a queued task, "
                                  "falling back to LRU eviction\n");
                    SRV_DBG("%s", "__TEST_TAG_SCHED_PROTECT_FALLBACK__\n");
                } else {
                    victim = pick;
                    SRV_DBG("%s", "__TEST_TAG_SCHED_VICTIM_PROTECTED__\n");
                }
            }

            SRV_WRN(" - making room for prompt cache entry, removing oldest entry (size = %.3f MiB)\n",
                    victim->size() / (1024.0 * 1024.0));

            states.erase(victim);
        }
```

Add `#include "server-sched.h"` to `tools/server/server-task.cpp` if Task 1 did not already.

- [ ] **Step 3: Thread the predicate through `prompt_save`**

In `tools/server/server-context.h`, change `prompt_save` (around line 250) to take and
forward the predicate:

```cpp
    bool prompt_save(server_prompt_cache & prompt_cache, const sched_demand_fn & has_demand) const {
```

and its `alloc` call to:

```cpp
        auto * cur = prompt_cache.alloc(prompt, cur_size_tgt, cur_size_dft, has_demand);
```

- [ ] **Step 4: Build and verify the two existing call sites compile**

Run:
```bash
cmake --build build --target llama-server -j"$(nproc)" 2>&1 | grep -E 'error|prompt_save' | head
```
Expected: errors at the two `prompt_save` call sites (`server-context.cpp:2314` and
`:1543`). Those are fixed in Step 7 below — this step only confirms which sites need updating.

The second half of this task is where the Critical Pitfall applies. Demand is **argmax**,
computed once per `alloc()`, not a per-entry threshold.

- [ ] **Step 5: Expose the queue hooks**

In `tools/server/server-queue.h`, add to the private callback block:

```cpp
    std::function<size_t(const server_task &, int)> callback_score_task;
```

and to the public section:

```cpp
    // Rank deferred tasks by how much resident prompt they could reuse on a given slot.
    // Must be installed before start_loop(). The callback runs while mutex_tasks is held
    // and on the start_loop thread, so it MUST NOT call back into server_queue.
    void set_score_task(std::function<size_t(const server_task &, int)> cb) {
        callback_score_task = std::move(cb);
    }

    // Visit each deferred task's tokens under the queue lock.
    void for_each_deferred(const std::function<void(const server_tokens &)> & fn);
```

- [ ] **Step 10: Implement `for_each_deferred` and rank in `pop_deferred_task`**

Add to `tools/server/server-queue.cpp`:

```cpp
void server_queue::for_each_deferred(const std::function<void(const server_tokens &)> & fn) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    for (const auto & task : queue_tasks_deferred) {
        fn(task.tokens);
    }
}
```

In `pop_deferred_task`, replace the `if (!found) { ... }` block at lines 105-109 with:

```cpp
        // rank by resident prefix when cache-aware scheduling is enabled
        if (!found && callback_score_task) {
            auto   best       = queue_tasks_deferred.begin();
            size_t best_score = callback_score_task(*best, id_slot);

            for (auto it = std::next(queue_tasks_deferred.begin()); it != queue_tasks_deferred.end(); ++it) {
                const size_t score = callback_score_task(*it, id_slot);

                // strict > keeps FIFO order among equal scores
                if (score > best_score) {
                    best_score = score;
                    best       = it;
                }
            }

            QUE_DBG("pop deferred task (score %zu), id_task = %d\n", best_score, best->id);
            QUE_DBG("%s", "__TEST_TAG_SCHED_POP_BY_SCORE__\n");
            queue_tasks.emplace_front(std::move(*best));
            queue_tasks_deferred.erase(best);
            found = true;
        }

        // if no task found using the slot, just pop the first deferred task (default behavior)
        if (!found) {
            QUE_DBG("pop deferred task, id_task = %d\n", queue_tasks_deferred.front().id);
            queue_tasks.emplace_front(std::move(queue_tasks_deferred.front()));
            queue_tasks_deferred.pop_front();
        }
```

- [ ] **Step 11: Build the protected set and fix the call sites**

In `tools/server/server-context.cpp`, add a member function to `server_context`:

```cpp
    // An entry is protected iff it is the argmax candidate for at least one deferred task.
    // NOT a threshold: every conversation here shares a long system-prompt preamble, so a
    // threshold predicate would protect every entry and degrade to permanent LRU fallback.
    sched_demand_fn make_demand_fn() {
        if (!params_base.cache_aware_sched || !prompt_cache) {
            return {};
        }

        auto protected_set = std::make_shared<std::unordered_set<const server_prompt_cache_state *>>();

        queue_tasks.for_each_deferred([&](const server_tokens & tokens) {
            const server_prompt_cache_state * best = nullptr;
            size_t best_lcp = 0;

            for (const auto & st : prompt_cache->states) {
                const size_t lcp = st.prompt.tokens.get_common_prefix(tokens);
                if (lcp > best_lcp) {
                    best_lcp = lcp;
                    best     = &st;
                }
            }

            if (best != nullptr) {
                protected_set->insert(best);
            }
        });

        return [protected_set](const server_prompt_cache_state & st) {
            return protected_set->count(&st) > 0;
        };
    }
```

Add `#include <unordered_set>` and `#include <memory>` to `tools/server/server-context.cpp`.

Update both `prompt_save` call sites to pass it — `server-context.cpp:1543`:

```cpp
                ret->prompt_save(*prompt_cache, make_demand_fn());
```

and `server-context.cpp:2314`:

```cpp
                            if (slot.prompt_save(*prompt_cache, make_demand_fn())) {
```

- [ ] **Step 8: Install the flag and the scoring callback at startup**

In `tools/server/server-context.cpp`, where `prompt_cache` is constructed, set the flag:

```cpp
        prompt_cache->cache_aware_sched = params_base.cache_aware_sched;
```

Next to the existing `--cache-idle-slots` validation at `server-context.cpp:1327-1341`, add:

```cpp
        if (params_base.cache_aware_sched) {
            if (params_base.cache_ram_mib == 0) {
                SRV_WRN("%s", "--cache-aware-sched requires --cache-ram, disabling\n");
                params_base.cache_aware_sched = false;
            } else {
                SRV_INF("%s", "cache-aware scheduling enabled: queued requests are ranked by "
                              "resident prefix; requests with no resident prefix may be deferred "
                              "indefinitely under sustained load\n");
                SRV_DBG("%s", "__TEST_TAG_CACHE_AWARE_SCHED_ENABLED__\n");
            }
        }
```

Where the other queue callbacks are installed (search for `queue_tasks.on_new_task`), add:

```cpp
    if (params_base.cache_aware_sched) {
        queue_tasks.set_score_task([this](const server_task & task, int id_slot) {
            const server_tokens * slot_prompt = nullptr;
            for (const auto & slot : slots) {
                if (slot.id == id_slot) {
                    slot_prompt = &slot.prompt.tokens;
                    break;
                }
            }

            static const server_tokens empty;
            return sched_score(task.tokens,
                               slot_prompt ? *slot_prompt : empty,
                               prompt_cache ? prompt_cache->states : sched_states{});
        });
    }
```

- [ ] **Step 9: Build and run the unit tests**

Run:
```bash
cmake --build build --target llama-server test-server-sched -j"$(nproc)"
./build/bin/test-server-sched
```
Expected: build succeeds with no errors; 11 tests, 12 assertions, 0 failures.

- [ ] **Step 10: Verify the flag off path is unchanged**

Run:
```bash
cd tools/server/tests && ./tests.sh unit/test_completion.py unit/test_chat_completion.py -v ; cd -
```
Expected: PASS. No flag is set, so `callback_score_task` is never installed and
`cache_aware_sched` is false throughout.

- [ ] **Step 11: Commit**

```bash
git add tools/server/server-task.h tools/server/server-task.cpp \
        tools/server/server-queue.h tools/server/server-queue.cpp \
        tools/server/server-context.h tools/server/server-context.cpp
git commit -m "server: wire cache-aware eviction and queue ranking behind the flag

Demand is argmax, not a threshold: every conversation shares a long system-prompt
preamble, so a threshold would protect every entry and silently degrade to
permanent LRU fallback.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6 post-review amendments

Applied after Task 6's review. These supersede the code blocks above where they conflict.

1. **Install the score callback after the flag validation, not before.** The auto-disable
   (`cache_ram_mib == 0` -> `cache_aware_sched = false`) must run first, or
   `--cache-aware-sched --cache-ram 0` leaves the callback installed and keeps ranking while
   the log says the feature is disabled.

2. **The scoring pass considers only unpinned tasks (`id_slot == -1`).** A task pinned to a
   busy slot can be the argmax when a *different* slot frees; it is then picked, re-deferred
   by `process_single_task`, and picked again because it is still the argmax, so the freeing
   slot can spin without running anything. Baseline FIFO rotated it to the back and made
   progress. Tasks pinned to the slot being filled are already handled by the explicit-slot
   precedence check that runs before scoring, so restricting the scoring candidates to
   `id_slot == -1` is both correct and sufficient.

3. **Hoist `make_demand_fn()` out of the per-idle-slot loop.** Each rebuild takes
   `mutex_tasks` via `for_each_deferred` and does O(deferred x states) prefix comparisons;
   once before the loop is equivalent and shortens lock hold time.

4. **`__TEST_TAG_SCHED_VICTIM_PROTECTED__` fires only when protection changed the outcome** —
   when the chosen victim differs from the LRU front. Emitting it on every successful
   demand-aware pick makes it useless as evidence that protection was exercised, which is how
   Task 9 would otherwise read it.

5. Remove the dead re-forward-declaration of `server_prompt_cache_state` in
   `tools/server/server-task.h`; the full definition already precedes it.

6. **`server_prompt_cache::update()` must be demand-aware too.** It has its own strict-LRU
   `pop_front()` trim loops and runs right after every `prompt_save`
   (`server-context.cpp:1614` and `:2382`). Its token loop is always live — the cache is
   constructed `server_prompt_cache(cache_ram_mib, n_ctx)` so `limit_tokens` is never zero —
   and its dynamic limit derives from the same budget `alloc()` enforces. Use
   `sched_pick_victim` with the same advisory-LRU fallback in both of `update()`'s loops.
   Without this, `alloc()` protects an entry and `update()` evicts it moments later.

7. **Compute the demand predicate once per eviction call, not once per caller loop.**
   Amendment 3 hoisted it too far: the protected set is keyed on element addresses, and
   `alloc()`/`update()` erase *and* `push_back` inside the caller's per-slot loop, so a new
   entry can land on a freed node's address and be misread as protected. Per-call is safe
   because no insertion happens inside a single call's victim loop. This supersedes
   amendment 3.

---

### Task 8: Use the tightened restore selection under the flag

**Files:**
- Modify: `tools/server/server-task.cpp` (`server_prompt_cache::load()`)

**Interfaces:**
- Consumes: `sched_pick_restore` (Task 4), `server_prompt_cache::cache_aware_sched` (Task 6).

- [ ] **Step 1: Switch selection on the flag**

In `server_prompt_cache::load()`, replace **only** the `sched_pick_restore_baseline` call
(the line assigning `it_best_const`) with the version below. Leave the `sched_restore_trace
trace;` declaration, the three `SRV_TRC` emissions, and the `const_iterator` conversion that
follows them exactly as they are — they already work for both paths.

```cpp
    static const sched_thresholds th;

    // `trace` is the local Task 1 already declared at the top of load(); both paths must
    // fill it or the three SRV_TRC diagnostics below it print zeros.
    auto it_best_const = cache_aware_sched
        ? sched_pick_restore(states, tokens_new, prompt.tokens, th, &trace)
        : sched_pick_restore_baseline(states, tokens_new, prompt.tokens, &trace);

    if (cache_aware_sched && it_best_const == states.cend()) {
        SRV_DBG("%s", "__TEST_TAG_SCHED_RESTORE_REJECTED__\n");
    }
```

- [ ] **Step 2: Build and run all unit tests**

Run:
```bash
cmake --build build --target llama-server test-server-sched -j"$(nproc)" && ./build/bin/test-server-sched
```
Expected: PASS, 11 tests, 12 assertions, 0 failures. The characterization test still passes
because it calls `sched_pick_restore_baseline` directly.

- [ ] **Step 3: Commit**

```bash
git add tools/server/server-task.cpp
git commit -m "server: use the distinctive-match restore test under --cache-aware-sched

Containment for #27148, not a fix: demand-aware eviction keeps entries resident
longer, which would otherwise increase exposure to the boilerplate-only match.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Integration tests

**Files:**
- Create: `tools/server/tests/unit/test_cache_aware_sched.py`

**Interfaces:**
- Consumes: the debug markers emitted in Tasks 6 and 8 (`__TEST_TAG_SCHED_VICTIM_PROTECTED__`, `__TEST_TAG_SCHED_PROTECT_FALLBACK__`, `__TEST_TAG_SCHED_POP_BY_SCORE__`, `__TEST_TAG_SCHED_RESTORE_REJECTED__`, `__TEST_TAG_CACHE_AWARE_SCHED_ENABLED__`).

**Prerequisite:** add the server knob. In `tools/server/tests/utils.py`, add beside
`cache_ram` (line 113):

```python
    cache_aware_sched: bool | None = None
```

and beside its arg emission (lines 272-273):

```python
        if self.cache_aware_sched:
            server_args.append("--cache-aware-sched")
```

- [ ] **Step 1: Write the four failing tests**

Create `tools/server/tests/unit/test_cache_aware_sched.py`. This follows
`tools/server/tests/unit/test_kv_keep_only_active.py`, which is the existing
marker-assertion template — same `LogReader`, same `debug = True` plus `log_path` fixture:

```python
import os
import tempfile
import pytest
from utils import *

server = ServerPreset.tinyllama2()


class LogReader:
    def __init__(self, path):
        self.path = path
        self.pos = 0
    def drain(self):
        with open(self.path) as f:
            f.seek(self.pos)
            content = f.read()
            self.pos = f.tell()
        return content


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 1
    server.n_predict = 4
    server.temperature = 0.0
    server.cache_ram = 1          # MiB: room for roughly one conversation
    server.kv_unified = True
    server.debug = True
    fd, server.log_path = tempfile.mkstemp(suffix='.log')
    os.close(fd)
    yield


CONVO_A = (
    "Once upon a time in a land far away there lived a brave knight "
    "who traveled across mountains and rivers to find the golden sword "
    "hidden deep within the enchanted forest of whispers and shadows."
)

CONVO_B = (
    "The quick brown fox jumped over the lazy dog again and again while "
    "the farmer watched from his porch and drank his morning coffee slowly "
    "considering whether to mend the fence before the rain arrived."
)


def test_flag_off_matches_baseline_ordering():
    global server
    server.start()
    log = LogReader(server.log_path)

    assert "__TEST_TAG_CACHE_AWARE_SCHED_ENABLED__" not in log.drain()

    for prompt in (CONVO_A, CONVO_B, CONVO_A + " And then"):
        res = server.make_request("POST", "/completion", data={
            "prompt": prompt, "cache_prompt": True,
        })
        assert res.status_code == 200

    assert "__TEST_TAG_SCHED_POP_BY_SCORE__" not in log.drain()


def test_warm_request_served_before_cold():
    global server
    server.cache_aware_sched = True
    server.start()
    log = LogReader(server.log_path)

    assert "__TEST_TAG_CACHE_AWARE_SCHED_ENABLED__" in log.drain()

    # seed A so its prefix is resident
    res = server.make_request("POST", "/completion", data={
        "prompt": CONVO_A, "cache_prompt": True,
    })
    assert res.status_code == 200

    # queue a cold request and a warm continuation of A concurrently.
    # parallel_function_calls (utils.py:649) takes (callable, args_tuple) pairs.
    results = parallel_function_calls([
        (server.make_request, ("POST", "/completion",
                               {"prompt": CONVO_B, "cache_prompt": True})),
        (server.make_request, ("POST", "/completion",
                               {"prompt": CONVO_A + " And then", "cache_prompt": True})),
    ])
    assert all(r.status_code == 200 for r in results)

    assert "__TEST_TAG_SCHED_POP_BY_SCORE__" in log.drain()


def test_all_victims_protected_falls_back_to_lru():
    global server
    server.cache_aware_sched = True
    server.start()
    log = LogReader(server.log_path)

    # every resident entry is the argmax of some queued task, and cache_ram is tiny
    prompts = [CONVO_A, CONVO_B, CONVO_A + " And then", CONVO_B + " Meanwhile"]
    results = parallel_function_calls([
        (server.make_request, ("POST", "/completion", {"prompt": pr, "cache_prompt": True}))
        for pr in prompts
    ])
    assert all(r.status_code == 200 for r in results)

    drained = log.drain()
    assert "__TEST_TAG_SCHED_PROTECT_FALLBACK__" in drained


def test_explicit_slot_binding_beats_score():
    global server
    server.n_slots = 2
    server.cache_aware_sched = True
    server.start()
    log = LogReader(server.log_path)

    # make slot 1's prefix resident, then contend: an explicitly-bound task with a poor
    # score must still be popped ahead of a higher-scoring unbound task
    res = server.make_request("POST", "/completion", data={
        "prompt": CONVO_A, "id_slot": 1, "cache_prompt": True,
    })
    assert res.status_code == 200

    results = parallel_function_calls([
        (server.make_request, ("POST", "/completion",
                               {"prompt": CONVO_B, "id_slot": 1, "cache_prompt": True})),
        (server.make_request, ("POST", "/completion",
                               {"prompt": CONVO_A + " And then", "cache_prompt": True})),
    ])
    assert all(r.status_code == 200 for r in results)

    assert "pop deferred task (use slot 1)" in log.drain()
```

`parallel_function_calls` is defined at `tools/server/tests/utils.py:649` with signature
`List[Tuple[Callable[..., Any], Tuple[Any, ...]]]` — pass `(callable, args_tuple)` pairs, not
thunks. Do not add a new concurrency helper.

- [ ] **Step 3: Run to verify they fail before the markers exist**

If Tasks 6 and 8 are already committed the markers exist and tests 2-4 should pass; test 1 is a
regression guard and must pass. To confirm the tests have teeth, temporarily revert the
`set_score_task` install from Task 6 Step 8, run, and observe tests 2 and 4 fail:

```bash
cd tools/server/tests && ./tests.sh unit/test_cache_aware_sched.py -v ; cd -
```
Expected with the install reverted: tests 2 and 4 FAIL. Restore the install afterwards.

- [ ] **Step 4: Run to verify they pass**

Run:
```bash
cd tools/server/tests && ./tests.sh unit/test_cache_aware_sched.py -v ; cd -
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/server/tests/unit/test_cache_aware_sched.py
git commit -m "server: integration tests for cache-aware scheduling wiring

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Discrete-event simulator over the real selection functions

Prefill and generation rates are parameters, so throughput is computable — no model, no GPU,
no HTTP, no wall-clock waiting. A 24h workload simulates in milliseconds. Crucially the
simulator links the **same** `sched_*` functions the server calls, so it exercises the real
algorithm rather than a reimplementation. This is only possible because Tasks 1-4 made them
pure; that design choice pays off here.

This replaces the trace-capture, replay-driver and log-analyser approach entirely.

**Files:**
- Create: `tests/test-server-sched-sim.cpp`
- Modify: `tests/CMakeLists.txt` (register, link `server-context`)

**Interfaces:**
- Consumes: `sched_score`, `sched_pick_task`, `sched_pick_victim`, `sched_pick_restore` (Tasks 2-4).
- Produces: `sim_params`, `sim_result`, `sim_run(const sim_params &, bool cache_aware)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test-server-sched-sim.cpp`:

```cpp
#include "testing.h"
#include "server-sched.h"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <list>
#include <vector>

// Whether turn k+1's prompt strictly extends turn k's is decided by the chat template and
// the client, not by the server. It is therefore a first-class dimension of the workload:
// it determines whether prefix reuse exists at all. Model it generically as a *rewrite
// policy* -- which previously-rendered positions get re-rendered, and when.
//
// Qwen3 is one instance, not the model: its templates gate reasoning on
// `preserve_thinking or loop.index0 > last_query_index`
// (froggeric/Qwen-Fixed-Chat-Templates chat_template.jinja:298), where last_query_index is
// the last *genuine* user message -- user messages rendered as <tool_response> are skipped
// (lines 196-205). That yields APPEND_ONLY when preserve_thinking is true (the default) and
// DROP_REASONING_PRIOR_TURNS when it is false.
enum sim_history_policy {
    // Nothing is ever re-rendered. Best case for prefix caching. Non-reasoning templates,
    // and Qwen with preserve_thinking=true (what the measured 24h window ran).
    SIM_HIST_APPEND_ONLY,
    // Reasoning is kept for assistant messages in the current human turn and dropped for
    // earlier turns. Tool calls within a turn still extend exactly; each new human prompt
    // diverges at the previous turn's first assistant message. Qwen preserve_thinking=false.
    SIM_HIST_DROP_REASONING_PRIOR_TURNS,
    // Reasoning is stripped from every prior assistant message, including within the tool
    // loop. Diverges at the previous assistant message on every single request. Hostile.
    SIM_HIST_DROP_REASONING_ALWAYS,
    // Sliding window: once max_ctx_tokens is reached, the front is trimmed, so every
    // position shifts and the prefix match collapses to zero. Maximally hostile -- included
    // to prove the feature does no harm where it cannot help.
    SIM_HIST_TRUNCATE_FRONT,
    // History is replaced by a summary once compact_at_tokens is exceeded, then grows again.
    // Diverges at the summary insertion point. This is what omp's own compaction does.
    SIM_HIST_COMPACT_AT_THRESHOLD,
};

struct sim_params {
    size_t n_sessions       = 6;
    size_t n_turns          = 20;     // total requests per session
    size_t n_slots          = 1;
    size_t preamble_tokens  = 20000;
    size_t first_turn       = 40000;
    size_t growth_tokens    = 600;    // context growth per turn
    size_t gen_tokens       = 527;    // generated tokens per turn (median, measured)
    double think_s          = 0.0;    // agentic loop resubmits immediately
    double prefill_tok_s    = 900.0;  // measured ~765-1275 t/s
    double gen_tok_s        = 15.7;   // measured aggregate
    size_t kv_bytes_per_tok = 60u * 1024u;
    size_t cache_bytes      = 32ull * 1024 * 1024 * 1024;

    sim_history_policy history = SIM_HIST_APPEND_ONLY;
    size_t max_ctx_tokens      = 262144;  // for TRUNCATE_FRONT
    size_t compact_at_tokens   = 223000;  // for COMPACT_AT_THRESHOLD (omp's observed trigger)
    // Tool calls per human prompt. NOT measured -- the server log cannot distinguish a tool
    // response from a human prompt. Sweep it; do not present any value as observed.
    size_t tools_per_prompt = 8;
    // Share of generated tokens that is reasoning. Also NOT measured. Sweep it.
    double reasoning_frac   = 0.7;
};

struct sim_result {
    double makespan_s      = 0.0;
    size_t prefill_tokens  = 0;
    size_t gen_tokens      = 0;
    size_t n_requests      = 0;
    size_t n_high_reuse    = 0;   // >=95% of prompt reused
    size_t n_lost          = 0;   // >=30k prompt, <5% reused
    double wait_p99_cold_s = 0.0;
};

sim_result sim_run(const sim_params & p, bool cache_aware);

// Test: with the cache too small for the working set, cache-aware scheduling must
// re-prefill strictly fewer tokens and finish sooner than baseline.
// The rendering model must be self-consistent: the measured common prefix between
// consecutive renders must equal the divergence position the policy declares.
static void test_sim_render_is_self_consistent(testing & t) {
    for (auto pol : {SIM_HIST_APPEND_ONLY, SIM_HIST_DROP_REASONING_PRIOR_TURNS,
                     SIM_HIST_DROP_REASONING_ALWAYS, SIM_HIST_TRUNCATE_FRONT,
                     SIM_HIST_COMPACT_AT_THRESHOLD}) {
        t.test("policy_" + std::to_string((int) pol), [&](testing & t) {
            sim_params p;
            p.history = pol;
            sim_session s;
            s.id = 3;
            for (size_t k = 1; k < 12; k++) {
                const server_tokens a(sim_prompt(p, s, k - 1), false);
                const server_tokens b(sim_prompt(p, s, k), false);
                const size_t expect = std::min(sim_render_at(p, k).diverge_at,
                                               (size_t) a.size());
                t.assert_equal("common prefix equals declared divergence",
                               expect, (size_t) a.get_common_prefix(b));
            }
        });
    }
}

// The feature must help wherever prefix reuse exists, and must not hurt where it does not.
// A reviewer running a reasoning-stripping or truncating client must not see a regression.
static void test_sim_across_history_policies(testing & t) {
    struct expectation { sim_history_policy pol; const char * name; bool expect_gain; };

    const expectation cases[] = {
        {SIM_HIST_APPEND_ONLY,                "append_only",           true},
        {SIM_HIST_DROP_REASONING_PRIOR_TURNS, "drop_reasoning_prior",  true},
        {SIM_HIST_DROP_REASONING_ALWAYS,      "drop_reasoning_always", true},
        // nothing is reusable once the window slides, so the only requirement is no harm
        {SIM_HIST_TRUNCATE_FRONT,             "truncate_front",        false},
        {SIM_HIST_COMPACT_AT_THRESHOLD,       "compact_at_threshold",  true},
    };

    for (const auto & c : cases) {
        t.test(c.name, [&](testing & t) {
            sim_params p;
            p.history     = c.pol;
            p.cache_bytes = 8ull * 1024 * 1024 * 1024;

            const sim_result b = sim_run(p, false);
            const sim_result s = sim_run(p, true);

            if (c.expect_gain) {
                t.assert_true("fewer tokens re-prefilled", s.prefill_tokens < b.prefill_tokens);
                t.assert_true("shorter makespan", s.makespan_s < b.makespan_s);
            } else {
                // do no harm: allow 1% for tie-breaking differences, forbid real regression
                t.assert_true("no prefill regression",
                              s.prefill_tokens <= (size_t) (b.prefill_tokens * 1.01));
                t.assert_true("no makespan regression", s.makespan_s <= b.makespan_s * 1.01);
            }
        });
    }
}

static void test_sim_beats_baseline_under_pressure(testing & t) {static void test_sim_beats_baseline_under_pressure(testing & t) {
    t.test("fewer_reprefilled_tokens_and_shorter_makespan", [&](testing & t) {
        sim_params p;
        p.cache_bytes = 8ull * 1024 * 1024 * 1024;   // room for ~2 of 6 sessions

        const sim_result base = sim_run(p, false);
        const sim_result sched = sim_run(p, true);

        t.assert_true("same work offered", base.n_requests == sched.n_requests);
        t.assert_true("fewer tokens re-prefilled",
                      sched.prefill_tokens < base.prefill_tokens);
        t.assert_true("shorter makespan", sched.makespan_s < base.makespan_s);
        t.assert_true("more high-reuse requests",
                      sched.n_high_reuse > base.n_high_reuse);
    });
}

int main(int argc, char ** argv) {
    testing t;
    if (argc > 1) {
        t.set_filter(argv[1]);
    }
    t.test("render_consistency", test_sim_render_is_self_consistent);
    t.test("sim", test_sim_beats_baseline_under_pressure);
    t.test("history_policies", test_sim_across_history_policies);
    return t.summary();
}
```

Register in `tests/CMakeLists.txt` next to the Task 1 entry:

```cmake
    llama_build_and_test(test-server-sched-sim.cpp)
    target_link_libraries(test-server-sched-sim PRIVATE server-context)
    target_include_directories(test-server-sched-sim PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
```

- [ ] **Step 2: Run to verify it fails**

Run:
```bash
cd /home/claude-aiven-4/code/llama.cpp/.claude/worktrees/cache-aware-sched
cmake --build build --target test-server-sched-sim -j"$(nproc)"
```
Expected: **FAIL** — undefined reference to `sim_run`.

- [ ] **Step 3: Implement the simulator**

Add to `tests/test-server-sched-sim.cpp`, above `test_sim_beats_baseline_under_pressure`.
The event loop must call the real `sched_*` functions for every decision:

```cpp
namespace {

struct sim_session {
    size_t                   id      = 0;
    size_t                   turn    = 0;
    std::vector<llama_token> body;                 // divergent tail, session-unique
    double                   ready_at = 0.0;       // earliest send time
    bool                     pending  = false;     // queued, not yet launched
};

struct sim_slot {
    server_tokens prompt;
    double        free_at = 0.0;
};

// Rendered prompt geometry for a turn: total length, and the position at which this render
// diverges from the previous one. Everything before diverge_at is byte-identical to what the
// previous render produced, so it is exactly what a prefix cache can reuse.
struct sim_render {
    size_t len        = 0;
    size_t diverge_at = 0;
};

sim_render sim_render_at(const sim_params & p, size_t turn) {
    const size_t tpp        = std::max<size_t>(1, p.tools_per_prompt);
    const size_t human_turn = turn / tpp;
    const double keep       = 1.0 - p.reasoning_frac;

    // nominal appended length per request, before any rewriting
    auto nominal_len = [&](size_t k) {
        return p.preamble_tokens + p.first_turn + k * p.growth_tokens;
    };

    sim_render r;
    switch (p.history) {
        case SIM_HIST_APPEND_ONLY:
            r.len        = nominal_len(turn);
            r.diverge_at = turn == 0 ? 0 : nominal_len(turn - 1);
            break;

        case SIM_HIST_DROP_REASONING_PRIOR_TURNS: {
            // prior human turns contribute only their non-reasoning share
            const size_t prior = human_turn * tpp;
            const size_t in_turn = turn - prior;
            r.len = p.preamble_tokens + p.first_turn
                  + (size_t) (double(prior) * p.growth_tokens * keep)
                  + in_turn * p.growth_tokens;
            if (turn == 0) {
                r.diverge_at = 0;
            } else if (in_turn > 0) {
                // still inside the same human turn: pure append
                r.diverge_at = r.len - p.growth_tokens;
            } else {
                // new human prompt: the previous turn is re-rendered without reasoning
                const size_t prev_prior = (human_turn - 1) * tpp;
                r.diverge_at = p.preamble_tokens + p.first_turn
                             + (size_t) (double(prev_prior) * p.growth_tokens * keep);
            }
            break;
        }

        case SIM_HIST_DROP_REASONING_ALWAYS:
            r.len = p.preamble_tokens + p.first_turn
                  + (size_t) (double(turn) * p.growth_tokens * keep);
            // the immediately preceding assistant message is re-rendered every request
            r.diverge_at = turn == 0 ? 0
                : p.preamble_tokens + p.first_turn
                  + (size_t) (double(turn - 1) * p.growth_tokens * keep);
            break;

        case SIM_HIST_TRUNCATE_FRONT: {
            const size_t nominal = nominal_len(turn);
            if (nominal <= p.max_ctx_tokens) {
                r.len        = nominal;
                r.diverge_at = turn == 0 ? 0 : nominal_len(turn - 1);
            } else {
                r.len = p.max_ctx_tokens;
                // every position shifted: nothing before the end matches
                r.diverge_at = 0;
            }
            break;
        }

        case SIM_HIST_COMPACT_AT_THRESHOLD: {
            const size_t nominal = nominal_len(turn);
            if (nominal <= p.compact_at_tokens) {
                r.len        = nominal;
                r.diverge_at = turn == 0 ? 0 : nominal_len(turn - 1);
            } else {
                // history collapsed to a summary just after the preamble, then regrows
                const size_t cycle = (nominal - p.compact_at_tokens) / std::max<size_t>(1, p.growth_tokens);
                r.len        = p.preamble_tokens + 8192 + cycle * p.growth_tokens;
                r.diverge_at = cycle == 0 ? p.preamble_tokens : r.len - p.growth_tokens;
            }
            break;
        }
    }

    r.diverge_at = std::min(r.diverge_at, r.len);
    return r;
}

// Build the token list so that positions below diverge_at reproduce the previous render
// exactly and positions at or above it do not. Tag each position with the turn at which it
// was last rewritten; the token is a function of (session, position, that turn). This makes
// get_common_prefix(prompt(k), prompt(k-1)) == diverge_at by construction, rather than by
// arithmetic that has to be got right twice.
std::vector<llama_token> sim_prompt(const sim_params & p, const sim_session & s, size_t turn) {
    std::vector<size_t> gen(sim_render_at(p, turn).len, 0);
    for (size_t k = 1; k <= turn; k++) {
        const sim_render r = sim_render_at(p, k);
        for (size_t i = r.diverge_at; i < std::min(r.len, gen.size()); i++) {
            gen[i] = k;
        }
    }

    std::vector<llama_token> v;
    v.reserve(gen.size());
    for (size_t i = 0; i < gen.size(); i++) {
        if (i < p.preamble_tokens) {
            v.push_back((llama_token) (i + 1));       // preamble: identical for all sessions
        } else {
            v.push_back((llama_token) (1 + (s.id * 1000003 + i * 31 + gen[i] * 7919) % 900000));
        }
    }
    return v;
}

size_t entry_bytes(const sim_params & p, const server_prompt_cache_state & st) {
    return st.prompt.tokens.size() * p.kv_bytes_per_tok;
}

size_t cache_bytes_used(const sim_params & p, const sched_states & states) {
    size_t n = 0;
    for (const auto & st : states) {
        n += entry_bytes(p, st);
    }
    return n;
}

} // namespace

sim_result sim_run(const sim_params & p, bool cache_aware) {
    const sched_thresholds th;

    std::vector<sim_session> sessions(p.n_sessions);
    for (size_t i = 0; i < p.n_sessions; i++) {
        sessions[i].id = i;
        // unique body per session, diverging immediately after the preamble
        sessions[i].body.resize(4096);
        for (size_t j = 0; j < sessions[i].body.size(); j++) {
            sessions[i].body[j] = (llama_token) (1000000 + i * 100000 + j);
        }
    }

    std::vector<sim_slot> slots(p.n_slots);
    sched_states states;

    sim_result   res;
    std::vector<double> cold_waits;
    double now = 0.0;

    size_t remaining = p.n_sessions * p.n_turns;

    while (remaining > 0) {
        // ---- collect sessions whose next request has arrived
        std::vector<size_t> queued;
        for (auto & s : sessions) {
            if (s.turn < p.n_turns && s.ready_at <= now) {
                queued.push_back(s.id);
            }
        }

        // pick the earliest free slot
        size_t slot_i = 0;
        for (size_t i = 1; i < slots.size(); i++) {
            if (slots[i].free_at < slots[slot_i].free_at) {
                slot_i = i;
            }
        }
        auto & slot = slots[slot_i];

        if (queued.empty()) {
            // advance to the next arrival
            double next = 1e18;
            for (auto & s : sessions) {
                if (s.turn < p.n_turns) {
                    next = std::min(next, s.ready_at);
                }
            }
            now = std::max(now, next);
            continue;
        }

        now = std::max(now, slot.free_at);

        // ---- ordering decision: the real sched_pick_task over the real sched_score
        std::vector<server_tokens> prompts;
        std::vector<size_t>        scores;
        prompts.reserve(queued.size());
        for (size_t id : queued) {
            prompts.emplace_back(sim_prompt(p, sessions[id], sessions[id].turn), false);
        }
        if (cache_aware) {
            for (const auto & pr : prompts) {
                scores.push_back(sched_score(pr, slot.prompt, states));
            }
        } else {
            scores.assign(queued.size(), 0);       // FIFO: all equal, first wins
        }
        const size_t pick = sched_pick_task(scores);
        const size_t sid  = queued[pick];
        auto &       sess = sessions[sid];
        server_tokens prompt = std::move(prompts[pick]);

        const double wait_s = now - sess.ready_at;

        // ---- restore decision: the real sched_pick_restore / baseline
        size_t resident = slot.prompt.get_common_prefix(prompt);
        auto   chosen   = cache_aware
            ? sched_pick_restore(states, prompt, slot.prompt, th)
            : sched_pick_restore_baseline(states, prompt, slot.prompt);
        if (chosen != states.cend()) {
            resident = std::max(resident,
                                (size_t) chosen->prompt.tokens.get_common_prefix(prompt));
            states.erase(chosen);
        }

        const size_t to_prefill = prompt.size() - std::min(resident, prompt.size());

        // ---- cost model
        const double t_prefill = double(to_prefill) / p.prefill_tok_s;
        const double t_gen     = double(p.gen_tokens) / p.gen_tok_s;

        now         += t_prefill + t_gen;
        slot.free_at = now;

        // ---- save the finished prompt into the cache, evicting via the real picker
        server_prompt_cache_state fresh;
        {
            std::vector<llama_token> full;
            full.reserve(prompt.size());
            for (size_t i = 0; i < prompt.size(); i++) {
                full.push_back(prompt[i]);
            }
            fresh.prompt.tokens = server_tokens(full, false);
        }
        const size_t need = entry_bytes(p, fresh);

        // demand = argmax candidate for some still-queued session
        auto has_demand = [&](const server_prompt_cache_state & st) {
            for (size_t id : queued) {
                if (id == sid || sessions[id].turn >= p.n_turns) {
                    continue;
                }
                const server_tokens q(sim_prompt(p, sessions[id], sessions[id].turn), false);
                const server_prompt_cache_state * best = nullptr;
                size_t best_lcp = 0;
                for (const auto & c : states) {
                    const size_t lcp = c.prompt.tokens.get_common_prefix(q);
                    if (lcp > best_lcp) {
                        best_lcp = lcp;
                        best     = &c;
                    }
                }
                if (best == &st) {
                    return true;
                }
            }
            return false;
        };

        while (!states.empty() && cache_bytes_used(p, states) + need > p.cache_bytes) {
            auto victim = states.cbegin();
            if (cache_aware) {
                const auto v = sched_pick_victim(states, has_demand);
                if (v != states.cend()) {
                    victim = v;
                }
            }
            states.erase(victim);
        }
        if (cache_bytes_used(p, states) + need <= p.cache_bytes) {
            states.push_back(std::move(fresh));
        }

        // ---- bookkeeping
        slot.prompt   = server_tokens(std::vector<llama_token>{}, false);
        res.prefill_tokens += to_prefill;
        res.gen_tokens     += p.gen_tokens;
        res.n_requests++;
        const double reuse = prompt.size() ? double(resident) / prompt.size() : 1.0;
        if (reuse >= 0.95) {
            res.n_high_reuse++;
        } else {
            cold_waits.push_back(wait_s);
            if (prompt.size() >= 30000 && reuse < 0.05) {
                res.n_lost++;
            }
        }

        sess.turn++;
        sess.ready_at = now + p.think_s;
        remaining--;
        res.makespan_s = now;
    }

    std::sort(cold_waits.begin(), cold_waits.end());
    if (!cold_waits.empty()) {
        res.wait_p99_cold_s = cold_waits[(size_t) (cold_waits.size() * 0.99) % cold_waits.size()];
    }
    return res;
}
```

- [ ] **Step 4: Run to verify it passes**

Run:
```bash
cmake --build build --target test-server-sched-sim -j"$(nproc)" && ./build/bin/test-server-sched-sim
```
Expected: **PASS**, 1 test, 4 assertions, 0 failures. If `makespan_s` is not shorter, print
both results and check the cache is genuinely under pressure — with a cache large enough to
hold every session there is nothing to schedule around and the arms are identical by design.

- [ ] **Step 5: Calibrate against the measured production baseline**

The simulator is only credible if its **baseline** arm reproduces the pathology that was
actually observed. Add a second test asserting the baseline arm, run with the measured
production parameters (defaults above, `cache_bytes` 32 GiB, 6 sessions), lands in the
observed region: high-reuse fraction near 0.83 and a non-zero `n_lost`.

```cpp
static void test_sim_reproduces_measured_baseline(testing & t) {
    t.test("baseline_arm_matches_observed_pathology", [&](testing & t) {
        sim_params p;                          // measured production defaults
        const sim_result base = sim_run(p, false);

        const double high_reuse = double(base.n_high_reuse) / base.n_requests;
        t.assert_true("high-reuse fraction in the observed region (0.70-0.92)",
                      high_reuse > 0.70 && high_reuse < 0.92);
        t.assert_true("baseline loses some caches", base.n_lost > 0);
    });
}
```

Register it in `main`. If the baseline arm does not reproduce the pathology, the cost or
cache model is wrong — **fix the model, not the assertion**. A simulator whose baseline
disagrees with the measured 24h window cannot support any claim about the treatment arm.

**Calibrate in `SIM_HIST_APPEND_ONLY` only.** The 24h window was produced with thinking retained
in history (context grew by 1.13x the prior turn's generated tokens, and the median returning
request re-prefilled just 181 tokens — both signatures of retention, and consistent with the
template's `preserve_thinking` defaulting to true). The other policies have no measured
ground truth on this deployment, so they are additional scenarios, not calibrated ones. Say so.

- [ ] **Step 6: Commit**

```bash
git add tests/test-server-sched-sim.cpp tests/CMakeLists.txt
git commit -m "tests: discrete-event simulator for cache-aware scheduling

Prefill and generation rates are parameters, so throughput is computable without a
model or a server: a 24h workload simulates in milliseconds. Drives the real
sched_* functions rather than a reimplementation, which is only possible because
they are pure.

Baseline arm is calibrated against the measured 24h production window; a simulator
whose baseline disagrees with observation cannot support a claim about the
treatment arm.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 10 post-review amendment

The calibration requirement is **withdrawn** — see the spec section "The simulator demonstrates
mechanism; it does not predict production". It was a self-imposed standard appropriate to a
predictive model, not to a mechanism demonstration and regression gate, and the model has more
free parameters than the single observable constrained.

Remove:
- the baseline-calibration test and its 0.70-0.92 band assertion;
- `per_session_think_s` and `active_session_indices`, which exist only to serve calibration
  (this also returns runtime from 4.67s to ~13ms, since the duty cycle was the cost).

Retain:
- render self-consistency across all policies and session sizes — a genuine gate;
- treatment-beats-baseline under cache pressure;
- `TRUNCATE_FRONT` no-harm within 1%;
- heterogeneous session sizes, kept because a mixed workload exercises the policy more
  thoroughly than a uniform one, not to reproduce a distribution.

Record the observed baseline reuse figure in the report as a note, not an assertion.

---

### Task 11: Scenario sweep and pinned regression thresholds

Because the simulator is exact and fast, thresholds can be **equalities within a tight
tolerance** rather than envelopes. This is the reusable gate for future prompt-cache work.

**Files:**
- Modify: `tests/test-server-sched-sim.cpp`

**Interfaces:**
- Consumes: `sim_run` (Task 10).

- [ ] **Step 1: Write the failing sweep test**

Add to `tests/test-server-sched-sim.cpp`:

```cpp
// Sweep slots x cache pressure. Thresholds are pinned so any future change to the
// prompt-cache or scheduling logic that regresses throughput fails here.
static void test_sim_sweep_thresholds(testing & t) {
    struct scenario {
        const char * name;
        size_t       n_slots;
        double       cache_gib;
        sim_history_policy history;
        double       min_prefill_reduction;  // fraction, sched vs baseline
        double       min_makespan_reduction;
    };

    // Values are filled in from Step 2's observed output, then pinned.
    const scenario scenarios[] = {
        {"1slot-tight-append",   1, 4.0,  SIM_HIST_APPEND_ONLY,                0.0, 0.0},
        {"1slot-prod-append",    1, 32.0, SIM_HIST_APPEND_ONLY,                0.0, 0.0},
        {"2slot-tight-append",   2, 4.0,  SIM_HIST_APPEND_ONLY,                0.0, 0.0},
        {"4slot-tight-append",   4, 4.0,  SIM_HIST_APPEND_ONLY,                0.0, 0.0},
        {"8slot-roomy-append",   8, 64.0, SIM_HIST_APPEND_ONLY,                0.0, 0.0},
        {"1slot-tight-dropprev", 1, 4.0,  SIM_HIST_DROP_REASONING_PRIOR_TURNS, 0.0, 0.0},
        {"4slot-tight-dropprev", 4, 4.0,  SIM_HIST_DROP_REASONING_PRIOR_TURNS, 0.0, 0.0},
        {"1slot-tight-dropall",  1, 4.0,  SIM_HIST_DROP_REASONING_ALWAYS,      0.0, 0.0},
        {"1slot-tight-compact",  1, 4.0,  SIM_HIST_COMPACT_AT_THRESHOLD,       0.0, 0.0},
    };

    for (const auto & sc : scenarios) {
        t.test(sc.name, [&](testing & t) {
            sim_params p;
            p.n_slots     = sc.n_slots;
            p.history     = sc.history;
            p.cache_bytes = (size_t) (sc.cache_gib * 1024 * 1024 * 1024);

            const sim_result b = sim_run(p, false);
            const sim_result s = sim_run(p, true);

            const double dp = 1.0 - double(s.prefill_tokens) / double(b.prefill_tokens);
            const double dm = 1.0 - s.makespan_s / b.makespan_s;

            t.assert_true("prefill reduction meets pinned threshold",
                          dp >= sc.min_prefill_reduction - 1e-9);
            t.assert_true("makespan reduction meets pinned threshold",
                          dm >= sc.min_makespan_reduction - 1e-9);
        });
    }
}
```

Register in `main`.

- [ ] **Step 2: Run, observe, then pin**

Run:
```bash
cmake --build build --target test-server-sched-sim -j"$(nproc)"
./build/bin/test-server-sched-sim 2>&1 | tee /tmp/sweep.txt
```
It passes trivially with zero thresholds. Now add a temporary `printf` of `dp` and `dm` per
scenario, re-run, record the values, and set each `min_*` to the observed value **less a 10%
margin**. Delete the `printf`. Re-run and confirm it still passes.

Document the pinned numbers and the machine-independence claim in a comment above
`scenarios`: the simulator has no timing dependence, so these are exact and identical on any
host — that is what makes tight thresholds legitimate here where they would not be for a
wall-clock benchmark.

- [ ] **Step 3: Verify the thresholds have teeth**

Temporarily make `sched_pick_victim` ignore `has_demand` (return `states.cbegin()`
unconditionally), rebuild, and run.

Run:
```bash
cmake --build build --target test-server-sched-sim -j"$(nproc)" && ./build/bin/test-server-sched-sim
```
Expected: **FAIL** on the tight-cache scenarios. Revert the change and confirm PASS. A
threshold that cannot fail is not a gate.

- [ ] **Step 4: Commit**

```bash
git add tests/test-server-sched-sim.cpp
git commit -m "tests: pin throughput thresholds across slot and cache-pressure scenarios

The simulator has no timing dependence, so these are exact and host-independent,
which is what makes tight thresholds legitimate here.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Producing the PR numbers

Run `./build/bin/test-server-sched-sim` and report the sweep table. There is no measurement
run, no production instrumentation and no 12-hour window.

**Be precise about what each layer of evidence establishes**, because a reviewer will be:

- The **motivation** is a real measurement: 24h of production journal logs, 712 requests,
  73% of lost-cache requests evicted during their own queue wait, 1.45h of 1.79h wasted
  prefill. That is observed, not modelled.
- **Correctness** is the tier-1 unit tests.
- **Integration** is the tier-2 server tests.
- **Throughput** is *simulated* under a fixed-rate cost model. Do not present simulated
  hours as measured hours. The claim it supports is "under this cost model, with the
  baseline arm calibrated to the observed window, the algorithm re-prefills N% fewer tokens
  and finishes M% sooner" — which is exactly the claim a scheduling change should make.
- The simulator's limit: it assumes constant prefill and generation rates. Real generation
  degrades with context length (measured 30.9 t/s under 25k, 13.2 t/s over 180k), but the
  scheduler does not change context lengths, so that term largely cancels between arms.
  Say so rather than leaving it to be found.
- Report the cold-request p99 wait from the sweep alongside the throughput gain.

## Before opening the PR

- [ ] Drop the `docs/superpowers/` commits from the upstream branch.
- [ ] Decide whether to report the #27148 reproduction on the issue separately, so it is
      attributable to the reproduction rather than arriving tangled with a feature PR.
- [ ] Add a comment to the characterization test noting it must be deleted when #27148 is
      fixed upstream.
- [ ] State the cold-request p99 wait under both arms in the PR description. A scheduling
      change presented without its tail-latency cost should be rejected.
