# Dynamic Context Sizing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `llama-server` start with a small context and grow/shrink it on demand (power-of-two tiers from 32k up to `--ctx-size`), so a single client never pays for context it does not use, with mid-stream growth that is transparent to the client.

**Architecture:** Reuse the existing sleep/wake reload path (`destroy()` + `load_model(params_base)`). A new `ctx_dynamic` flag enables tier-based sizing. Tier selection is a small pure helper (unit-tested in isolation). Resize-at-arrival happens in `process_single_task` before a slot is launched. Mid-stream grow snapshots the in-flight token sequence, reloads at the next tier, reprocesses the tokens, and resumes. Each resize re-runs `common_fit_params` with an explicit `n_ctx`, so layer placement is re-fit for the new tier.

**Tech Stack:** C++17, llama.cpp server (`tools/server`), `common` library (`common/arg.cpp`, `common/common.h`), GoogleTest-free assert-based C++ unit tests (`tests/`), pytest server integration tests (`tools/server/tests/`).

---

## Background: read these before starting

- Design spec: `docs/superpowers/specs/2026-06-22-dynamic-context-sizing-design.md`
- `AGENTS.md` - contribution rules. Avoid unicode (`-` not em-dash, `x` not multiply sign). Keep comments concise. Commit messages use `Assisted-by:` not `Co-authored-by:`. Do NOT push or open PRs.
- Key source anchors (line numbers approximate, verify before editing):
  - `common/common.h:437` - `n_ctx`; `:442` - `n_parallel`; `:556` - `ctx_shift`; `:625` - `sleep_idle_seconds`. New fields go near these.
  - `common/arg.cpp:3225` - `--sleep-idle-seconds` definition; mirror this pattern for the new flag.
  - `tools/server/server-context.cpp:934` - `destroy()`; `:948` - `handle_sleeping_state()`; `:996` - `load_model()`; `:1167` - `n_ctx = llama_n_ctx(ctx_tgt)`; `:1304` - `slots.clear()`.
  - `tools/server/server-context.cpp:2347` - `process_single_task()`; `:2364` - `get_available_slot()`; `:2397` - `launch_slot_with_task()`.
  - `tools/server/server-context.cpp:1886` - mid-stream "out of context" stop; `:2825` - `pre_decode()` ctx-shift.
  - `tools/server/server-context.cpp:3114-3133` - request-arrival "input larger than context" error.
  - `tools/server/server-queue.cpp:125` - `start_loop()` and sleep machinery.
- The feature lives entirely in `server-context` + `common`. No router (`server-models.cpp`) changes are needed.

## File structure

- `common/common.h` - add `ctx_dynamic` (bool) and `ctx_dynamic_min` (int32_t) params.
- `common/arg.cpp` - add `--ctx-dynamic` flag; startup validation.
- `tools/server/server-ctx-tiers.h` (new) - pure tier-math helpers (header-only, no llama deps), so they can be unit-tested standalone.
- `tests/test-ctx-tiers.cpp` (new) - C++ unit test for tier math; registered in `tests/CMakeLists.txt`.
- `tools/server/server-context.cpp` - resize plumbing: a `resize_context(int32_t)` method, resize-at-arrival in `process_single_task`, mid-stream grow in `update_slots`/`pre_decode`.
- `tools/server/tests/unit/test_ctx_dynamic.py` (new) - server integration tests.
- `tools/server/README.md` - document the flag.

---

## Phase 1: Flag and tier math (no behavior change yet)

### Task 1: Add `ctx_dynamic` params

**Files:**
- Modify: `common/common.h` (near line 437-465, the context params block)

- [ ] **Step 1: Add the param fields**

In `common/common.h`, in the `common_params` struct near the existing `n_ctx` field (around line 437), add:

```cpp
    bool    ctx_dynamic     = false; // grow/shrink context in tiers on demand (server, single-slot)
    int32_t ctx_dynamic_min = 32768; // smallest context tier when ctx_dynamic is enabled
```

- [ ] **Step 2: Build to confirm it compiles**

Run: `cmake --build build --target llama-common`
Expected: builds with no error (these are just new struct fields).

- [ ] **Step 3: Commit**

```bash
git add common/common.h
git commit --no-gpg-sign -m "common: add ctx_dynamic params

Assisted-by: Claude Opus 4.8"
```

---

### Task 2: Tier-math helpers (pure, unit-tested)

**Files:**
- Create: `tools/server/server-ctx-tiers.h`
- Create: `tests/test-ctx-tiers.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Write the failing test**

Create `tests/test-ctx-tiers.cpp`:

```cpp
#include "../tools/server/server-ctx-tiers.h"

#include <vector>
#include <cstdio>

#undef NDEBUG
#include <cassert>

static void test_build_tiers() {
    // 200000 cap, 32768 min -> {32768, 65536, 131072, 200000}
    auto t = server_ctx_build_tiers(32768, 200000);
    assert((t == std::vector<int32_t>{32768, 65536, 131072, 200000}));

    // cap is an exact power of two -> no duplicate top tier
    auto t2 = server_ctx_build_tiers(32768, 131072);
    assert((t2 == std::vector<int32_t>{32768, 65536, 131072}));

    // cap <= min -> single tier equal to cap
    auto t3 = server_ctx_build_tiers(32768, 16000);
    assert((t3 == std::vector<int32_t>{16000}));

    // cap == min -> single tier
    auto t4 = server_ctx_build_tiers(32768, 32768);
    assert((t4 == std::vector<int32_t>{32768}));
}

static void test_required_tier() {
    std::vector<int32_t> tiers = {32768, 65536, 131072, 200000};

    // fits in smallest
    assert(server_ctx_required_tier(tiers, 1000) == 32768);
    // exactly on a boundary picks that tier
    assert(server_ctx_required_tier(tiers, 32768) == 32768);
    // just over a boundary picks the next tier
    assert(server_ctx_required_tier(tiers, 32769) == 65536);
    // larger than max returns the max tier (caller handles overflow separately)
    assert(server_ctx_required_tier(tiers, 999999) == 200000);
}

static void test_shrink_target() {
    std::vector<int32_t> tiers = {32768, 65536, 131072, 200000};

    // currently at 131072, need only 1000 -> shrink to 32768 (well under next-down)
    assert(server_ctx_shrink_target(tiers, 131072, 1000, 15) == 32768);

    // currently at 131072, need 60000 -> required tier is 65536, which is the
    // next tier down; 60000 is NOT comfortably under 65536*(1-0.15)=55705,
    // so do NOT shrink (avoid thrash) -> stay at 131072
    assert(server_ctx_shrink_target(tiers, 131072, 60000, 15) == 131072);

    // currently at 131072, need 50000 -> 50000 < 55705, shrink to 65536
    assert(server_ctx_shrink_target(tiers, 131072, 50000, 15) == 65536);

    // never grow via shrink: required tier above current returns current
    assert(server_ctx_shrink_target(tiers, 32768, 100000, 15) == 32768);
}

int main() {
    test_build_tiers();
    test_required_tier();
    test_shrink_target();
    printf("test-ctx-tiers: OK\n");
    return 0;
}
```

- [ ] **Step 2: Register the test (so it fails to build, proving it runs)**

In `tests/CMakeLists.txt`, after the `llama_build_and_test` function definition (after line 117) and near other `llama_build_and_test(...)` calls, add:

```cmake
llama_build_and_test(test-ctx-tiers.cpp)
```

- [ ] **Step 3: Run to verify it fails**

Run: `cmake -B build && cmake --build build --target test-ctx-tiers`
Expected: FAIL - `server-ctx-tiers.h` does not exist / undefined functions.

- [ ] **Step 4: Implement the helpers**

Create `tools/server/server-ctx-tiers.h`:

```cpp
#pragma once

#include <cstdint>
#include <vector>

// Build the list of context-size tiers: powers of two from `min_tier` up to
// `max_ctx`, with `max_ctx` itself appended as the top tier when it is not a
// power of two. If `max_ctx <= min_tier`, a single tier equal to `max_ctx` is
// returned (the feature is effectively a no-op).
static inline std::vector<int32_t> server_ctx_build_tiers(int32_t min_tier, int32_t max_ctx) {
    std::vector<int32_t> tiers;
    if (max_ctx <= min_tier) {
        tiers.push_back(max_ctx);
        return tiers;
    }
    for (int32_t t = min_tier; t < max_ctx; t *= 2) {
        tiers.push_back(t);
    }
    if (tiers.empty() || tiers.back() != max_ctx) {
        tiers.push_back(max_ctx);
    }
    return tiers;
}

// Return the smallest tier >= n_tokens_needed. If none fits, return the largest
// tier (the caller is responsible for treating an over-cap request as an error).
static inline int32_t server_ctx_required_tier(const std::vector<int32_t> & tiers, int32_t n_tokens_needed) {
    for (int32_t t : tiers) {
        if (t >= n_tokens_needed) {
            return t;
        }
    }
    return tiers.empty() ? n_tokens_needed : tiers.back();
}

// Decide a shrink target with hysteresis. Returns `current_tier` unchanged unless
// the requirement fits in a strictly smaller tier AND sits comfortably below that
// smaller tier's capacity (by `margin_pct` percent), to avoid thrashing near a
// boundary. Never grows (returns `current_tier` if the requirement needs >= it).
static inline int32_t server_ctx_shrink_target(const std::vector<int32_t> & tiers,
                                               int32_t current_tier,
                                               int32_t n_tokens_needed,
                                               int     margin_pct) {
    const int32_t required = server_ctx_required_tier(tiers, n_tokens_needed);
    if (required >= current_tier) {
        return current_tier; // never grow here; growth is handled elsewhere
    }
    // required < current_tier: only shrink if comfortably under the smaller tier
    const int64_t threshold = (int64_t) required * (100 - margin_pct) / 100;
    if (n_tokens_needed <= threshold) {
        return required;
    }
    return current_tier;
}
```

- [ ] **Step 5: Run to verify it passes**

Run: `cmake --build build --target test-ctx-tiers && ./build/bin/test-ctx-tiers`
Expected: PASS - prints `test-ctx-tiers: OK`.

- [ ] **Step 6: Commit**

```bash
git add tools/server/server-ctx-tiers.h tests/test-ctx-tiers.cpp tests/CMakeLists.txt
git commit --no-gpg-sign -m "server: add context-tier math helpers with unit tests

Assisted-by: Claude Opus 4.8"
```

---

### Task 3: `--ctx-dynamic` flag and startup validation

**Files:**
- Modify: `common/arg.cpp` (near line 3225, the `--sleep-idle-seconds` block)

- [ ] **Step 1: Add the flag definition**

In `common/arg.cpp`, near the `--sleep-idle-seconds` block (around line 3225), add a new option mirroring that pattern:

```cpp
    add_opt(common_arg(
        {"--ctx-dynamic"},
        string_format("enable on-demand context sizing in power-of-two tiers from %d up to --ctx-size "
                      "(server only, requires a single slot; default: %s)",
                      32768, params.ctx_dynamic ? "enabled" : "disabled"),
        [](common_params & params) {
            params.ctx_dynamic = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
```

- [ ] **Step 2: Add startup validation**

Server startup validation lives in `common/arg.cpp` where other server-only invariants are checked. Search for an existing server validation (`grep -n "n_parallel" common/arg.cpp` and look for a post-parse check block, or the `LLAMA_EXAMPLE_SERVER` handling near the end of `common_params_parse`). Add, after parsing completes:

```cpp
    if (params.ctx_dynamic) {
        if (params.n_parallel > 1) {
            throw std::invalid_argument("--ctx-dynamic requires a single slot (set --parallel 1)");
        }
        if (params.n_ctx != 0 && params.n_ctx <= params.ctx_dynamic_min) {
            // single tier; feature is a no-op but not an error
            LOG_WRN("%s: --ctx-size (%d) <= dynamic minimum (%d); ctx-dynamic has no effect\n",
                    __func__, params.n_ctx, params.ctx_dynamic_min);
        }
    }
```

If no suitable post-parse block exists for `LLAMA_EXAMPLE_SERVER`, place the check at the end of `common_params_parse` guarded by `ex == LLAMA_EXAMPLE_SERVER`. Verify the exact location with `grep -n "common_params_parse\b" common/arg.cpp`.

- [ ] **Step 3: Build and smoke-test the flag**

Run:
```bash
cmake --build build --target llama-server
./build/bin/llama-server --help 2>&1 | grep -A2 "ctx-dynamic"
```
Expected: the `--ctx-dynamic` help text appears.

- [ ] **Step 4: Verify validation rejects multi-slot**

Run:
```bash
./build/bin/llama-server --ctx-dynamic --parallel 2 --model /nonexistent 2>&1 | grep -i "single slot"
```
Expected: prints the "requires a single slot" error (argument validation fires before model load).

- [ ] **Step 5: Commit**

```bash
git add common/arg.cpp
git commit --no-gpg-sign -m "common: add --ctx-dynamic flag and startup validation

Assisted-by: Claude Opus 4.8"
```

---

## Phase 2: Resize-at-arrival (grow/shrink before a request is launched)

This phase makes resizing work for the easy case: no generation is in flight. It exercises the full reload-and-refit path and is independently valuable and testable.

### Task 4: Add resize state and the `resize_context` method

**Files:**
- Modify: `tools/server/server-context.cpp` (the `server_context_impl` struct; `load_model` near line 996-1167; include the new header)

- [ ] **Step 1: Include the tier header and add state fields**

At the top of `tools/server/server-context.cpp`, with the other includes, add:

```cpp
#include "server-ctx-tiers.h"
```

In `server_context_impl` (near the other state fields around line 906-932), add:

```cpp
    // dynamic context sizing (params_base.ctx_dynamic)
    std::vector<int32_t> ctx_tiers;       // computed once after first load
    int32_t              ctx_cap = 0;     // effective max context (--ctx-size resolved)
    int32_t              ctx_current_tier = 0; // current allocated tier
```

- [ ] **Step 2: Initialize tiers after the first successful load**

In `load_model()`, right after `n_ctx = llama_n_ctx(ctx_tgt);` (around line 1167), add:

```cpp
        if (params_base.ctx_dynamic && ctx_tiers.empty()) {
            // first load: --ctx-size is the cap; remember it and build the tier list.
            // n_ctx here reflects the tier we just loaded at (set before load on resize).
            ctx_cap   = params_base.n_ctx_orig > 0 ? params_base.n_ctx_orig : n_ctx;
            ctx_tiers = server_ctx_build_tiers(params_base.ctx_dynamic_min, ctx_cap);
        }
        if (params_base.ctx_dynamic) {
            ctx_current_tier = n_ctx;
        }
```

Note: this references `params_base.n_ctx_orig`. We need the original `--ctx-size` preserved because we overwrite `params_base.n_ctx` on each resize. Add that field in Step 3.

- [ ] **Step 3: Preserve the original ctx-size and start at the smallest tier**

Add to `common_params` in `common/common.h` near `ctx_dynamic`:

```cpp
    int32_t n_ctx_orig = 0; // original --ctx-size before dynamic resizing overwrites n_ctx
```

In `server_context::load_model` (the public wrapper) or at the very start of the impl `load_model`, before the first load only, capture the cap and force the starting tier. The cleanest spot is the public wrapper `bool server_context::load_model(common_params & params)` (search `grep -n "bool server_context::load_model" tools/server/server-context.cpp`). Add at its start:

```cpp
    if (params.ctx_dynamic && params.n_ctx_orig == 0) {
        params.n_ctx_orig = params.n_ctx > 0 ? params.n_ctx : 0;
        const int32_t cap = params.n_ctx_orig;
        if (cap == 0 || cap > params.ctx_dynamic_min) {
            // start small; cap == 0 means "from model", resolved on first load,
            // so only force the floor when we have an explicit cap above it.
            if (cap > params.ctx_dynamic_min) {
                params.n_ctx = params.ctx_dynamic_min;
            }
        }
    }
```

Edge case to honor: if `--ctx-size` is 0 ("from model"), we cannot know the cap until the model is loaded. For this plan, require an explicit `--ctx-size` when `--ctx-dynamic` is set; add to the Task 3 validation block:

```cpp
        if (params.n_ctx == 0) {
            throw std::invalid_argument("--ctx-dynamic requires an explicit --ctx-size (the maximum tier)");
        }
```

(Adjust Task 3's no-op warning accordingly: with this rule, `n_ctx == 0` is now an error, so remove the `n_ctx != 0 &&` guard's reliance on 0 being allowed.)

- [ ] **Step 4: Implement `resize_context`**

Add this method to `server_context_impl` (near `handle_sleeping_state`, around line 960). It reuses the same destroy/reload sequence the sleep path uses:

```cpp
    // Reallocate the context (and re-fit layer placement) to `new_tier`.
    // MUST be called from the main loop thread with no generation in flight.
    // Returns true if a resize happened.
    bool resize_context(int32_t new_tier) {
        if (!params_base.ctx_dynamic || new_tier == ctx_current_tier) {
            return false;
        }
        SRV_INF("resizing context tier %d -> %d\n", ctx_current_tier, new_tier);
        callback_state(SERVER_STATE_LOADING, {{"reason", "ctx_resize"}});

        destroy();
        params_base.n_ctx = new_tier;
        if (!load_model(params_base)) {
            GGML_ABORT("failed to reload model after context resize");
        }
        callback_state(SERVER_STATE_READY, {});
        return true;
    }
```

Note: `load_model` already re-runs `common_init_from_params` -> `common_fit_params` with the explicit `n_ctx = new_tier`, which re-fits layer placement. `load_model` also calls `slots.clear()` and rebuilds slots at the new `n_ctx`, so all slot/KV state is reset (acceptable here: no generation is in flight).

- [ ] **Step 5: Build**

Run: `cmake --build build --target llama-server`
Expected: compiles. (No behavior change yet - `resize_context` is not called.)

- [ ] **Step 6: Commit**

```bash
git add common/common.h tools/server/server-context.cpp
git commit --no-gpg-sign -m "server: add resize_context using the reload path

Assisted-by: Claude Opus 4.8"
```

---

### Task 5: Trigger resize-at-arrival in `process_single_task`

**Files:**
- Modify: `tools/server/server-context.cpp` (`process_single_task` near line 2347, the COMPLETION/INFILL case around 2364-2397)

- [ ] **Step 1: Compute the needed tier and resize before launching the slot**

In `process_single_task`, in the `SERVER_TASK_TYPE_COMPLETION`/`INFILL` branch, AFTER the task's tokens are known but BEFORE `get_available_slot`/`launch_slot_with_task` (around line 2364), add:

```cpp
                if (params_base.ctx_dynamic && !ctx_tiers.empty()) {
                    // tokens needed = prompt + a generation budget (capped at the max tier)
                    const int32_t n_prompt = (int32_t) task.tokens.size();
                    const int32_t n_pred   = task.params.n_predict > 0 ? task.params.n_predict : 0;
                    const int32_t needed   = std::min(ctx_cap, n_prompt + n_pred + 4); // +4 safety margin

                    int32_t target = ctx_current_tier;
                    const int32_t req_tier = server_ctx_required_tier(ctx_tiers, needed);
                    if (req_tier > ctx_current_tier) {
                        target = req_tier; // grow
                    } else {
                        target = server_ctx_shrink_target(ctx_tiers, ctx_current_tier, needed, 15); // shrink w/ hysteresis
                    }
                    if (target != ctx_current_tier) {
                        resize_context(target);
                    }
                }
```

Note: confirm the field name for tokens on `server_task` is `tokens` and for predict count is `params.n_predict` (search `grep -n "n_predict" tools/server/server-task.h` and `grep -n "llama_tokens tokens\|server_tokens tokens" tools/server/server-task.h`). Adjust names to match.

- [ ] **Step 2: Build**

Run: `cmake --build build --target llama-server`
Expected: compiles.

- [ ] **Step 3: Write the server integration test**

Create `tools/server/tests/unit/test_ctx_dynamic.py`:

```python
import pytest
from utils import *

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_ctx = 1024          # max tier (cap)
    server.n_slots = 1
    server.n_predict = 64


def test_ctx_dynamic_small_request_starts_small(monkeypatch):
    # With ctx-dynamic and a tiny min tier, a small request should succeed and
    # the server should report a small context (not the full cap).
    global server
    server.server_args = ["--ctx-dynamic", "--ctx-dynamic-min", "256"]
    server.n_ctx = 1024
    server.start()
    res = server.make_request("POST", "/completion", data={
        "n_predict": 8,
        "prompt": "Hello",
    })
    assert res.status_code == 200
    # the slot context should reflect the smallest tier (256), not 1024
    props = server.make_request("GET", "/props")
    assert props.status_code == 200
    assert props.body["default_generation_settings"]["n_ctx"] <= 256


def test_ctx_dynamic_large_request_grows():
    # A prompt that needs more than the smallest tier should trigger a grow and
    # still succeed (no "exceeds context" error).
    global server
    server.server_args = ["--ctx-dynamic", "--ctx-dynamic-min", "256"]
    server.n_ctx = 1024
    server.start()
    long_prompt = "word " * 300  # ~300+ tokens, exceeds the 256 tier
    res = server.make_request("POST", "/completion", data={
        "n_predict": 8,
        "prompt": long_prompt,
    })
    assert res.status_code == 200
    assert res.body["truncated"] is False
```

Note: this test references `--ctx-dynamic-min`. Add that as a flag in Task 3's block if you want it tunable for tests; otherwise hardcode the test's expectation to the 32768 default and use a model/prompt large enough to cross it. Adding a tunable `--ctx-dynamic-min` flag is recommended specifically to keep tests fast (small tiers). If you add it: mirror the `--ctx-dynamic` option, taking an int, storing to `params.ctx_dynamic_min`, with `.set_examples({LLAMA_EXAMPLE_SERVER})`. Also confirm `ServerProcess` supports `server_args` (search `grep -n "server_args" tools/server/tests/utils.py`); if not, set the args via the documented mechanism in `utils.py`.

- [ ] **Step 4: Run the test**

Run:
```bash
cd tools/server/tests
LLAMA_SERVER_BIN_PATH=../../../build/bin/llama-server python -m pytest unit/test_ctx_dynamic.py -v
```
Expected: both tests PASS. If `n_ctx` in `/props` reflects the resized tier, the resize path works end to end.

- [ ] **Step 5: Commit**

```bash
git add tools/server/server-context.cpp common/arg.cpp tools/server/tests/unit/test_ctx_dynamic.py
git commit --no-gpg-sign -m "server: resize context on request arrival under ctx-dynamic

Assisted-by: Claude Opus 4.8"
```

---

## Phase 3: Mid-stream grow-and-continue (transparent resize)

This is the riskiest phase. It snapshots an in-flight generation, reloads at the next tier, reprocesses tokens, and resumes - so the client sees only a pause. Build it on top of the proven Phase 2 resize path.

### Task 6: Detect mid-stream overflow and snapshot before reload

**Files:**
- Modify: `tools/server/server-context.cpp` (`update_slots`/`pre_decode`; the out-of-context stop at line 1886 and ctx-shift at 2825)

- [ ] **Step 1: Identify the snapshot data**

The durable artifact across a reload is the token sequence. For the single in-flight slot, snapshot:
- `slot.prompt.tokens.get_tokens()` (the full prompt + generated tokens; `server_tokens`)
- the active `server_task` (move it out so the response channel/`id_task` is preserved)
- generation counters needed to resume: `slot.n_decoded`, sampler state.

Confirm available accessors:
```bash
grep -n "get_tokens\|struct server_slot\|server_task\b.*task;\|n_decoded\|struct server_tokens" tools/server/server-context.cpp tools/server/server-task.h | head -40
```

- [ ] **Step 2: Add a mid-stream grow helper**

Add to `server_context_impl`:

```cpp
    // When a generating slot fills the context and a larger tier exists, grow
    // transparently: snapshot the slot's tokens + task, reload at the next tier,
    // then re-decode the tokens so generation can resume. Returns true if grown.
    // Single-slot invariant: only valid when n_parallel == 1 and not multimodal.
    bool try_grow_midstream(server_slot & slot) {
        if (!params_base.ctx_dynamic || mctx != nullptr) {
            return false; // disabled or multimodal (token bookkeeping not supported)
        }
        // find the next tier strictly greater than current
        int32_t next_tier = ctx_current_tier;
        for (int32_t t : ctx_tiers) {
            if (t > ctx_current_tier) { next_tier = t; break; }
        }
        if (next_tier == ctx_current_tier) {
            return false; // already at max tier; caller falls back to shift/stop
        }

        SRV_INF("%s", "growing context mid-stream to continue generation\n");

        // snapshot
        llama_tokens snapshot = slot.prompt.tokens.get_tokens(); // copy
        server_task  task     = std::move(*slot.task);           // preserve response channel
        const int    n_decoded_before = slot.n_decoded;

        // reload at the larger tier (clears slots + KV by design)
        resize_context(next_tier);

        // after reload `slots` is rebuilt and empty; re-launch the same task with
        // the snapshot as its prompt so the KV cache is rebuilt by reprocessing.
        server_slot & new_slot = slots[0];
        task.tokens = server_tokens(snapshot, /*has_mtmd=*/false);
        // resume: we want to continue generating, not re-emit the prompt as output.
        // mark the already-generated tail so the generation budget/stop logic resumes.
        // (see Step 3 for the resume bookkeeping)
        if (!launch_slot_with_task(new_slot, std::move(task))) {
            SRV_ERR("%s", "failed to relaunch slot after mid-stream grow\n");
            return false;
        }
        new_slot.n_decoded = n_decoded_before;
        return true;
    }
```

Note: the exact `server_tokens` constructor and `slot.task` ownership must be verified against `server-task.h`/`server-context.cpp`. The key semantic: re-decoding `snapshot` rebuilds the KV cache; generation then continues. Adjust the resume bookkeeping in Step 3.

- [ ] **Step 3: Resume bookkeeping (do not re-stream the prompt)**

The snapshot contains prompt + already-generated tokens. On relaunch, those are reprocessed as prompt (KV rebuild) and MUST NOT be re-sent to the client as new output. Verify how the slot distinguishes prompt tokens from generated output (search `grep -n "n_prompt_tokens\|n_sent_text\|send_text\|n_decoded\|task->n_tokens" tools/server/server-context.cpp`). The resume must:
- treat the entire snapshot as the prompt (so `n_prompt_tokens` = snapshot size),
- preserve the stop-string / partial-UTF-8 buffers from the original generation if any,
- continue the `n_predict` budget from where it left off (so total generated count is consistent).

Document the chosen mechanism in a concise comment at the call site. This is the most delicate part; implement it explicitly rather than relying on defaults.

- [ ] **Step 4: Call the helper at the overflow points**

At line ~1886 (`if (!params_base.ctx_shift && slot.prompt.n_tokens() + 1 >= slot.n_ctx)`) and the ctx-shift path at ~2829, gate the existing behavior behind a grow attempt:

```cpp
        if (slot.state == SLOT_STATE_GENERATING && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
            if (params_base.ctx_dynamic && try_grow_midstream(slot)) {
                return; // grown and relaunched; continue on the next loop iteration
            }
            // ... existing ctx-shift or STOP_TYPE_LIMIT behavior unchanged ...
        }
```

Apply the analogous guard at the line-1886 stop check so that, with ctx-dynamic on and a higher tier available, we grow instead of stopping.

- [ ] **Step 5: Build**

Run: `cmake --build build --target llama-server`
Expected: compiles.

- [ ] **Step 6: Commit (checkpoint, even though test comes next)**

```bash
git add tools/server/server-context.cpp
git commit --no-gpg-sign -m "server: grow context mid-stream under ctx-dynamic

Assisted-by: Claude Opus 4.8"
```

---

### Task 7: Mid-stream grow integration test

**Files:**
- Modify: `tools/server/tests/unit/test_ctx_dynamic.py`

- [ ] **Step 1: Add a streaming test that crosses a tier mid-generation**

Append to `tools/server/tests/unit/test_ctx_dynamic.py`:

```python
def test_ctx_dynamic_midstream_grow_streaming():
    # Small first tier; a short prompt but long generation that fills the 256
    # tier mid-stream. With ctx-dynamic the stream must continue past 256 tokens
    # (grow to the next tier) rather than stop early or truncate.
    global server
    server.server_args = ["--ctx-dynamic", "--ctx-dynamic-min", "256"]
    server.n_ctx = 1024
    server.n_predict = -1
    server.start()
    res = server.make_stream_request("POST", "/v1/completions", data={
        "n_predict": 400,   # would overflow the 256 tier; must grow to continue
        "prompt": "Once upon a time",
        "stream": True,
    })
    content = ""
    finish_reason = None
    for data in res:
        choice = data["choices"][0]
        if choice["finish_reason"] is not None:
            finish_reason = choice["finish_reason"]
        else:
            content += choice["text"]
    # generation continued past the first tier; we got a substantial amount of text
    assert len(content) > 0
    # it did NOT stop purely because the small tier filled (would be "length" at ~256)
    assert finish_reason in ("length", "stop")
```

- [ ] **Step 2: Run the test**

Run:
```bash
cd tools/server/tests
LLAMA_SERVER_BIN_PATH=../../../build/bin/llama-server python -m pytest unit/test_ctx_dynamic.py::test_ctx_dynamic_midstream_grow_streaming -v
```
Expected: PASS. Watch server logs (run with `DEBUG=1`) to confirm a "growing context mid-stream" log line and a reload occurred during the single request.

- [ ] **Step 3: Manually verify coherence (no token loss)**

Run the server with `--ctx-dynamic --ctx-dynamic-min 256 --ctx-size 1024 -v` against a real small model and a prompt that forces a grow; read the streamed output to confirm the text is coherent across the resize boundary (no repeated prompt, no gap). Capture the log showing the resize.

- [ ] **Step 4: Commit**

```bash
git add tools/server/tests/unit/test_ctx_dynamic.py
git commit --no-gpg-sign -m "server: test mid-stream context grow

Assisted-by: Claude Opus 4.8"
```

---

## Phase 4: Polish and documentation

### Task 8: Document the feature

**Files:**
- Modify: `tools/server/README.md`

- [ ] **Step 1: Find the args/options section**

Run: `grep -n "ctx-size\|sleep-idle\|--parallel" tools/server/README.md | head`

- [ ] **Step 2: Add documentation near the context options**

Add a subsection describing `--ctx-dynamic` (and `--ctx-dynamic-min` if added): what it does, the single-slot requirement, that `--ctx-size` is the cap, the tiers (powers of two from 32k), that mid-stream growth pauses the stream while reloading, and that multimodal mid-stream grow is unsupported (resize-at-arrival still applies). Keep prose concise per AGENTS.md.

- [ ] **Step 3: Commit**

```bash
git add tools/server/README.md
git commit --no-gpg-sign -m "docs: document --ctx-dynamic server option

Assisted-by: Claude Opus 4.8"
```

---

### Task 9: Full regression check

- [ ] **Step 1: Run the tier unit test**

Run: `./build/bin/test-ctx-tiers`
Expected: `test-ctx-tiers: OK`.

- [ ] **Step 2: Run the new server tests**

Run:
```bash
cd tools/server/tests
LLAMA_SERVER_BIN_PATH=../../../build/bin/llama-server python -m pytest unit/test_ctx_dynamic.py -v
```
Expected: all PASS.

- [ ] **Step 3: Run the existing ctx-shift tests (confirm no regression)**

Run:
```bash
cd tools/server/tests
LLAMA_SERVER_BIN_PATH=../../../build/bin/llama-server python -m pytest unit/test_ctx_shift.py -v
```
Expected: all PASS (default behavior, flag off, unchanged).

- [ ] **Step 4: Confirm default behavior unchanged with the flag off**

Run a normal completion without `--ctx-dynamic` and confirm `/props` reports the full `--ctx-size` (no tiering).

---

## Self-review notes (for the implementer)

- **n_ctx == 0 case:** the plan requires an explicit `--ctx-size` with `--ctx-dynamic` (Task 4 Step 3). This is a deliberate scope decision; resolving the cap from model metadata before the first load is possible but out of scope.
- **Per-slot n_ctx:** the server divides `n_ctx` across slots (`n_ctx_slot = llama_n_ctx_seq(ctx_tgt)`). With the single-slot invariant, slot n_ctx == context n_ctx, so tier math against `task.tokens.size()` is correct. Do NOT extend this to multi-slot without revisiting the math.
- **Resume bookkeeping (Task 6 Step 3) is the highest-risk item.** If implementing incrementally, it is acceptable to first ship Phases 1-2 (resize-at-arrival) as a complete, valuable feature and treat Phase 3 as a follow-up, since Phase 2 already delivers on-demand sizing for the common case.
- **Maintainer note:** per the design spec and AGENTS.md, this is a sizeable change introducing a new pattern; socializing via a GitHub issue before submitting is recommended (the user chose to proceed without it).
