# Dynamic context sizing for llama-server

Date: 2026-06-22
Status: Design (pre-implementation)

## Problem

`llama-server` reserves the full `--ctx-size` context (KV cache + compute buffers)
at load time, for the entire process lifetime. In the single-client use-case this
is wasteful: most requests need far less context than the configured maximum, yet
the large context is always allocated. When VRAM is constrained, reserving a large
context forces model layers onto the CPU, slowing inference even for small requests.

We want the context size to be determined on demand. The single-client scenario
makes reload cost cheap relative to the performance and memory cost of permanently
reserving a large context.

## Goals

- Start with a small context and grow/shrink the allocation on demand.
- In the constrained-VRAM regime, small contexts should free VRAM so that *more
  model layers* run on the GPU (faster inference), not merely leave VRAM idle.
- Mid-stream context overflow must be transparent to the client: the response
  continues coherently after a pause, with no session restart and no silent token
  loss, until the configured maximum is genuinely reached.
- Opt-in; default behavior unchanged.

## Non-goals

- Multi-client / `n_parallel > 1` support (single-slot only; see Constraints).
- Multimodal mid-stream grow (text-only; resize-at-arrival still works).
- In-place context resize or in-place layer migration (not supported by the API).

## Key findings from the codebase

These findings shaped the design and reverse one earlier assumption.

1. **Sleep/wake already tears down and rebuilds.** `server_context_impl` has a
   sleep path (`tools/server/server-context.cpp`): `handle_sleeping_state(true)`
   calls `destroy()` (frees context + model), and waking calls
   `load_model(params_base)` to rebuild. Dynamic resize reuses this path.

2. **The KV cache is allocated per layer, not per token-position**
   (`src/llama-kv-cache.cpp` ~228-246). Each layer's K/V tensor is sized for the
   full context and placed entirely on that layer's device. There is no way to keep
   "the first 32k of positions on GPU and the rest on CPU" - the cache follows
   *layer placement*, not position.

3. **`common_fit_params` shrinks context before offloading layers, but only when
   `n_ctx == 0`** (`common/fit.cpp` step 2 vs step 3). When `n_ctx` is set
   explicitly (our tier value), context-reduction is skipped and fit re-runs *layer
   placement* for that exact context size. This is the mechanism we exploit: pin the
   tier, let fit decide the layer split for it.

4. **Reload clears all slot state.** `load_model()` calls `slots.clear()` and
   `slot.reset()` (~1304), wiping slots and the KV cache. Transparent continuation
   therefore requires snapshotting tokens before reload and reprocessing them after.

5. **The durable cross-reload artifact is the token sequence.** `slot.prompt.tokens`
   (a `server_tokens` list) holds prompt + all generated-so-far tokens. Saved KV
   bytes (prompt cache / checkpoints) are tied to a specific context and are invalid
   after a re-fit, so reprocessing is required regardless.

### Consequence: context-only resize vs. performance are mutually exclusive in the constrained regime

Because the KV cache is per-layer and layer placement cannot change in place,
getting model layers onto the freed VRAM at small tiers *requires* recreating the
model (a full reload + re-fit). A "context-only, weights resident" resize only frees
VRAM as idle headroom; it does not move layers back onto the GPU. In the target
constrained-VRAM regime we therefore use **full reload + re-fit per tier**. Weights
are re-read on each resize, but warm from the OS page cache (memory-bandwidth-bound,
not disk-bound). This reverses an earlier "context-only" assumption made before
finding (2) was understood.

## Mechanism

Resizing reuses the sleep/wake reload path:

1. Set `params_base.n_ctx` to the target tier.
2. Trigger the reload (`destroy()` then `load_model(params_base)`).
3. Because the tier is an explicit `n_ctx`, `common_fit_params` skips
   context-reduction and re-runs layer placement for that tier: smaller tiers put
   more layers on the GPU; larger tiers push layers to the CPU as needed.

## Tiers and resize policy

**Tiers:** powers of two from 32768 up to `--ctx-size`, plus `--ctx-size` itself as
the top tier when it is not a power of two. Example for `--ctx-size 200000`:
`{32768, 65536, 131072, 200000}`. If `--ctx-size <= 32768` there is effectively one
tier; the feature becomes a no-op (logged). The server starts at the smallest tier.

**Required tier for a request:** the smallest tier `>= prompt_tokens +
n_predict_budget`, where `n_predict_budget` reflects requested generation, capped at
`--ctx-size`. A prompt larger than `--ctx-size` is the existing "context exceeded"
error, unchanged.

**Grow (at request arrival):** if the incoming request needs a higher tier than the
current one, reload at the required tier before processing. No generation is in
flight, so this is straightforward.

**Shrink (with hysteresis):** evaluated at request arrival, never mid-stream. Shrink
only when the required tier is below the current tier *and* the requirement sits
comfortably below the next-tier-down boundary (suggested margin 10-15%), so requests
hovering near a tier edge do not thrash between sizes.

## Mid-stream grow-and-continue (transparent resize)

When a generating slot reaches `prompt.n_tokens() + 1 >= slot.n_ctx`
(`server-context.cpp` ~1886 and the ctx-shift path ~2829) and a higher tier exists:

1. **Snapshot** `slot.prompt.tokens` (prompt + generated-so-far) and the in-flight
   `server_task` plus generation/sampler state. Keep the streaming connection open.
2. **Reload** at the next tier via the sleep/wake path with
   `params_base.n_ctx = tier`, which re-fits the layer split. This clears slots and
   KV by design.
3. **Re-decode** the snapshotted tokens to rebuild the KV cache, then **resume
   generation** from the next token.

The client sees only a pause (the reload plus a reprocessing pass over existing
tokens). No restart, no truncation.

**Fallback:** if already at the max tier (`--ctx-size`), revert to today's behavior -
context-shift if enabled, otherwise a clean `STOP_TYPE_LIMIT`.

**Pause UX:** the SSE/HTTP connection stays open and tokens simply stop arriving until
reprocessing completes, then resume. (Known risk: a very long reprocessing pause
could approach client read timeouts; a keep-alive heartbeat is a possible later
enhancement but is out of scope here.)

## Constraints

- **Enablement:** new off-by-default flag (working name `--ctx-dynamic`).
  `--ctx-size` is the cap and top tier.
- **Single-slot only:** requires `n_parallel == 1`. Error at startup if
  `--ctx-dynamic` is combined with `n_parallel > 1`.
- **Multimodal:** mid-stream grow is gated off when `mctx != nullptr` (KV/token
  bookkeeping assumes text-only; cf. the `GGML_ABORT` in the ctx-shift path).
  Resize-at-arrival may still apply.
- **Standalone:** the feature lives in `server_context` and works whether or not the
  process is a `--models-preset` child. No router involvement is required.

## Testing

- Unit test the tier-selection math (tier set construction, required-tier lookup,
  shrink hysteresis boundary).
- Server integration test: drive a prompt/generation past the 32k boundary and
  assert the stream continues coherently, with a reload logged and no truncation.
- Regression: confirm default behavior (flag off) is unchanged.

## Risks and open questions

- **Maintainer acceptance.** This touches the model lifecycle and introduces a new
  pattern. Per `AGENTS.md`, large changes are likely to draw pushback unless
  socialized first. Opening a GitHub issue/discussion before implementation is
  recommended (the user has chosen to proceed without it for now).
- **Snapshot/restore complexity.** Preserving the active `server_task`, sampler
  state, and partial UTF-8 / stop-string buffers across a `destroy()`/`load_model()`
  is the most delicate part and needs careful, well-tested handling.
- **Reprocessing latency** for very large contexts may be significant; acceptable per
  the single-client tradeoff, but worth measuring.
