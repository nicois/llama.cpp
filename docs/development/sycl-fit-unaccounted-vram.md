# SYCL: `--fit` under-accounts device memory (unaccounted VRAM grows with context depth)

Investigation reference, 2026-08-19. Hardware: Intel Arc Pro B70 32 GB (Battlemage, `xe` driver),
SYCL build in a podman container, host `rainbow`. Model: `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL`.

**Goal of the follow-up work:** make `--fit-target 128` sufficient — i.e. after fitting, actual device
memory at *full context depth* should land within ~128 MiB of the fitter's projection. Today the error
is **2265 MiB (f16 KV)** to **3562 MiB (q8_0 KV)**.

---

## 1. Problem statement

`common_params_fit_impl` sizes `n_ctx` (and layer offload) so that its projected usage plus a margin
fits in free device memory. The projection is `mb.total() = model + context + compute`
(`src/llama-ext.h:72-74`, consumed at `common/fit.cpp:252`). On SYCL a large amount of device memory
is allocated **outside** that accounting, so the fitter over-commits. The gap:

- is **flat in `n_ctx`** — ~1561 MiB at load for the 27B at any context from 65536 to 262144
  (§2.5). **64 % of this is now identified**: the weight-reorder scratch, a whole-tensor staging buffer
  sized by `output.weight`, confirmed causally and to the MiB (§2.6),
- **grows further with context depth**, at **3.56 KiB/token with f16 KV and 7.64 KiB/token with
  q8_0 KV**, and
- **roughly doubles with MTP** (`--spec-type draft-mtp`), which adds +1484 to +2252 MiB *beyond* the
  weights and context it does account for — the single largest contributor.

Because quantizing the KV cache frees budget that `--fit` immediately spends on a deeper context,
*and* roughly doubles the per-token unaccounted cost, a quantized-KV configuration is materially more
likely to OOM than the f16 configuration it replaced. This presents to users as "quantizing the KV
cache made things worse", even though at a **pinned** `--ctx-size` quantized KV behaves correctly
(see §2.3).

llama.cpp already computes and prints the discrepancy: `common_memory_breakdown_print`
(`common/fit.cpp:817-905`) emits an `unaccounted` column = `total - free - self` (`fit.cpp:892`).
The fitter simply does not consult it.

---

## 2. Evidence

### 2.1 Measured OOM

`--fit on --fit-target 1024`, `--ctx-size` unset, `--spec-type draft-mtp`, `--batch-size 8192`,
`--ubatch-size 4096`, `--flash-attn on`. The q8_0 preset fitted to `n_ctx = 232704` and died during a
60 %-depth prefill:

```
level_zero backend failed with error: 39 (UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY)
  in function alloc at ggml/src/ggml-sycl/ggml-sycl.cpp:1744
  ggml_sycl_pool_vmm::alloc
  ← ggml_sycl_flash_attn_ext_onednn
  ← ggml_sycl_flash_attn_ext
  ← ggml_backend_sched_graph_compute_async
```

Note the backtrace names the allocation that asked *last*, not necessarily the largest consumer.

### 2.2 Depth ladder, MTP removed as a confound

Identical presets apart from `cache-type-k/v`; `--fit on`, `--fit-target 1024`, `-v`. Device memory
read from `ggml_backend_dev_memory` via the `llamacpp:vram_*` metrics (see §5.1).

| | f16 KV | q8_0 KV |
|---|---|---|
| fitted `n_ctx` | 182016 | 262144 (= `n_ctx_train`, ceiling-bound) |
| KV cache (`kv_cache_k+v_bytes`) | 11376 MiB | 8704 MiB |
| `self` (model + context + compute) | 30051 MiB | 28005 MiB |
| free @ load | 944.6 MiB | 2959.9 MiB |
| free @ 97 % depth | 338.5 MiB | 1088.7 MiB |
| depth reached | 174143 tok | 250774 tok |
| growth over ladder | +606.1 MiB | +1871.2 MiB |
| **unaccounted @ load** | ~1220 MiB | ~1250 MiB |
| **unaccounted @ depth** | **2265 MiB** | **3562 MiB** |
| **unaccounted growth / token** | **3.56 KiB** | **7.64 KiB** |

Per-rung f16: `+134 → +172 → +346 → +522 → +606 MiB` at 18k/63k/108k/153k/174k tokens.
Per-rung q8_0: `+235 → +637 → +1113 → +1621 → +1871 MiB` at 26k/91k/155k/220k/251k tokens.

The f16 config was memory-bound and finished with **338.5 MiB free**. The q8_0 config survived only
because it hit `n_ctx_train` and stopped early, leaving 2959.9 MiB instead of the target margin —
accidental headroom, not correct behaviour. With MTP consuming budget it *was* memory-bound, and OOMed.

llama.cpp's own shutdown breakdown for the f16 run, which agrees with the external reading
(`free = 338` vs measured 338.5):

```
memory breakdown [MiB] | total    free     self   model   context   compute    unaccounted |
  - SYCL0 (Arc Pro B70) | 32656 =   338 + (30051 = 16128 +   11525 +    2398) +        2265 |
```

### 2.3 Control: at a pinned `--ctx-size`, quantized KV is fine

Same model, `--ctx-size` pinned, 3 interleaved replicates each (stdev 0.0 MiB):

| `--ctx-size` | KV type | reported KV | device used | nominal saving | measured saving |
|---|---|---|---|---|---|
| 8192 | f16 | 512 MiB | 20341.1 MiB | — | — |
| 8192 | q4_0 | 144 MiB | 20009.9 MiB | 368 MiB | 331.2 MiB (−10 %) |

At `--ctx-size 32768`, all six K/V combinations were monotonic in cache size
(`ff` 22163 > `8f`/`f8` 21900 > `4f`/`f4` 21645 > `88` 21399 > `44` 20888 MiB). So **KV cache sizing
itself is correct** — `mb.context` tracked measured KV within ~150 MiB in every case. The defect is
entirely in the unaccounted term.

### 2.4 The effective margin is not `--fit-target`

```
common_params_fit_impl: cannot meet free memory target of 2160 MiB, need to reduce device memory by 5629 MiB
common_params_fit_impl: will leave 4210 >= 2160 MiB of free device memory, no changes needed
```

`--fit-target 1024` became an effective 2160 MiB. **This is deliberate, not a bug**:
`tools/server/server-context.cpp:1069` does `params_base.fit_params_target[i] += bytes`, adding the
draft/MTP context's measured `context + compute` to the margin so the second context has room. So the
MTP configurations reserve ~1136 MiB extra by design.

Even so, 2160 MiB < the 2265–3562 MiB that goes unaccounted. **The safety margin is smaller than the
error it exists to absorb.** Note the consequence for the goal: because the margin is a floor plus
reservations, a `--fit-target 128` request will not mean 128 MiB of free memory unless the reserved
terms are themselves accurate.

### 2.5 The accounting model

Sweep over `n_ctx` with weights, `--batch-size 8192` and `--ubatch-size 4096` held constant, MTP as a
controlled variable, `--fit off` on the pinned presets, measured at near-zero depth. Both models are
**dense** (`n_expert = 0`): 27B is `n_vocab 248320, n_embd 5120, n_ff 17408, n_layer 64`; 2B is
`n_embd 2048, n_ff 6144, n_layer 24`.

| config | n_ctx | KV | model | context | compute | `self` | **unacc** |
|---|---|---|---|---|---|---|---|
| 27B f16 | 65536 | 4096 | 16128 | 4245 | 1488 | 21861 | **1561** |
| 27B f16 + MTP | 65536 | 4096 | 16400 | 4694 | 1488 | 22583 | **3045** |
| 27B f16 | 131072 | 8192 | 16128 | 8341 | 2000 | 26469 | **1561** |
| 27B f16 + MTP | 131072 | 8192 | 16400 | 8790 | 2000 | 27191 | **3813** |
| 27B f16 (fitted) | 182016 | 11376 | 16128 | 11525 | 2398 | 30051 | **1659** |
| 27B f16 + MTP (fitted) | 115200 | 7200 | 16400 | 7798 | 1876 | 26075 | **5349** |
| 27B q8_0 (fitted) | 262144 | 8704 | 16128 | 8853 | 3024 | 28006 | **1690** |
| 27B q8_0 + MTP (fitted) | 197632 | 6562 | 16400 | 7160 | 2520 | 26081 | **5384** |
| 2B q8_0 (fitted) | 262144 | 1632 | 1908 | 1651 | 2400 | 5960 | **332** |

**Three of the four terms are exactly predictable:**

- `model` — weights on device. MTP adds exactly **+272 MiB**.
- `context` = `KV + 149 MiB` (27B), `KV + 598 MiB` (27B + MTP), `KV + 19 MiB` (2B). Constant per
  configuration, so **KV cache accounting is sound**.
- `compute` = `base + ubatch × n_ctx × 2 bytes`, with `base` = 976 MiB (27B), 352 MiB (2B). Exact to
  the MiB at every context size — 1488 / 2000 / 2398 / 3024 for 65536 / 131072 / 182016 / 262144.
  The `ubatch × n_ctx × 2` term is the F16 KQ mask, and it is why `--ubatch-size 4096` is expensive at
  long context: 2048 MiB of mask at `n_ctx` 262144.

**`unaccounted` is the only unexplained term, and it is flat in `n_ctx`:** 1561 MiB at both 65536 and
131072 — a 2× context change with no movement — and only 1659/1690 at 182016/262144. It is therefore
**not** the flash-attention staging, which would scale with `n_kv`. It scales sub-linearly with model
size (1561 vs 332 MiB for 8.45× the weight bytes, a ratio of 4.70).

**MTP is the single largest unaccounted contributor**: +1484 MiB at `n_ctx` 65536 and +2252 MiB at
131072 on top of the base term, over and above the +272 MiB of weights and +449 MiB of context that
*are* accounted. The two fitted MTP configurations show +3690/+3694 MiB, which does not fit a
monotonic relationship with `n_ctx` (5349 MiB at 115200 vs 3813 MiB at 131072) and is unexplained.

Candidate mechanism for the base term, **not confirmed**: `sycl_reorder_temp_buffer`
(`ggml-sycl.cpp:3864`) allocates a whole-tensor scratch through `sycl_ext_malloc_device`, which uses
`syclex::async_malloc`; the matching `async_free` returns memory to a **driver pool rather than the
OS**, so the peak stays resident and invisible. The arithmetic does not support it on its own, though:
the largest reordered tensor is the `output`/`lm_head` at 682 MiB (Q4_K) to 1288 MiB (Q8_0) for the
27B and 273–515 MiB for the 2B, i.e. a predicted ratio of 2.5 against a measured 4.70. So either an
additional mechanism is involved or the pool rounds. Two zero-code tests bisect it, both requiring
only a container restart:

```sh
GGML_SYCL_ENABLE_OPT=0          # skip the weight reorder entirely (slower)
GGML_SYCL_USE_ASYNC_MEM_OP=0    # use real malloc/free instead of the driver pool
```

If either collapses `unaccounted` from ~1561 MiB toward the ~450 MiB floor, the reorder path is
implicated. Otherwise use `GGML_SYCL_MEMTRACE=1` (§4.1), which attributes it directly.

### 2.6 Root cause of the load-time term: the weight-reorder scratch

Found by running the instrumentation of §4.1 on a **Meteor Lake iGPU** (Xe-LPG, `i915`), chosen
because oneDNN flash-attention is Battlemage-only (`fattn-onednn.cpp:29`) and is therefore *disabled*
there — so anything that still appears cannot be oneDNN.

`GGML_SYCL_MEMTRACE=1`, `Qwen3.8-27B-UD-Q2_K_XL`, `--ctx-size 8192 --ubatch-size 512`:

```
buffer   peak 10359 MiB      <- vs breakdown self 10359 (9567 + 661 + 130). Exact.
async    peak   520 MiB      <- the reorder scratch
pool_vmm peak     2 MiB
fattn_kv          never allocated
```

`buffer` reconciles with llama.cpp's own `self` to within 1 MiB, which validates the hooks. And the
A/B is causal:

| | reorder on | `GGML_SYCL_ENABLE_OPT=0` |
|---|---|---|
| `buffer` peak | 10359 MiB | 10359 MiB |
| `async` peak | **520 MiB** | **never allocated** |
| unaccounted | **522 MiB** | **0 MiB** |

**The mechanism.** `reorder_qw_*` allocates a scratch buffer the size of the *whole tensor*
(`sycl_reorder_temp_buffer`, `ggml-sycl.cpp:3864`) via `sycl_ext_malloc_device`. Peak demand is
therefore the largest reordered tensor — in practice `output.weight` — and it is invisible to the
memory breakdown. The sizes match analytically, not approximately:

| model | measured `async` peak | `output.weight` = `n_vocab × n_embd` at… |
|---|---|---|
| 27B UD-Q2_K_XL | 520 MiB | **521 MiB** at Q3_K |
| 27B UD-Q4_K_XL | 994 MiB | **995 MiB** at Q6_K |

(1.271 G elements for `248320 × 5120`; unsloth's `UD-*_XL` mixes keep `output.weight` at a higher bpw
than the name suggests, which is why the earlier largest-tensor estimate assuming a uniform quant was
wrong.)

This explains the **flat-in-`n_ctx`** signature in §2.5: a per-largest-tensor scratch has no `n_ctx`
dependence. For the 27B UD-Q4_K_XL it accounts for **994 of the 1561 MiB** the B70 shows at load
(64 %). The residual ~567 MiB is most likely oneDNN plus pool growth under real compute — this run did
2 tokens at `ubatch 512`, versus the B70's `ubatch 4096` with oneDNN active. Running memtrace on the
B70 will attribute the remainder exactly.

**`GGML_SYCL_USE_ASYNC_MEM_OP=0` does not avoid the cost** — the same 520 MiB is allocated, merely
attributed to `direct` because `sycl_ext_malloc_device` falls through to `ggml_sycl_malloc_device`.
Only the *free* path changes: `async_free` returns to a driver pool (retained), a real `free` returns
it. So the peak must be budgeted either way; retention only decides whether it also permanently eats
the margin.

**Fix options**, cheapest first:

1. **Reorder in chunks.** The scratch is a staging copy; slicing the tensor bounds it to a fixed
   working-set instead of the largest tensor. Removes ~1 GiB of demand outright and needs no
   accounting changes.
2. **Allocate it through a `ggml_backend_buffer`** so it lands in `mb.model` and the fitter sees it.
3. **At minimum**, add the peak to the fit budget the way the draft/MTP context already is
   (`tools/server/server-context.cpp:1069`).

**Caveat if reproducing on an iGPU:** the breakdown's `unaccounted` column is meaningless on UMA
hardware — `total`/`free` are *system* RAM, so `total - free - self` absorbs every other process
(observed: 44339 MiB). Use `GGML_SYCL_MEMTRACE`, which is process-local, not the breakdown.

---

## 3. Root cause: three classes of unaccounted device memory

### Class A — flash-attention K/V staging (depth-dependent, the dominant growth term)

`ggml_sycl_flash_attn_ext_onednn` (`ggml/src/ggml-sycl/fattn-onednn.cpp:231-324`) allocates, per call,
from the SYCL pool:

- `Qf` — `H * q * d` halves
- `Kf_pool`, `Vf_pool` — **`Hkv * seq * d` halves each**, where `seq = K->ne[1]` is the *full current
  `n_kv`* (`fattn-onednn.cpp:217`)
- `scbuf`, `outf`

Both the F16 branch (`:240-246`) and the quantized branch (`:247-313`) allocate the same
`Hkv * seq * d` halves — there is **no zero-copy path for an already-F16 cache**. These scale linearly
with `n_kv`, are pool-allocated at graph-execution time, and are invisible to `mb.compute`.

`launch_fattn` (`ggml/src/ggml-sycl/fattn-common.hpp:927-1000`) has a second instance of the same
problem for the TILE/VEC kernels, using `ggml_sycl_fattn_kv_buffers` — **persistent, grow-only,
16 MiB-chunked, freed only at context destruction** (`fattn-buffers.hpp:20-47`,
`fattn-buffers.cpp:15-56`). TILE always requests F16 K/V (`fattn-tile.hpp`, every `launch_fattn` call
passes `need_f16_K = need_f16_V = true`), so a quantized cache always materialises a copy there.

`ggml_sycl_flash_attn_ext_mkl` (`fattn-mkl.cpp:241-306`, `mkl_fa_dequant_chunk`) is the counter-example to copy from — it
dequantizes **one KV-head chunk at a time**, with an explicit comment that whole-cache dequant makes
the "footprint scale with context".

Kernel selection matters: **oneDNN is tried first** (`fattn.cpp:136-140`), gated on `Q->ne[1] >= 32`,
Battlemage arch, and — for non-F16 KV — `K->ne[1] >= 1024`. So prefill takes the whole-cache path, not
the chunked MKL one.

Why quantized KV is ~2× worse per token is **not yet established**. Both branches allocate the same
bytes per call, so the 3.56 vs 7.64 KiB/token difference must come from *which* paths run and how the
pool high-water accumulates (e.g. F16 caches can be consumed in place by VEC and by the
`need_f16 && type == F16` short-circuit in `launch_fattn`, whereas a quantized cache must be
materialised in every F16-requiring path). This needs the instrumentation in §4.1 to confirm.

### Class B — the SYCL scratch pool itself

`ggml_sycl_pool_leg` (`ggml-sycl.cpp:1536`) and `ggml_sycl_pool_vmm` (`:1675`) both track
`pool_size`, retain their high-water mark for the process lifetime, and are reported nowhere.
Class A flows through these, but so does every other op's scratch.

### Class C — a ~450 MiB floor at load, plus a large-`n_ctx` term that turns on somewhere above 32768

Measured at near-zero depth (one 4-token request, then swap out to flush the shutdown breakdown),
same weights and `--ubatch-size 4096` throughout:

| `n_ctx` | KV | `self` | unaccounted |
|---|---|---|---|
| 8192 (q4_0) | 144 MiB | 17462 | **462 MiB** |
| 8192 (f16) | 512 MiB | 17829 | **513 MiB** |
| 32768 (q8_0) | 1088 MiB | 18598 | **455 MiB** |
| 32768 (mixed q4_0/f16) | 1312 MiB | 18822 / 18821 | **439 / 440 MiB** |
| 32768 (mixed q8_0/f16) | 1568 MiB | 19078 / 19077 | **439 / 439 MiB** |
| Qwen3.8-**2B**, `model = 1908` | 1651 MiB | 5960 | **361 MiB** |

Two conclusions:

- **The load-time floor is ~450 MiB and is independent of `n_ctx` and of KV cache type.** A 2B model
  still carries 361 MiB despite 8.5× fewer weight bytes, so it is only weakly weight-dependent. Of
  this, **126 MiB is the router process itself** — a llama.cpp process with SYCL initialised and *no*
  model loaded occupies 126 MiB of device memory (32656 − 32530), which is device-wide and therefore
  outside any child's accounting. Note this is automatically handled by the fitter, since it queries
  free memory device-wide.
- **The jump above 32768 is a step, not a slope.** The 65536/131072 sweep in §2.5 measured 1561 MiB at
  *both*, so between 32768 (~450 MiB) and 65536 (~1561 MiB) something switches on and then stops
  scaling. A per-token model is therefore falsified; look for a threshold, e.g. a kernel-selection
  change (`GGML_SYCL_MKL_FA_DEBUG=1` names the kernel per call) or a pool reserve granularity.

Incidental regularity worth knowing: `mb.context` was **exactly `KV + 149 MiB`** in all seven 27B
configurations, so `mb.context` itself is well-behaved and the KV cache accounting is sound.

Remaining candidates for the floor, none yet confirmed:

- Level Zero context / module / kernel-binary residency (bounded above by the 126 MiB router baseline)
- `sycl_reorder_temp_buffer` and the quantized-weight reorder path (`ggml-sycl.cpp:3860-3895`);
  buffers are transient but the pool retains the peak — would explain the weak weight-dependence
- driver-side allocation granularity / rounding over ~866 tensors

---

## 4. Proposed work, staged

### 4.1 Stage 1 (prerequisite): attribute every device allocation — **implemented**

`GGML_SYCL_MEMTRACE=1` (`ggml/src/ggml-sycl/memtrace.{hpp,cpp}`) tracks every device allocation by
site with live and peak bytes, logging whenever the peak grows by `GGML_SYCL_MEMTRACE_STEP` MiB
(default 64) and whenever anything queries device memory — so a `/metrics` scrape also dumps the
attribution. Sites:

| site | meaning |
|---|---|
| `buffer` | passes through a `ggml_backend_buffer`; should reconcile with `model + context + compute` |
| `pool_leg` | `ggml_sycl_pool_leg`, retained until context teardown |
| `pool_vmm` | `ggml_sycl_pool_vmm` physical pages mapped into the reserved range |
| `async` | `syclex::async_malloc`; `async_free` returns to a **driver pool, not the OS** |
| `fattn_kv` | `ggml_sycl_fattn_kv_buffers`, grow-only F16 K/V staging |
| `direct` | anything else |

Hooks sit at four choke points: `ggml_sycl_malloc_device` (which gained a defaulted `site` parameter,
so existing callers are unchanged), the VMM pool's page mapping, `sycl_ext_malloc_device` /
`sycl_ext_free`, and `ggml_sycl_fattn_kv_buffers::ensure_half`. Off and near-free when unset.

**Exit criterion:** the ~1561 MiB base term (27B) and the ~1484–2252 MiB MTP term are attributed to
named sites summing to within ~50 MiB, and `buffer` reconciles with the breakdown's `self`.

Still to add: a line in `ggml_sycl_flash_attn_ext_onednn` reporting `n_kv`, `Hkv`, `d` and the exact
bytes requested for K, V, Q, mask, out; and exposing the peak as `llamacpp:sycl_pool_peak_bytes` so
this is observable in Grafana without a rebuild.


### 4.2 Stage 2: reserve FA staging in the compute buffer (port the CUDA fix)

CUDA already solved Class A. Upstream #23907 / `f8f0a47a5` ("cuda: reserve space for quantize kv-cache
at startup"):

- `ggml_cuda_flash_attn_ext_get_f16_extra_data` (`ggml/src/ggml-cuda/fattn-common.cuh:47-85`) lays out
  the F16 K/V staging **immediately after the FA output tensor's own bytes**, 128-byte aligned.
- `ggml_cuda_flash_attn_ext_get_alloc_size` (`ggml/src/ggml-cuda/fattn.cu:545-568`) returns
  `f16_extra.end - dst->data`, deriving `need_f16_K/V` from the selected kernel.
- `ggml_backend_cuda_buffer_type_get_alloc_size` (`ggml/src/ggml-cuda/ggml-cuda.cu:906-921`) adds it
  when `tensor->op == GGML_OP_FLASH_ATTN_EXT`.

The graph allocator therefore reserves the staging inside the compute buffer, it lands in
`mb.compute`, and the fitter sees it. SYCL's equivalent
(`ggml_backend_sycl_buffer_type_get_alloc_size`, `ggml-sycl.cpp:950-963`) only handles quantized row
padding and has no FA case.

**This is sized correctly for free.** `llama_context::sched_reserve` (`src/llama-context.cpp:585-640`)
calls `memory->init_full()` and reserves the worst-case graph, so during the reserve the FA node's
`K->ne[1]` is the full padded `n_ctx`. A `get_alloc_size` hook is therefore called with worst-case
`n_kv`, and the depth-dependent growth disappears from runtime entirely: it is paid once, up front,
and reported.

Work items:

1. Add `ggml_sycl_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst)` mirroring the
   CUDA layout, deriving `need_f16_K/V` from `ggml_sycl_get_best_fattn_kernel` (TILE ⇒ both true;
   VEC ⇒ only for F32; ONEDNN ⇒ both, plus `Qf`/`outf`/mask staging; MKL ⇒ chunk-sized only).
2. Call it from `ggml_backend_sycl_buffer_type_get_alloc_size` for `GGML_OP_FLASH_ATTN_EXT`.
3. Point `launch_fattn` (`fattn-common.hpp:927-928`) at the reserved region instead of
   `ggml_sycl_fattn_kv_buffers`, and **delete `fattn-buffers.{cpp,hpp}`** — the grow-only persistent
   buffers become dead once the space is reserved. (They were added by `e20b83930`, upstream #22732,
   to cut pool churn; reservation subsumes that.)
4. Point the oneDNN path's `Kf_pool`/`Vf_pool`/`Qf`/`outf` at the reserved region.
5. Mirror the `memset` of the padding region that CUDA does (`ggml-cuda.cu:764-768`) if any kernel
   can read uninitialised staging.

**Expect fitted `n_ctx` to drop.** That is the fix working: the previous larger value was
over-committed. Quantify the reduction — if it is severe, §4.3 recovers it.

### 4.3 Stage 3 (optional, recovers context): chunk the oneDNN staging

Reservation makes the cost *visible*; chunking makes it *small*. `mkl_fa_dequant_chunk` (`fattn-mkl.cpp:306`) already
dequantizes per KV-head chunk with a bounded buffer. Applying the same slicing to
`ggml_sycl_flash_attn_ext_onednn` turns an `O(n_kv)` reservation into `O(chunk)`, which directly buys
back context. Do this only after Stage 2, so the before/after is measurable.

A cheap interim mitigation exists today with no code change, documented at
`docs/backend/SYCL.md:801-802`:

- `GGML_SYCL_FA_ONEDNN=0` — fall back to the chunked MKL kernel
- `GGML_SYCL_FA_ONEDNN_MAX_KV=32768` — keep oneDNN for shallow contexts only

Neither has been measured on this hardware yet; doing so is a good way to bound the size of the
Class A term before writing any code.

### 4.4 Stage 4: make the margin mean what it says

- Establish and document what turns `--fit-target 1024` into an effective 2160 MiB (§2.4).
- Consider having the fitter consult `unaccounted` from a probe context, so it self-corrects on
  backends whose accounting is incomplete, rather than silently over-committing.

### 4.5 Independent smaller defects found along the way

1. **`fit-params` device-compute estimate is wrong and context-independent.** `--fit-print on`
   reported `compute = 4936 MiB` at *both* `-c 135680` and `-c 182016`, while the real loaded context
   reports `2398 MiB` and self-verifies (`~llama_context: SYCL0 compute buffer size is 2398.1252 MiB,
   matches expectation of 2398.1252 MiB`). The in-server fit probe reported a third value, 3024 MiB.
   `fit-params` output is therefore unsafe for capacity planning.
2. **The `unaccounted` column is nonsense during the fit probe** — `-35246`, `-17102`, `-27566` MiB,
   because `free` is sampled before the probe context is allocated. Suppress the column when the
   context is not resident.
3. **`fit-params` cannot model MTP configs.** `--spec-type` is tagged
   `{SPECULATIVE, SERVER, CLI}` (`common/arg.cpp:4162`), so the tool cannot reproduce a server fit for
   any `draft-mtp` model — exactly the configs most likely to need capacity planning. Adding
   `LLAMA_EXAMPLE_FIT_PARAMS` to the spec options' `set_examples` is a one-line-per-option fix.
4. **`--fit-print` is unreachable from llama-server, including via its env var.** `LLAMA_ARG_FIT_ESTIMATE`
   is silently ignored because env vars are only read for options in the current example's filtered
   list (`common/arg.cpp:785`, filtered at `:1436`). Silent ignore is worse than the error the flag
   itself produces.

---

## 5. Verification harness

`scripts/server-vram-probe.py` implements all three measurements used above:

```sh
# §2.3 control: A/B pinned-context presets with error bars
scripts/server-vram-probe.py --url http://host:8080 \
    replicate --depth 6900 --reps 3 preset-f16 preset-q4

# §2.2 depth ladder over each model's fitted n_ctx
scripts/server-vram-probe.py --url http://host:8080 \
    ladder preset-max-f16 preset-max-q8

# used VRAM at fixed prefill depths
scripts/server-vram-probe.py --url http://host:8080 \
    compare --depths 8000,16000,30000 preset-f16 preset-q4
```

It refuses to report if `free == total` (see §5.1), settles 15 s before each
reading, uses prefix caching so a ladder is incremental, and treats an OOM at
depth as a result rather than a harness crash. **Growth figures require a freshly
loaded child** — a router reuses an already-loaded model whose scratch pool is
already at high-water, which reports `+0.0` growth; pass two or more models so
they alternate.

### 5.1 Measuring device memory

Use `llamacpp:vram_total_bytes - llamacpp:vram_free_bytes` from `/metrics`. Chain:
`tools/server/server-context.cpp:2470` → `ggml_backend_dev_memory` →
`ggml_backend_sycl_get_device_memory` (`ggml-sycl.cpp:5430`) →
`dpct::device_ext::get_memory_info` (`ggml/src/ggml-sycl/dpct/helper.hpp:687-709`), which returns
`sycl::ext::intel::info::device::free_memory` — the real device-wide figure `--list-devices` prints,
so it captures allocations ggml does not track.

Two gotchas, both learned the hard way:

- It **silently degrades to `free = total`** without `ZES_ENABLE_SYSMAN=1`. Assert `free != total`.
- In router mode, **wait ~15 s after a model swap** before reading, or the previous child's VRAM is
  still resident and readings scatter by ~190 MiB. With settling, replicates are bit-identical
  (stdev 0.0 MiB across 3 interleaved runs).

To see `unaccounted` from inside the process, run llama-server with **`-v`** (`--log-verbose`, sets
`verbosity = INT_MAX`). Required because `common/common.cpp:1302` only passes `GGML_LOG_LEVEL_DEBUG`
into the fitter when `verbosity >= LOG_LEVEL_DEBUG` (= 5). Note `-v` takes no argument; `-lv 2` is a
*different* option and would set level 2, below the threshold. The breakdown at
`tools/server/server.cpp:532` only prints at clean shutdown, and is skipped entirely on an OOM abort.

### 5.2 Diagnosing kernel selection

`GGML_SYCL_MKL_FA_DEBUG=1` logs `[FA-DISP] #n MKL|ONEDNN|VEC|TILE D=… n_kv=…` per FA call
(`fattn.cpp:278-305`). One line per layer per ubatch, so use a shallow request or cap the output.

### 5.3 Acceptance test

Two presets identical apart from `cache-type-k/v` (f16 vs q8_0), `--fit on`, `--ctx-size` unset, `-v`.
Prefill a ladder at 10/35/60/85/97 % of the fitted `n_ctx` using prefix caching so the ladder is
incremental, settling 15 s before each reading.

**Pass:** with `--fit-target 128`, `unaccounted` at 97 % depth is ≤ ~128 MiB for **both** cache types,
and neither OOMs. Additionally the q8_0 config must be driven to be *memory*-bound rather than
`n_ctx_train`-bound (pin `--ctx-size` at the ceiling, or use a model with a higher `n_ctx_train`) —
otherwise the test passes on accidental headroom, which is how the original OOM escaped notice.

Regression guard: keep the pinned-`--ctx-size` control from §2.3. Quantized KV must remain monotonic
in cache size, and the f16→q4_0 saving at `--ctx-size 8192` must stay ≈331 MiB of a nominal 368 MiB.

---

## 6. References

- Upstream #19979 — same symptom on ROCm: `-fa 1 -ctk q4_0 -ctv q4_0`, VRAM creeps during prefill
  until OOM; Vulkan stays flat.
- Upstream PR #21830 (closed) — the diagnosis: *"the kernels would increasingly allocate more vram as
  they require the entire kv cache to be dequantized to f16 types … completely cancelling out any
  benefit to be had from using quantized kv cache types for non-synthetic workloads."*
- Upstream #23907 / `f8f0a47a5` — **the CUDA fix to port.**
- `e20b83930` (upstream #22732) — the SYCL partial mitigation: removed pool churn via persistent
  buffers, but left the allocation full-cache-sized and unaccounted.
- Upstream #26409 — SYCL `--split-mode tensor` hangs with a quantized KV cache (possibly related).
