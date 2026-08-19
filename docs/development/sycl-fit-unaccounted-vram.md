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

- is **~1.2 GiB immediately at load**, roughly independent of KV cache type, and
- **grows with context depth**, at **3.56 KiB/token with f16 KV and 7.64 KiB/token with q8_0 KV**.

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

`--fit-target 1024` became an effective 2160 MiB. Even so, 2160 MiB < the 2265–3562 MiB that goes
unaccounted. **The safety margin is smaller than the error it exists to absorb.** Any work here should
first establish what transforms 1024 → 2160, so that a `--fit-target 128` request means 128 MiB.

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

### Class C — ~1.2 GiB fixed, present at load, unattributed

This is the **largest single chunk** and it is *not* flash-attention-related — it is already present
before any prefill, and is nearly identical for f16 and q8_0. **`--fit-target 128` is unreachable
until this is attributed.** Candidates, none yet confirmed:

- Level Zero context / module / kernel-binary residency for a large JIT'd backend
- `sycl_reorder_temp_buffer` and the quantized-weight reorder path (`ggml-sycl.cpp:3860-3895`);
  buffers are transient but the peak may be retained by the pool
- `--load-mode mlock` interaction
- driver-side allocation granularity / rounding over ~866 tensors

---

## 4. Proposed work, staged

### 4.1 Stage 1 (prerequisite): attribute every device allocation

Without this, Class C is guesswork. Add, behind an env guard in the style of the existing
`GGML_SYCL_MKL_FA_DEBUG` (`fattn.cpp:278`):

1. A counter around every device allocation that does **not** pass through a `ggml_backend_buffer` —
   `ggml_sycl_pool_leg::alloc`, `ggml_sycl_pool_vmm::alloc` (`:1744` is the failing site),
   `ggml_sycl_fattn_kv_buffers::kv_buffer::ensure_half`, `sycl_reorder_temp_buffer`, and any bare
   `sycl::malloc_device` / `sycl::malloc_host`.
2. Log request size, running total, and high-water per site.
3. One line in `ggml_sycl_flash_attn_ext_onednn` reporting `n_kv`, `Hkv`, `d`, and the exact bytes
   requested for K, V, Q, mask, out.

Ship the pool high-water as a gauge too — this deployment already scrapes `llamacpp:*` into
Mimir/Grafana, so exposing `pool_size` / high-water makes the whole class observable without a rebuild
next time. Suggested: `llamacpp:sycl_pool_bytes`, `llamacpp:sycl_pool_peak_bytes`.

**Exit criterion:** the ~1.2 GiB at load is attributed to named call sites, summing to within ~50 MiB.

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
