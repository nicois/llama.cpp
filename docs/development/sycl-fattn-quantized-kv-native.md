# Quantized KV caches and flash attention: why in-kernel dequantization was rejected

**Status: rejected.** An implementation existed, was correct, and was removed. This
document is kept for the measurements and for the two findings that came out of it, one of
which is an unreported upstream bug. Do not revive the approach without reading §5.

## 1. The observation that started it

Arc Pro B70 (BMG-G31, oneAPI 2026.1), Qwen3-27B-Q4 at 48,848 tokens of context:

| KV type | cache bytes/token | generation |
|---|---|---|
| f16  | 64.0 KiB | **14.74 t/s** |
| q8_0 | 34.0 KiB | 12.92 t/s |

A q8_0 cache holds 47% fewer bytes and decodes 12% *slower*. Users pick a quantized cache
to fit more context and reasonably expect it not to cost speed.

## 2. Mechanism

`launch_fattn` converts a non-F16 cache to F16 before calling the kernel, because the TILE
and MMA kernels are typed on `half2`. The conversion is not incremental — it covers
`ggml_nelements(K)`, the whole cache, per layer per call, so during generation the entire
cache is re-materialized as F16 for every token even though one position was appended.

Per element of KV:

| KV type | read quantized | write F16 | read F16 | total | vs f16 |
|---|---|---|---|---|---|
| f16  | –    | –   | 2.0 | **2.0**  | 1.0x |
| q8_0 | 1.06 | 2.0 | 2.0 | **5.06** | 2.5x |
| q4_0 | 0.56 | 2.0 | 2.0 | **4.56** | 2.3x |

Quantizing shrinks the smallest term and adds two larger ones. Calibrating against the
measurements above (taking weights as ~17 GB/token) gives 324 GB/s from the q8_0 point and
298 GB/s from the f16 point — a 9% spread, so the model is at least self-consistent.

**That model turned out to be the wrong one.** See §4.

## 3. What was built, and that it worked

`flash_attn_tile_load_tile_q` dequantized straight into the SLM tile that the F16 loader
fills with bulk copies, leaving the tile layout and every consumer untouched. It reused
`get_dequantize_V<type, T, ne>()` — the functor family the VEC kernel already uses — so
there was no new dequantization logic, only a new tile producer. Types and byte strides
were threaded through `flash_attn_tile_iter_KQ`, `flash_attn_tile_iter` and
`flash_attn_tile`, since neither the row nor the head-dimension offset can be folded into
a pointer for a block-quantized type.

Correctness was solid: 4017/4017 in `test-backend-ops -o FLASH_ATTN_EXT`, with the native
path confirmed taken, and again with it disabled. Deliberately corrupting the element
offset failed 168 cases, so the suite genuinely exercised it rather than passing vacuously.

Kernel microbenchmark, Xe-LPG iGPU, `test-backend-ops perf`, kv=7680, one query column:

| case | converting | native | ratio |
|---|---|---|---|
| q8_0 | 1077.88 us/run | 644.83 us/run | 1.67x |
| q4_0 | 1048.97 us/run | 810.74 us/run | 1.29x |
| f16 (reference) | 301.45 us/run | – | – |
| nb=512 (prompt-shaped control) | 114136 us/run | 118495 us/run | within 4%, converts in both |

## 4. Why it was rejected

**It regressed the target hardware.** Same B70, same model, same depths:

| depth | staging | native | delta |
|---|---|---|---|
| 1,964 | 21.50 | 20.90 | −2.8% |
| 25,407 | 16.15 | 13.19 | −18.3% |
| 48,848 | 12.92 | 9.57 | −25.9% |

Causation is established, not inferred: the f16 preset reproduced its numbers exactly
across both builds (21.77 / 17.57 / 14.74), and forcing the conversion path back on the
*same* build restored 21.52 / 16.13 / 12.92. The regression scales with depth, i.e. with
cache size.

**Upstream moved the other way.** Commit `70aff2525` (2026-08-20), "metal : dequantize
quantized KV to F16 before flash attention (#27390)", adds a pre-pass that dequantizes K
and V into a contiguous F16 scratch buffer and runs the existing F16 kernels on it,
explicitly "instead of the in-kernel dequantization path". Its gate is type-only, the
attention kernels are untouched, and the pre-pass is a separate stride-aware kernel
dispatched once for K and once for V. So Metal — the backend whose in-kernel design this
work was ported *from* — has retired that design, and SYCL's existing staging already
matches where upstream is going.

**The lesson worth keeping**: a kernel microbenchmark said 1.67x faster while the model
end-to-end got 26% slower. The microbenchmark measures the FA operation in isolation and
therefore cannot see that a separate conversion pass overlaps with other work, whereas
in-kernel dequantization serialises with the attention maths. GPU telemetry from the same
B70 during steady-state decode is consistent with this: the copy engine sat at 77% busy
concurrently with compute at 78%. "Fewer DRAM bytes must be faster" is not a valid
inference when the bytes you removed were being moved in the shadow of other work.

## 5. What would still be worth doing

Every backend, including upstream's new Metal pre-pass, converts the **entire** cache on
every call. Generation appends one position per step, so this is O(n_kv) work per token to
add O(1) of new data — the single largest inefficiency left in this area, and it is
backend-independent.

An incremental conversion is the obvious fix and the obstacles are not in the kernel:

* **Invalidation.** The cache is written between FA calls, positions are overwritten on
  context shift and slot reuse, and a cached F16 copy would have to be invalidated on any
  of that. Whether a backend can detect this cheaply and correctly, or needs plumbing from
  `llama_context`, is the open question.
* **Memory.** A persistent F16 copy forfeits the saving that motivates a quantized cache
  at all. A sliding window over recent positions, or converting only the tail, may be the
  middle ground.

## 6. Two findings worth acting on independently

**A test coverage gap.** `tests/test-backend-ops.cpp` gates quantized K/V on
`hsk != 64 && hsk != 72`, so a quantized cache is never tested at head size 128 — the size
essentially every current GQA model uses. 36 generation-shaped cases at hsk=128 were added
(both GQA shapes, and kv values that are not all multiples of `FATTN_KQ_STRIDE` so the
tails are covered). Independent of everything above.

**An upstream SYCL correctness bug.** On pristine `origin/master`, built and run with none
of this work applied:

    test-backend-ops -o FLASH_ATTN_EXT -b SYCL0
    FLASH_ATTN_EXT(hsk=256,hsv=256,nh=2,nr23=[16,1],kv=1025,nb=64,mask=1,sinks=0,
                   max_bias=0,logit_softcap=0,prec=f32,type_K=q8_0,type_V=q8_0,
                   permute=[0,2,1,3],kv_view=1): FAIL   ERR = 1.30 > 0.0005

An error of 1.3 is garbage output, not a tolerance miss. The case comes from the test
cases added in `70aff2525`, so it is newly exposed rather than newly broken. The shape is
quantized KV at head size 256, permuted, with V as a view of K, in a prompt-shaped batch.
Worth reporting: it is reproducible on a clean tree with an exact test case, and it sits
in territory upstream's own sweep does not cover.

## 7. Interaction with the FA staging reservation

The staging reservation on the FA output tensor remains necessary and correct — the
conversion still happens, so the worst-case graph still needs the space. Nothing about
rejecting in-kernel dequantization changes it.
