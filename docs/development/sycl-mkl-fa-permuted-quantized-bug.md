# SYCL: MKL flash attention returns wrong results for a permuted quantized KV cache

Draft bug report. Reproduced on pristine `origin/master` (`749f688fc`) with none of our
local changes applied.

## Reproduction

```
$ test-backend-ops -o FLASH_ATTN_EXT -b SYCL0
FLASH_ATTN_EXT(hsk=256,hsv=256,nh=2,nr23=[16,1],kv=1025,nb=64,mask=1,sinks=0,
               max_bias=0.000000,logit_softcap=0.000000,prec=f32,
               type_K=q8_0,type_V=q8_0,permute=[0,2,1,3],kv_view=1,
               v_is_view_of_k=0): FAIL
[FLASH_ATTN_EXT] ERR = 1.302934477 > 0.000500000
```

An error of 1.3 against the CPU reference is garbage output, not a tolerance miss. The
case is one of those added in `70aff2525` ("metal : dequantize quantized KV to F16 before
flash attention", #27390), so it is newly *exposed*, not newly broken.

Environment: Intel Arc Graphics iGPU (Xe-LPG, Core Ultra 7 255H), Level Zero driver
1.15.38308, oneAPI 2026.1.1, `GGML_SYCL_F16=ON`.

## Which kernel is at fault

`GGML_SYCL_ENABLE_MKL_FA=0` makes the case pass, `GGML_SYCL_FA_ONEDNN=0` does not:

| configuration | result |
|---|---|
| defaults | FAIL, ERR = 1.282 |
| `GGML_SYCL_ENABLE_MKL_FA=0` | **PASS** |
| `GGML_SYCL_FA_ONEDNN=0` | FAIL, ERR = 1.276 |
| both disabled | **PASS** |

So it is the oneMKL flash-attention path (`ggml/src/ggml-sycl/fattn-mkl.cpp`).

## Minimal trigger

Varying one attribute at a time from the failing case:

| variant | result |
|---|---|
| as above (q8_0, permuted, nb=64) | **FAIL** |
| `kv=1024` instead of 1025 | **FAIL** — not a KV-padding issue |
| `permute=[0,1,2,3]` | PASS — **the permutation is required** |
| `hsk=hsv=128` | **FAIL** — not specific to head size 256 |
| `type_K/V=f16` | PASS — **quantized KV is required** |
| `nb=1` (generation-shaped) | PASS — needs ≥32 query columns, i.e. the MKL gate |

Trigger: **quantized K/V + a permuted KV view + a batch large enough to select MKL.**

`kv_view=1` means K and V are created as views of a larger buffer, the way a real KV cache
is -- it does *not* mean V is a view of K, which is the separate `v_is_view_of_k` parameter
(0 here) that upstream split out in #27394. So this is the ordinary KV-cache shape, not an
MLA one.

## Root cause

`fattn-mkl.cpp:464`:

```c
const bool k_interleaved = ((int64_t)K->ne[1] * K->nb[1] != K->nb[2]) && K->ne[2] > 1;
```

This is meant to detect a Gemma-style layout where KV heads are interleaved *within* a
row. But `ggml_permute(K, 0, 2, 1, 3)` swaps dims 1 and 2, so `ne[1]*nb[1] != nb[2]`
holds for any permuted view, and the heuristic reports interleaved for a tensor that is
merely permuted.

`mkl_fa_make_desc` then acts on it (`fattn-mkl.cpp:288-300`):

```c
const bool gemma = interleaved &&
    ((int64_t)T->nb[2] < (int64_t)T->ne[1] * (int64_t)T->nb[1]);
if (gemma) {
    d.s01 = (int64_t)n_kv_heads * blk_per_row;   // reconstructed, ignores T->nb
    d.s02 = blk_per_row;
} else {
    d.s01 = d.nb1 / d.ts;                        // the tensor's real strides
    d.s02 = d.nb2 / d.ts;
}
```

For a permuted view the real strides (`else` branch) are correct, but the `gemma` branch
discards them and reconstructs from `n_kv_heads`, so the per-chunk dequantization
(`mkl_fa_dequant_chunk`) reads from the wrong addresses. An F16 cache never reaches this
code, which is why f16 passes on the identical shape.

## Suggested fix

Not attempted here, because a correct discriminator depends on the intended Gemma
interleaved layout, and getting it wrong would silently corrupt that path instead —
the same class of bug, moved. Two options for whoever owns this code:

1. **Discriminate properly.** The heuristic needs to distinguish "heads packed within a
   row" from "dims 1 and 2 transposed". A permuted view still has `nb[0] == type_size` and
   strides that correctly describe the data, so a test on the tensor's own consistency
   (rather than on the relative magnitudes of `nb[1]` and `nb[2]`) would separate them.
2. **Fail safe.** Exclude non-contiguous quantized K/V from the MKL gate so it falls
   through to TILE, which is proven correct here. Conservative, and only costs performance
   for layouts that currently produce wrong numbers — though it would also drop Gemma
   interleaved shapes off the MKL path.

## Coverage note

The generated sweep gates quantized K/V on `hsk != 64 && hsk != 72`
(`tests/test-backend-ops.cpp:9840`), so quantized caches are not covered at head size
128 — the size essentially every current GQA model uses. Only the hand-written cases at
the end of the file exercise larger head sizes with a quantized cache, which is why this
went unnoticed. Worth widening independently of this bug.
