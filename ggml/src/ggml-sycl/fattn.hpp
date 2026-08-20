//
// MIT license
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef GGML_SYCL_FATTN_HPP
#define GGML_SYCL_FATTN_HPP

#include "common.hpp"

void ggml_sycl_flash_attn_ext(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

bool ggml_sycl_flash_attn_ext_supported(int device, const ggml_tensor * dst);

// Scratch that flash attention needs beyond the output tensor: F16 staging for K/V, and
// for the oneDNN SDPA path also dense Q, the scale scalar and the F16 output.
//
// Historically these came from the scratch pool (and, for K/V, from a grow-only side
// buffer), which made them invisible to llama.cpp's memory breakdown and therefore to
// --fit: the requirement scales with n_kv, so a context the fitter accepted could still
// OOM part-way through a long prefill. Reserving the space as padding on the FA output
// tensor instead means the graph allocator accounts for it, and because the worst-case
// graph is reserved with the full KV cache the reservation is sized for the deepest
// context up front. This mirrors the CUDA backend (upstream #23907).
//
// ggml_sycl_fattn_get_extra() is the single source of truth for the layout: it is used
// both to size the reservation and to hand out the pointers, so the two cannot disagree.
// Fields are 0 when that buffer is not needed for the selected kernel.
struct ggml_sycl_fattn_extra {
    uintptr_t K     = 0;
    uintptr_t V     = 0;
    uintptr_t Q     = 0;   // oneDNN only
    uintptr_t scale = 0;   // oneDNN only
    uintptr_t out   = 0;   // oneDNN only
    uintptr_t end   = 0;
};

ggml_sycl_fattn_extra ggml_sycl_fattn_get_extra(int device, const ggml_tensor * dst);

// Total bytes to allocate for `dst`: its own size plus the scratch above.
size_t ggml_sycl_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst);

void ggml_sycl_flash_attn_ext_mkl(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif // GGML_SYCL_FATTN_HPP
