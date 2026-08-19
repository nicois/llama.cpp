//
// Device-memory attribution for the SYCL backend.
//
// llama.cpp's memory breakdown (model / context / compute) only sees memory that
// passes through a ggml_backend_buffer. On SYCL a significant amount is allocated
// elsewhere -- the scratch pools, the flash-attention F16 staging buffers, and the
// async (driver-pool-backed) allocations used by the weight reorder path -- and
// shows up only in the breakdown's `unaccounted` column. That makes `--fit`
// over-commit; see docs/development/sycl-fit-unaccounted-vram.md.
//
// This tracks every device allocation by site so `unaccounted` can be attributed.
// Disabled (and near-free) unless GGML_SYCL_MEMTRACE=1.
//
// Env:
//   GGML_SYCL_MEMTRACE=1        enable tracking and logging
//   GGML_SYCL_MEMTRACE_STEP=N   log whenever the peak grows by N MiB (default 64)
//

#ifndef GGML_SYCL_MEMTRACE_HPP
#define GGML_SYCL_MEMTRACE_HPP

#include <cstddef>

enum ggml_sycl_mem_site {
    // Allocations that ggml accounts for: these should sum to roughly
    // model + context + compute in the memory breakdown.
    GGML_SYCL_MEM_BUFFER = 0,

    // Allocations ggml does NOT account for -- i.e. `unaccounted`.
    GGML_SYCL_MEM_POOL_LEG,   // ggml_sycl_pool_leg, retained until context teardown
    GGML_SYCL_MEM_POOL_VMM,   // ggml_sycl_pool_vmm, physical pages mapped into the pool
    GGML_SYCL_MEM_ASYNC,      // syclex::async_malloc; async_free returns to a DRIVER
                              // pool, not the OS, so the peak stays resident
    GGML_SYCL_MEM_FATTN_KV,   // ggml_sycl_fattn_kv_buffers, grow-only F16 K/V staging
    GGML_SYCL_MEM_DIRECT,     // anything else

    GGML_SYCL_MEM_SITE_COUNT,
};

// True when GGML_SYCL_MEMTRACE=1. Checked once, cached.
bool ggml_sycl_memtrace_enabled();

// No-ops unless enabled. `ptr` is the key used by _del, so pass what the
// matching free receives.
void ggml_sycl_memtrace_add(ggml_sycl_mem_site site, const void * ptr, size_t bytes);
void ggml_sycl_memtrace_del(const void * ptr);

// Log live and peak bytes per site. `tag` identifies the call site.
void ggml_sycl_memtrace_report(const char * tag);

#endif  // GGML_SYCL_MEMTRACE_HPP
