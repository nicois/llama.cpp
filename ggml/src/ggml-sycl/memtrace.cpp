#include "memtrace.hpp"

#include "ggml-impl.h"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

namespace {

constexpr size_t MIB = 1024 * 1024;

const char * site_name(ggml_sycl_mem_site site) {
    switch (site) {
        case GGML_SYCL_MEM_BUFFER:   return "buffer";
        case GGML_SYCL_MEM_POOL_LEG: return "pool_leg";
        case GGML_SYCL_MEM_POOL_VMM: return "pool_vmm";
        case GGML_SYCL_MEM_ASYNC:    return "async";
        case GGML_SYCL_MEM_FATTN_KV: return "fattn_kv";
        case GGML_SYCL_MEM_DIRECT:   return "direct";
        default:                     return "?";
    }
}

struct tracker {
    std::mutex mutex;
    std::unordered_map<const void *, std::pair<ggml_sycl_mem_site, size_t>> live_by_ptr;
    size_t live[GGML_SYCL_MEM_SITE_COUNT] = {};
    size_t peak[GGML_SYCL_MEM_SITE_COUNT] = {};
    size_t total_live = 0;
    size_t total_peak = 0;
    size_t last_logged_peak = 0;
    size_t step_bytes = 64 * MIB;
};

tracker & get() {
    static tracker t;
    return t;
}

// Caller must hold the mutex.
void report_locked(const char * tag) {
    tracker & t = get();

    size_t unaccounted = 0;
    for (int i = 0; i < GGML_SYCL_MEM_SITE_COUNT; i++) {
        if (i != GGML_SYCL_MEM_BUFFER) {
            unaccounted += t.live[i];
        }
    }

    GGML_LOG_INFO("[SYCL-MEMTRACE] %s: live %zu MiB (peak %zu MiB) = "
                  "accounted(buffer) %zu + unaccounted %zu MiB\n",
                  tag, t.total_live / MIB, t.total_peak / MIB,
                  t.live[GGML_SYCL_MEM_BUFFER] / MIB, unaccounted / MIB);

    for (int i = 0; i < GGML_SYCL_MEM_SITE_COUNT; i++) {
        if (t.peak[i] == 0) {
            continue;   // never used; don't clutter the log
        }
        GGML_LOG_INFO("[SYCL-MEMTRACE]   %-9s live %7zu MiB  peak %7zu MiB\n",
                      site_name((ggml_sycl_mem_site) i), t.live[i] / MIB,
                      t.peak[i] / MIB);
    }
}

}  // namespace

bool ggml_sycl_memtrace_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_SYCL_MEMTRACE");
        const bool on = env != nullptr && std::atoi(env) != 0;
        if (on) {
            const char * step = std::getenv("GGML_SYCL_MEMTRACE_STEP");
            const int step_mib = step ? std::atoi(step) : 64;
            get().step_bytes = (step_mib > 0 ? (size_t) step_mib : 64) * MIB;
            GGML_LOG_INFO("[SYCL-MEMTRACE] enabled, logging every %d MiB of peak growth\n",
                          step_mib > 0 ? step_mib : 64);
        }
        return on;
    }();
    return enabled;
}

void ggml_sycl_memtrace_add(ggml_sycl_mem_site site, const void * ptr, size_t bytes) {
    if (!ggml_sycl_memtrace_enabled() || ptr == nullptr || bytes == 0) {
        return;
    }
    tracker & t = get();
    std::lock_guard<std::mutex> lock(t.mutex);

    // A repeated pointer means we missed a free; correct rather than double count.
    auto it = t.live_by_ptr.find(ptr);
    if (it != t.live_by_ptr.end()) {
        t.live[it->second.first] -= it->second.second;
        t.total_live -= it->second.second;
    }

    t.live_by_ptr[ptr] = { site, bytes };
    t.live[site] += bytes;
    t.total_live += bytes;

    if (t.live[site] > t.peak[site]) {
        t.peak[site] = t.live[site];
    }
    if (t.total_live > t.total_peak) {
        t.total_peak = t.total_live;
    }

    if (t.total_peak >= t.last_logged_peak + t.step_bytes) {
        t.last_logged_peak = t.total_peak;
        char tag[96];
        std::snprintf(tag, sizeof(tag), "peak grew (+%zu MiB from %s)", bytes / MIB,
                 site_name(site));
        report_locked(tag);
    }
}

void ggml_sycl_memtrace_del(const void * ptr) {
    if (!ggml_sycl_memtrace_enabled() || ptr == nullptr) {
        return;
    }
    tracker & t = get();
    std::lock_guard<std::mutex> lock(t.mutex);

    auto it = t.live_by_ptr.find(ptr);
    if (it == t.live_by_ptr.end()) {
        return;
    }
    t.live[it->second.first] -= it->second.second;
    t.total_live -= it->second.second;
    t.live_by_ptr.erase(it);
}

void ggml_sycl_memtrace_report(const char * tag) {
    if (!ggml_sycl_memtrace_enabled()) {
        return;
    }
    tracker & t = get();
    std::lock_guard<std::mutex> lock(t.mutex);
    report_locked(tag);
}
