// Hold a fixed amount of device memory, so another process on the same GPU sees a
// smaller card than it really has.  Written to test llama.cpp's --fit behaviour on
// low-VRAM SYCL devices without owning one.
//
// Build (inside the SYCL container, or any oneAPI environment):
//   icpx -fsycl -O2 -o /tmp/vram-hog scripts/sycl-vram-hog.cpp
//
// Run, leaving ~12 GiB free on a 32656 MiB card:
//   ZES_ENABLE_SYSMAN=1 /tmp/vram-hog 20368
//
// Ctrl-C (or SIGTERM) releases everything.
//
// Why hold real memory rather than passing a large --fit-target: a large margin makes
// --fit *decide* as though the card were smaller, but the memory is still physically
// there at runtime, so a build that over-commits quietly survives on the slack.  To
// reproduce what a small card actually does, the memory has to be genuinely gone.
//
// Note the simulation is only partial: the device still reports its real total, so any
// logic keyed on total rather than free is not exercised, and absolute throughput will
// not match a real small card with fewer cores and less bandwidth.  What it does give
// you is a valid comparison between two llama.cpp builds at the same memory budget.

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

namespace {

constexpr size_t MIB = 1024ull * 1024ull;

volatile sig_atomic_t g_stop = 0;

void on_signal(int) {
    g_stop = 1;
}

// Real free memory needs ZES_ENABLE_SYSMAN=1; without it the runtime reports free ==
// total, which would make the whole exercise silently meaningless.
bool free_memory(const sycl::device & dev, size_t & free_bytes) {
    if (!dev.has(sycl::aspect::ext_intel_free_memory)) {
        return false;
    }
    free_bytes = dev.get_info<sycl::ext::intel::info::device::free_memory>();
    return true;
}

void report(const sycl::device & dev, const char * tag) {
    const size_t total = dev.get_info<sycl::info::device::global_mem_size>();
    size_t       free  = 0;
    if (free_memory(dev, free)) {
        printf("%-18s total %6zu MiB   free %6zu MiB\n", tag, total / MIB, free / MIB);
    } else {
        printf("%-18s total %6zu MiB   free unavailable (set ZES_ENABLE_SYSMAN=1)\n",
               tag, total / MIB);
    }
    fflush(stdout);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr,
                "usage: %s <MiB to hold> [device index]\n"
                "  e.g. %s 20368        # leaves ~12 GiB free on a 32656 MiB card\n",
                argv[0], argv[0]);
        return 2;
    }

    const size_t want  = strtoull(argv[1], nullptr, 10) * MIB;
    const int    index = argc > 2 ? atoi(argv[2]) : 0;

    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devices.empty()) {
        fprintf(stderr, "no GPU devices found\n");
        return 1;
    }
    if (index < 0 || index >= (int) devices.size()) {
        fprintf(stderr, "device index %d out of range (%zu GPUs)\n", index, devices.size());
        return 1;
    }

    sycl::queue q(devices[index]);
    printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    report(q.get_device(), "before:");

    // Chunked: one allocation of many GiB can exceed the driver's single-allocation
    // limit, and chunking also lets us stop cleanly at whatever the device will give.
    const size_t       chunk = 256 * MIB;
    std::vector<void *> blocks;
    size_t             held = 0;

    while (held < want) {
        const size_t n = std::min(chunk, want - held);
        void *       p = sycl::malloc_device(n, q);
        if (p == nullptr) {
            printf("allocation stopped short at %zu MiB\n", held / MIB);
            break;
        }
        // Touch it so there is no doubt the pages are committed and visible to the
        // device-wide free-memory figure another process will read.
        q.memset(p, 0, n).wait();
        blocks.push_back(p);
        held += n;
    }

    printf("holding %zu MiB in %zu blocks\n", held / MIB, blocks.size());
    report(q.get_device(), "after:");
    printf("Ctrl-C to release.\n");
    fflush(stdout);

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    while (!g_stop) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    printf("\nreleasing %zu MiB\n", held / MIB);
    for (void * p : blocks) {
        sycl::free(p, q);
    }
    report(q.get_device(), "released:");
    return 0;
}
