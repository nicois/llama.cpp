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
