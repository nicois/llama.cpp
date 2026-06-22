#pragma once

#include <cstdint>
#include <vector>

// Build the list of context-size tiers: powers of two from `min_tier` up to
// `max_ctx`, with `max_ctx` itself appended as the top tier when it is not a
// power of two. If `max_ctx <= min_tier`, a single tier equal to `max_ctx` is
// returned (the feature is effectively a no-op).
static inline std::vector<int32_t> server_ctx_build_tiers(int32_t min_tier, int32_t max_ctx) {
    std::vector<int32_t> tiers;
    if (max_ctx <= min_tier) {
        tiers.push_back(max_ctx);
        return tiers;
    }
    for (int32_t t = min_tier; t < max_ctx; t *= 2) {
        tiers.push_back(t);
    }
    if (tiers.empty() || tiers.back() != max_ctx) {
        tiers.push_back(max_ctx);
    }
    return tiers;
}

// Return the smallest tier >= n_tokens_needed. If none fits, return the largest
// tier (the caller is responsible for treating an over-cap request as an error).
static inline int32_t server_ctx_required_tier(const std::vector<int32_t> & tiers, int32_t n_tokens_needed) {
    for (int32_t t : tiers) {
        if (t >= n_tokens_needed) {
            return t;
        }
    }
    return tiers.empty() ? n_tokens_needed : tiers.back();
}

// Decide a shrink target with hysteresis. Returns `current_tier` unchanged unless
// the requirement fits in a strictly smaller tier AND sits comfortably below that
// smaller tier's capacity (by `margin_pct` percent), to avoid thrashing near a
// boundary. Never grows (returns `current_tier` if the requirement needs >= it).
static inline int32_t server_ctx_shrink_target(const std::vector<int32_t> & tiers,
                                               int32_t current_tier,
                                               int32_t n_tokens_needed,
                                               int     margin_pct) {
    const int32_t required = server_ctx_required_tier(tiers, n_tokens_needed);
    if (required >= current_tier) {
        return current_tier; // never grow here; growth is handled elsewhere
    }
    // required < current_tier: only shrink if comfortably under the smaller tier
    const int64_t threshold = (int64_t) required * (100 - margin_pct) / 100;
    if (n_tokens_needed <= threshold) {
        return required;
    }
    return current_tier;
}
