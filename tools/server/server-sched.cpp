#include "server-sched.h"

#include <algorithm>
#include <cstdint>

sched_state_it sched_pick_restore_baseline(const sched_states  & states,
                                           const server_tokens & tokens_new,
                                           const server_tokens & slot_prompt,
                                           sched_restore_trace * trace) {
    const int lcp_best = slot_prompt.get_common_prefix(tokens_new);

    float f_keep_best = slot_prompt.size() > 0 ? float(lcp_best) / slot_prompt.size() : -1.0f;
    float f_sim_best  = float(lcp_best) / tokens_new.size();

    if (trace) {
        trace->base_f_keep = f_keep_best;
        trace->base_f_sim  = f_sim_best;
        trace->candidates.clear();
    }

    auto it_best = states.end();

    for (auto it = states.begin(); it != states.end(); ++it) {
        const int lcp_cur = it->prompt.tokens.get_common_prefix(tokens_new);

        const float f_keep_cur = float(lcp_cur) / it->prompt.tokens.size();
        const float f_sim_cur  = float(lcp_cur) / tokens_new.size();

        if (trace) {
            trace->candidates.push_back({static_cast<size_t>(lcp_cur), f_keep_cur, f_sim_cur});
        }

        // don't trash large prompts
        if (f_keep_cur < 0.25f) {
            continue;
        }

        if (f_keep_best < f_keep_cur && f_sim_best < f_sim_cur) {
            f_keep_best = f_keep_cur;
            f_sim_best  = f_sim_cur;

            it_best = it;
        }
    }

    if (trace) {
        trace->found       = (it_best != states.end());
        trace->best_f_keep = f_keep_best;
        trace->best_f_sim  = f_sim_best;
    }

    return it_best;
}

size_t sched_score(const server_tokens & task_tokens,
                   const server_tokens & slot_prompt,
                   const sched_states  & states) {
    size_t best = slot_prompt.get_common_prefix(task_tokens);

    for (const auto & st : states) {
        best = std::max(best, (size_t) st.prompt.tokens.get_common_prefix(task_tokens));
    }

    return best;
}

size_t sched_pick_task(const std::vector<size_t> & scores) {
    if (scores.empty()) {
        return SIZE_MAX;
    }

    size_t best = 0;
    for (size_t i = 1; i < scores.size(); i++) {
        // strict > keeps the earliest task among equal scores, i.e. FIFO on ties
        if (scores[i] > scores[best]) {
            best = i;
        }
    }

    return best;
}
