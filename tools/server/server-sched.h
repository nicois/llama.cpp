#pragma once

#include "server-task.h"

#include <functional>
#include <list>
#include <vector>

using sched_states   = std::list<server_prompt_cache_state>;
using sched_state_it = sched_states::const_iterator;

struct sched_candidate_score {
    size_t lcp    = 0;
    float  f_keep = 0.0f;
    float  f_sim  = 0.0f;
};

struct sched_restore_trace {
    float base_f_keep = 0.0f;                      // pre-loop baseline
    float base_f_sim  = 0.0f;
    std::vector<sched_candidate_score> candidates;  // in states order, one per entry
    bool  found       = false;                      // post-selection
    float best_f_keep = 0.0f;
    float best_f_sim  = 0.0f;
};

// Current upstream selection, extracted verbatim. Retained only so the characterization
// test can demonstrate the #27148 boilerplate-only match. Do not call from new code.
sched_state_it sched_pick_restore_baseline(const sched_states  & states,
                                          const server_tokens & tokens_new,
                                          const server_tokens & slot_prompt,
                                          sched_restore_trace * trace = nullptr);

// How many already-resident prompt tokens this task could reuse if launched on the slot
// whose current prompt is `slot_prompt`. Absolute token count; higher is better.
size_t sched_score(const server_tokens & task_tokens,
                   const server_tokens & slot_prompt,
                   const sched_states  & states);

// Index of the deferred task to run next, given one score per task in queue order:
// highest score wins, earliest wins on ties (FIFO). Returns SIZE_MAX for an empty list.
size_t sched_pick_task(const std::vector<size_t> & scores);
