#include "testing.h"

#include "server-sched.h"

#include <cstdint>
#include <iterator>
#include <list>
#include <vector>

// Build a cache state holding the given tokens. Sizes are irrelevant to selection.
static server_prompt_cache_state make_state(const std::vector<llama_token> & toks) {
    server_prompt_cache_state st;
    st.prompt.tokens = server_tokens(toks, false);
    return st;
}

// A run of n tokens starting at `base`, so prefixes are easy to construct.
static std::vector<llama_token> run(llama_token base, size_t n) {
    std::vector<llama_token> v;
    v.reserve(n);
    for (size_t i = 0; i < n; i++) {
        v.push_back(base + (llama_token) i);
    }
    return v;
}

// Concatenate two token runs.
static std::vector<llama_token> cat(const std::vector<llama_token> & a,
                                   const std::vector<llama_token> & b) {
    std::vector<llama_token> v = a;
    v.insert(v.end(), b.begin(), b.end());
    return v;
}

// Tests 1-3: sched_score
static void test_score(testing & t) {
    t.test("returns_max_over_candidates_in_absolute_tokens", [&](testing & t) {
        const auto shared = run(1, 5000);

        sched_states states;
        states.push_back(make_state(cat(shared, run(60000, 10))));            // lcp 5000
        states.push_back(make_state(cat(run(1, 9000), run(70000, 10))));      // lcp 9000

        server_tokens task(cat(run(1, 12000), run(80000, 100)), false);
        server_tokens slot_prompt(cat(shared, run(90000, 10)), false);        // lcp 5000

        t.assert_equal("picks the deepest candidate", (size_t) 9000,
                       sched_score(task, slot_prompt, states));
    });

    t.test("returns_zero_when_nothing_is_shared", [&](testing & t) {
        sched_states states;
        states.push_back(make_state(run(500000, 1000)));

        server_tokens task(run(1, 1000), false);
        server_tokens slot_prompt(run(700000, 1000), false);

        t.assert_equal("no shared prefix", (size_t) 0,
                       sched_score(task, slot_prompt, states));
    });

    t.test("empty_slot_and_empty_states_score_zero", [&](testing & t) {
        sched_states states;
        server_tokens task(run(1, 1000), false);
        server_tokens slot_prompt;

        t.assert_equal("nothing resident", (size_t) 0,
                       sched_score(task, slot_prompt, states));
    });
}

// Test 4: sched_pick_task
static void test_pick_task(testing & t) {
    t.test("picks_highest_score_with_fifo_tie_break", [&](testing & t) {
        t.assert_equal("highest score wins", (size_t) 2,
                       sched_pick_task(std::vector<size_t>{10, 30, 90, 40}));
        t.assert_equal("earliest wins among equals", (size_t) 1,
                       sched_pick_task(std::vector<size_t>{10, 90, 90, 90}));
        t.assert_equal("all zero falls back to first", (size_t) 0,
                       sched_pick_task(std::vector<size_t>{0, 0, 0}));
        t.assert_equal("empty yields SIZE_MAX", SIZE_MAX,
                       sched_pick_task(std::vector<size_t>{}));
    });
}

// Test 5 (characterization): baseline selection accepts a boilerplate-only match.
// This encodes the #27148 defect deliberately. When #27148 is fixed upstream this test
// SHOULD fail and should then be deleted.
static void test_baseline_characterization(testing & t) {
    t.test("baseline_accepts_boilerplate_only_match", [&](testing & t) {
        const auto boiler = run(1000, 20000);          // shared system prompt + tool schemas
        const auto stale  = run(90000, 5000);          // unrelated conversation content
        const auto fresh  = run(50000, 30000);         // this request's real content

        sched_states states;
        states.push_back(make_state(cat(boiler, stale)));

        server_tokens tokens_new(cat(boiler, fresh), false);
        server_tokens slot_prompt;                     // fresh slot: empty

        const auto it = sched_pick_restore_baseline(states, tokens_new, slot_prompt);

        t.assert_true("baseline selects the unrelated entry on a boilerplate-only match",
                      it != states.end());
    });
}

int main(int argc, char ** argv) {
    testing t;

    const char * verbose = getenv("LLAMA_TEST_VERBOSE");
    if (verbose) {
        t.verbose = std::string(verbose) == "1";
    }

    if (argc > 1) {
        t.set_filter(argv[1]);
    }

    t.test("score", test_score);
    t.test("pick_task", test_pick_task);
    t.test("baseline", test_baseline_characterization);

    return t.summary();
}
