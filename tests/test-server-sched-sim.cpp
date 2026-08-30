#include "testing.h"
#include "server-sched.h"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <list>
#include <vector>

// Whether turn k+1's prompt strictly extends turn k's is decided by the chat template and
// the client, not by the server. It is therefore a first-class dimension of the workload:
// it determines whether prefix reuse exists at all. Model it generically as a *rewrite
// policy* -- which previously-rendered positions get re-rendered, and when.
//
// Qwen3 is one instance, not the model: its templates gate reasoning on
// `preserve_thinking or loop.index0 > last_query_index`
// (froggeric/Qwen-Fixed-Chat-Templates chat_template.jinja:298), where last_query_index is
// the last *genuine* user message -- user messages rendered as <tool_response> are skipped
// (lines 196-205). That yields APPEND_ONLY when preserve_thinking is true (the default) and
// DROP_REASONING_PRIOR_TURNS when it is false.
enum sim_history_policy {
    // Nothing is ever re-rendered. Best case for prefix caching. Non-reasoning templates,
    // and Qwen with preserve_thinking=true (what the measured 24h window ran).
    SIM_HIST_APPEND_ONLY,
    // Reasoning is kept for assistant messages in the current human turn and dropped for
    // earlier turns. Tool calls within a turn still extend exactly; each new human prompt
    // diverges at the previous turn's first assistant message. Qwen preserve_thinking=false.
    SIM_HIST_DROP_REASONING_PRIOR_TURNS,
    // Reasoning is stripped from every prior assistant message, including within the tool
    // loop. Diverges at the previous assistant message on every single request. Hostile.
    SIM_HIST_DROP_REASONING_ALWAYS,
    // Sliding window: once max_ctx_tokens is reached, the front is trimmed, so every
    // position shifts and the prefix match collapses to zero. Maximally hostile -- included
    // to prove the feature does no harm where it cannot help.
    SIM_HIST_TRUNCATE_FRONT,
    // History is replaced by a summary once compact_at_tokens is exceeded, then grows again.
    // Diverges at the summary insertion point. This is what omp's own compaction does.
    SIM_HIST_COMPACT_AT_THRESHOLD,
};

struct sim_params {
    size_t n_sessions       = 6;
    size_t n_turns          = 20;     // total requests per session
    size_t n_slots          = 1;
    size_t preamble_tokens  = 20000;  // NOT measured: estimated system prompt + tool schemas
    // Heterogeneous session sizes exercise the policy more thoroughly than a uniform workload.
    // Per-session final contexts: 40k, 60k, 88k, 120k, 200k, 230k
    // → first_turn values: 8k, 28k, 56k, 88k, 168k, 198k (final - preamble - n_turns*growth)
    // Total working set: 738k tokens × 60 KiB ≈ 43.2 GiB vs 32 GiB cache
    static constexpr size_t session_first_turn[6] = {
        8000,    // 40k final
        28000,   // 60k final
        56000,   // 88k final
        88000,   // 120k final
        168000,  // 200k final
        198000   // 230k final
    };
    size_t growth_tokens    = 600;    // context growth per turn
    size_t gen_tokens       = 527;    // generated tokens per turn (median, measured)
    double think_s          = 0.0;    // agentic loop resubmits immediately (default)
    double prefill_tok_s    = 900.0;  // measured ~765-1275 t/s
    double gen_tok_s        = 15.7;   // measured aggregate
    size_t kv_bytes_per_tok = 60u * 1024u;
    size_t cache_bytes      = 32ull * 1024 * 1024 * 1024;

    sim_history_policy history = SIM_HIST_APPEND_ONLY;
    size_t max_ctx_tokens      = 262144;  // for TRUNCATE_FRONT
    size_t compact_at_tokens   = 223000;  // for COMPACT_AT_THRESHOLD (omp's observed trigger)
    // Tool calls per human prompt. NOT measured -- the server log cannot distinguish a tool
    // response from a human prompt. Sweep it; do not present any value as observed.
    size_t tools_per_prompt = 8;
    // Share of generated tokens that is reasoning. Also NOT measured. Sweep it.
    double reasoning_frac   = 0.7;
};

struct sim_result {
    double makespan_s      = 0.0;
    size_t prefill_tokens  = 0;
    size_t gen_tokens      = 0;
    size_t n_requests      = 0;
    size_t n_high_reuse    = 0;   // >=95% of prompt reused
    size_t n_lost          = 0;   // >=30k prompt, <5% reused
    double wait_p99_cold_s = 0.0;
};

// Forward declaration
sim_result sim_run(const sim_params & p, bool cache_aware);

namespace {

struct sim_session {
    size_t                   id         = 0;
    size_t                   turn       = 0;
    size_t                   first_turn = 0;       // per-session initial context size
    std::vector<llama_token> body;                 // divergent tail, session-unique
    double                   ready_at   = 0.0;     // earliest send time
};

struct sim_slot {
    std::vector<llama_token> prompt_tokens;  // store tokens as vector
    double        free_at = 0.0;
};

// Rendered prompt geometry for a turn: total length, and the position at which this render
// diverges from the previous one. Everything before diverge_at is byte-identical to what the
// previous render produced, so it is exactly what a prefix cache can reuse.
struct sim_render {
    size_t len        = 0;
    size_t diverge_at = 0;
};

sim_render sim_render_at(const sim_params & p, size_t first_turn, size_t turn) {
    const size_t tpp        = std::max<size_t>(1, p.tools_per_prompt);
    const size_t human_turn = turn / tpp;
    const double keep       = 1.0 - p.reasoning_frac;

    // nominal appended length per request, before any rewriting
    auto nominal_len = [&](size_t k) {
        return p.preamble_tokens + first_turn + k * p.growth_tokens;
    };

    sim_render r;
    switch (p.history) {
        case SIM_HIST_APPEND_ONLY:
            r.len        = nominal_len(turn);
            r.diverge_at = turn == 0 ? 0 : nominal_len(turn - 1);
            break;

        case SIM_HIST_DROP_REASONING_PRIOR_TURNS: {
            // prior human turns contribute only their non-reasoning share
            const size_t prior = human_turn * tpp;
            const size_t in_turn = turn - prior;
            r.len = p.preamble_tokens + first_turn
                  + (size_t) (double(prior) * p.growth_tokens * keep)
                  + in_turn * p.growth_tokens;
            if (turn == 0) {
                r.diverge_at = 0;
            } else if (in_turn > 0) {
                // still inside the same human turn: pure append
                r.diverge_at = r.len - p.growth_tokens;
            } else {
                // new human prompt: the previous turn is re-rendered without reasoning
                const size_t prev_prior = (human_turn - 1) * tpp;
                r.diverge_at = p.preamble_tokens + first_turn
                             + (size_t) (double(prev_prior) * p.growth_tokens * keep);
            }
            break;
        }

        case SIM_HIST_DROP_REASONING_ALWAYS:
            r.len = p.preamble_tokens + first_turn
                  + (size_t) (double(turn) * p.growth_tokens * keep);
            // the immediately preceding assistant message is re-rendered every request
            r.diverge_at = turn == 0 ? 0
                : p.preamble_tokens + first_turn
                  + (size_t) (double(turn - 1) * p.growth_tokens * keep);
            break;

        case SIM_HIST_TRUNCATE_FRONT: {
            const size_t nominal = nominal_len(turn);
            if (nominal <= p.max_ctx_tokens) {
                r.len        = nominal;
                r.diverge_at = turn == 0 ? 0 : nominal_len(turn - 1);
            } else {
                r.len = p.max_ctx_tokens;
                // every position shifted: nothing before the end matches
                r.diverge_at = 0;
            }
            break;
        }

        case SIM_HIST_COMPACT_AT_THRESHOLD: {
            const size_t nominal = nominal_len(turn);
            if (nominal <= p.compact_at_tokens) {
                r.len        = nominal;
                r.diverge_at = turn == 0 ? 0 : nominal_len(turn - 1);
            } else {
                // history collapsed to a summary just after the preamble, then regrows
                const size_t cycle = (nominal - p.compact_at_tokens) / std::max<size_t>(1, p.growth_tokens);
                r.len        = p.preamble_tokens + 8192 + cycle * p.growth_tokens;
                r.diverge_at = cycle == 0 ? p.preamble_tokens : r.len - p.growth_tokens;
            }
            break;
        }
    }

    r.diverge_at = std::min(r.diverge_at, r.len);
    return r;
}

// Build the token list so that positions below diverge_at reproduce the previous render
// exactly and positions at or above it do not. Tag each position with the turn at which it
// was last rewritten; the token is a function of (session, position, that turn). This makes
// get_common_prefix(prompt(k), prompt(k-1)) == diverge_at by construction, rather than by
// arithmetic that has to be got right twice.
std::vector<llama_token> sim_prompt(const sim_params & p, const sim_session & s, size_t turn) {
    std::vector<size_t> gen(sim_render_at(p, s.first_turn, turn).len, 0);
    for (size_t k = 1; k <= turn; k++) {
        const sim_render r = sim_render_at(p, s.first_turn, k);
        for (size_t i = r.diverge_at; i < std::min(r.len, gen.size()); i++) {
            gen[i] = k;
        }
    }

    std::vector<llama_token> v;
    v.reserve(gen.size());
    for (size_t i = 0; i < gen.size(); i++) {
        if (i < p.preamble_tokens) {
            v.push_back((llama_token) (i + 1));       // preamble: identical for all sessions
        } else {
            v.push_back((llama_token) (1 + (s.id * 1000003 + i * 31 + gen[i] * 7919) % 900000));
        }
    }
    return v;
}

size_t entry_bytes(const sim_params & p, const server_prompt_cache_state & st) {
    return st.prompt.tokens.size() * p.kv_bytes_per_tok;
}

size_t cache_bytes_used(const sim_params & p, const sched_states & states) {
    size_t n = 0;
    for (const auto & st : states) {
        n += entry_bytes(p, st);
    }
    return n;
}

} // namespace

sim_result sim_run(const sim_params & p, bool cache_aware) {
    std::vector<sim_session> sessions(p.n_sessions);
    for (size_t i = 0; i < p.n_sessions; i++) {
        sessions[i].id = i;
        sessions[i].first_turn = p.session_first_turn[i % 6];
        // unique body per session, diverging immediately after the preamble
        sessions[i].body.resize(4096);
        for (size_t j = 0; j < sessions[i].body.size(); j++) {
            sessions[i].body[j] = (llama_token) (1000000 + i * 100000 + j);
        }
    }

    std::vector<sim_slot> slots(p.n_slots);
    sched_states states;

    sim_result   res;
    std::vector<double> cold_waits;
    double now = 0.0;

    size_t remaining = sessions.size() * p.n_turns;

    while (remaining > 0) {
        // ---- collect sessions whose next request has arrived
        // Build pairs of (ready_at, session_id) and sort by ready_at for true FIFO
        std::vector<std::pair<double, size_t>> ready_queue;
        for (auto & s : sessions) {
            if (s.turn < p.n_turns && s.ready_at <= now) {
                ready_queue.push_back({s.ready_at, s.id});
            }
        }
        std::sort(ready_queue.begin(), ready_queue.end());

        std::vector<size_t> queued;
        for (const auto & p : ready_queue) {
            queued.push_back(p.second);
        }

        // pick the earliest free slot
        size_t slot_i = 0;
        for (size_t i = 1; i < slots.size(); i++) {
            if (slots[i].free_at < slots[slot_i].free_at) {
                slot_i = i;
            }
        }
        auto & slot = slots[slot_i];

        if (queued.empty()) {
            // advance to the next arrival
            double next = 1e18;
            for (auto & s : sessions) {
                if (s.turn < p.n_turns) {
                    next = std::min(next, s.ready_at);
                }
            }
            now = std::max(now, next);
            continue;
        }

        now = std::max(now, slot.free_at);

        // ---- ordering decision: the real sched_pick_task over the real sched_score
        std::vector<std::vector<llama_token>> prompt_vecs;
        std::vector<size_t>        scores;
        prompt_vecs.reserve(queued.size());
        for (size_t id : queued) {
            prompt_vecs.push_back(sim_prompt(p, sessions[id], sessions[id].turn));
        }

        // Build server_tokens for scoring, then pick
        server_tokens slot_prompt(slot.prompt_tokens, false);
        if (cache_aware) {
            for (const auto & pv : prompt_vecs) {
                server_tokens pr(pv, false);
                scores.push_back(sched_score(pr, slot_prompt, states));
            }
        } else {
            scores.assign(queued.size(), 0);       // FIFO: all equal, first wins
        }
        const size_t pick = sched_pick_task(scores);
        const size_t sid  = queued[pick];
        auto &       sess = sessions[sid];
        std::vector<llama_token> prompt_vec = std::move(prompt_vecs[pick]);

        const double wait_s = now - sess.ready_at;

        // ---- restore decision: baseline only
        server_tokens prompt(prompt_vec, false);
        server_tokens slot_prompt_restore(slot.prompt_tokens, false);
        size_t resident = slot_prompt_restore.get_common_prefix(prompt);
        auto   chosen   = sched_pick_restore_baseline(states, prompt, slot_prompt_restore);
        if (chosen != states.cend()) {
            resident = std::max(resident,
                                (size_t) chosen->prompt.tokens.get_common_prefix(prompt));
            states.erase(chosen);
        }

        const size_t to_prefill = prompt.size() - std::min(resident, prompt.size());

        // ---- cost model
        const double t_prefill = double(to_prefill) / p.prefill_tok_s;
        const double t_gen     = double(p.gen_tokens) / p.gen_tok_s;

        now         += t_prefill + t_gen;
        slot.free_at = now;

        // ---- save the finished prompt into the cache, evicting via the real picker
        server_prompt_cache_state fresh;
        {
            std::vector<llama_token> full;
            full.reserve(prompt.size());
            for (size_t i = 0; i < prompt.size(); i++) {
                full.push_back(prompt[i]);
            }
            fresh.prompt.tokens = server_tokens(full, false);
        }
        const size_t need = entry_bytes(p, fresh);

        // demand = argmax candidate for some still-queued session
        auto has_demand = [&](const server_prompt_cache_state & st) {
            for (size_t id : queued) {
                if (id == sid || sessions[id].turn >= p.n_turns) {
                    continue;
                }
                std::vector<llama_token> q_vec = sim_prompt(p, sessions[id], sessions[id].turn);
                server_tokens q(q_vec, false);
                const server_prompt_cache_state * best = nullptr;
                size_t best_lcp = 0;
                for (const auto & c : states) {
                    const size_t lcp = c.prompt.tokens.get_common_prefix(q);
                    if (lcp > best_lcp) {
                        best_lcp = lcp;
                        best     = &c;
                    }
                }
                if (best == &st) {
                    return true;
                }
            }
            return false;
        };

        while (!states.empty() && cache_bytes_used(p, states) + need > p.cache_bytes) {
            states.erase(states.cbegin());
        }
        if (cache_bytes_used(p, states) + need <= p.cache_bytes) {
            states.push_back(std::move(fresh));
        }

        // ---- bookkeeping
        slot.prompt_tokens = std::move(prompt_vec);
        res.prefill_tokens += to_prefill;
        res.gen_tokens     += p.gen_tokens;
        res.n_requests++;
        const double reuse = prompt.size() ? double(resident) / prompt.size() : 1.0;
        if (reuse >= 0.95) {
            res.n_high_reuse++;
        } else {
            cold_waits.push_back(wait_s);
            if (prompt.size() >= 30000 && reuse < 0.05) {
                res.n_lost++;
            }
        }

        sess.turn++;
        sess.ready_at = now + p.think_s;
        remaining--;
        res.makespan_s = now;
    }

    std::sort(cold_waits.begin(), cold_waits.end());
    if (!cold_waits.empty()) {
        res.wait_p99_cold_s = cold_waits[(size_t) (cold_waits.size() * 0.99) % cold_waits.size()];
    }
    return res;
}

// Test: with the cache too small for the working set, cache-aware scheduling must
// re-prefill strictly fewer tokens and finish sooner than baseline.
// The rendering model must be self-consistent: the measured common prefix between
// consecutive renders must equal the divergence position the policy declares.
static void test_sim_render_is_self_consistent(testing & t) {
    for (auto pol : {SIM_HIST_APPEND_ONLY, SIM_HIST_DROP_REASONING_PRIOR_TURNS,
                     SIM_HIST_DROP_REASONING_ALWAYS, SIM_HIST_TRUNCATE_FRONT,
                     SIM_HIST_COMPACT_AT_THRESHOLD}) {
        t.test("policy_" + std::to_string((int) pol), [&](testing & t) {
            sim_params p;
            p.history = pol;
            // Test all 6 session sizes to verify render consistency per session
            for (size_t sess_id = 0; sess_id < 6; sess_id++) {
                sim_session s;
                s.id = sess_id;
                s.first_turn = p.session_first_turn[sess_id];
                for (size_t k = 1; k < 12; k++) {
                    std::vector<llama_token> a_vec = sim_prompt(p, s, k - 1);
                    std::vector<llama_token> b_vec = sim_prompt(p, s, k);
                    const server_tokens a(a_vec, false);
                    const server_tokens b(b_vec, false);
                    const size_t expect = std::min(sim_render_at(p, s.first_turn, k).diverge_at,
                                                   (size_t) a.size());
                    t.assert_equal("common prefix equals declared divergence",
                                   expect, (size_t) a.get_common_prefix(b));
                }
            }
        });
    }
}

// The feature must help wherever prefix reuse exists, and must not hurt where it does not.
// A reviewer running a reasoning-stripping or truncating client must not see a regression.
static void test_sim_across_history_policies(testing & t) {
    struct expectation { sim_history_policy pol; const char * name; bool expect_gain; };

    const expectation cases[] = {
        {SIM_HIST_APPEND_ONLY,                "append_only",           true},
        {SIM_HIST_DROP_REASONING_PRIOR_TURNS, "drop_reasoning_prior",  true},
        {SIM_HIST_DROP_REASONING_ALWAYS,      "drop_reasoning_always", true},
        // nothing is reusable once the window slides, so the only requirement is no harm
        {SIM_HIST_TRUNCATE_FRONT,             "truncate_front",        false},
        {SIM_HIST_COMPACT_AT_THRESHOLD,       "compact_at_threshold",  true},
    };

    for (const auto & c : cases) {
        t.test(c.name, [&](testing & t) {
            sim_params p;
            p.history     = c.pol;
            // Tighter pressure: 16 GiB cache vs ~65 GiB working set → ratio ~4.1
            // (cache holds ~1.5 of 6 sessions; baseline thrashes harder than production)
            p.cache_bytes = 16ull * 1024 * 1024 * 1024;

            const sim_result b = sim_run(p, false);
            const sim_result s = sim_run(p, true);

            if (c.expect_gain) {
                t.assert_true("fewer tokens re-prefilled", s.prefill_tokens < b.prefill_tokens);
                t.assert_true("shorter makespan", s.makespan_s < b.makespan_s);
            } else {
                // do no harm: allow 1% for tie-breaking differences, forbid real regression
                t.assert_true("no prefill regression",
                              s.prefill_tokens <= (size_t) (b.prefill_tokens * 1.01));
                t.assert_true("no makespan regression", s.makespan_s <= b.makespan_s * 1.01);
            }
        });
    }
}

static void test_sim_beats_baseline_under_pressure(testing & t) {
    t.test("fewer_reprefilled_tokens_and_shorter_makespan", [&](testing & t) {
        sim_params p;
        // ADVERSARIAL scenario: all 6 sessions active simultaneously with think_s=0.
        // This is more hostile than production (which had 2-3 concurrent sessions) and
        // serves as a stress test proving the feature helps under maximum contention.
        // Production config: 32 GiB cache, ~43 GiB working set (6 sessions) → ratio 1.34
        p.cache_bytes = 32ull * 1024 * 1024 * 1024;

        const sim_result base = sim_run(p, false);
        const sim_result sched = sim_run(p, true);

        t.assert_true("same work offered", base.n_requests == sched.n_requests);
        t.assert_true("fewer tokens re-prefilled",
                      sched.prefill_tokens < base.prefill_tokens);
        t.assert_true("shorter makespan", sched.makespan_s < base.makespan_s);
        t.assert_true("more high-reuse requests",
                      sched.n_high_reuse > base.n_high_reuse);
    });
}

static void print_metrics() {
    sim_params p;

    fprintf(stderr, "\n=== Workload Scale (Heterogeneous Sessions) ===\n");
    double total_gib = 0.0;
    size_t total_tokens = 0;
    for (size_t i = 0; i < 6; i++) {
        size_t final_ctx = p.preamble_tokens + p.session_first_turn[i] + (p.n_turns * p.growth_tokens);
        size_t entry_bytes = final_ctx * p.kv_bytes_per_tok;
        double entry_gib = double(entry_bytes) / (1024.0 * 1024.0 * 1024.0);
        total_gib += entry_gib;
        total_tokens += final_ctx;
        fprintf(stderr, "Session %zu: %zu tokens × 60 KiB = %.2f GiB\n", i, final_ctx, entry_gib);
    }
    fprintf(stderr, "Total working set: %.2f GiB\n", total_gib);
    fprintf(stderr, "Distribution: median %zu tokens, p90 %zu tokens, max %zu tokens\n",
            p.preamble_tokens + p.session_first_turn[2] + (p.n_turns * p.growth_tokens),
            p.preamble_tokens + p.session_first_turn[4] + (p.n_turns * p.growth_tokens),
            p.preamble_tokens + p.session_first_turn[5] + (p.n_turns * p.growth_tokens));
    fprintf(stderr, "Pressure @ 32 GiB: ratio %.2f (cache holds ~%.1f of 6 sessions)\n\n",
            total_gib/32.0, 32.0 / (total_gib / 6.0));

    fprintf(stderr, "=== 32 GiB (production config) ===\n");
    sim_result b32 = sim_run(p, false);
    sim_result t32 = sim_run(p, true);
    fprintf(stderr, "Baseline:    %zu prefill, %.0fs, %zu/%zu high-reuse (%.0f%%), %zu lost\n",
            b32.prefill_tokens, b32.makespan_s, b32.n_high_reuse, b32.n_requests,
            100.0*b32.n_high_reuse/b32.n_requests, b32.n_lost);
    fprintf(stderr, "Cache-aware: %zu prefill, %.0fs, %zu/%zu high-reuse (%.0f%%)\n",
            t32.prefill_tokens, t32.makespan_s, t32.n_high_reuse, t32.n_requests,
            100.0*t32.n_high_reuse/t32.n_requests);
    fprintf(stderr, "Delta: %.1f%% fewer prefill, %.1f%% shorter makespan\n\n",
            100.0*(1.0-double(t32.prefill_tokens)/b32.prefill_tokens),
            100.0*(1.0-t32.makespan_s/b32.makespan_s));

    p.cache_bytes = 16ull * 1024 * 1024 * 1024;
    fprintf(stderr, "=== 16 GiB (tighter pressure) ===\n");
    sim_result b16 = sim_run(p, false);
    sim_result t16 = sim_run(p, true);
    fprintf(stderr, "Baseline:    %zu prefill, %.0fs, %.0f%% high-reuse\n",
            b16.prefill_tokens, b16.makespan_s, 100.0*b16.n_high_reuse/b16.n_requests);
    fprintf(stderr, "Cache-aware: %zu prefill, %.0fs, %.0f%% high-reuse\n",
            t16.prefill_tokens, t16.makespan_s, 100.0*t16.n_high_reuse/t16.n_requests);
    fprintf(stderr, "Delta: %.1f%% fewer prefill, %.1f%% shorter makespan\n\n",
            100.0*(1.0-double(t16.prefill_tokens)/b16.prefill_tokens),
            100.0*(1.0-t16.makespan_s/b16.makespan_s));
}

int main(int argc, char ** argv) {
    if (argc > 1 && std::string(argv[1]) == "--metrics") {
        print_metrics();
        return 0;
    }

    testing t;
    if (argc > 1) {
        t.set_filter(argv[1]);
    }
    t.test("render_consistency", test_sim_render_is_self_consistent);
    t.test("sim", test_sim_beats_baseline_under_pressure);
    t.test("history_policies", test_sim_across_history_policies);
    return t.summary();
}
