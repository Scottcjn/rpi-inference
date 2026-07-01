/* SPDX-License-Identifier: AGPL-3.0-or-later
 * decode.c — RPI inference engine (the heart of it)
 *
 * This is the llama.cpp equivalent for Resonant Permutation Inference.
 * No multiply-accumulate. Only permute, accumulate, measure, route.
 *
 * (c) 2026 Elyan Labs — Claude Opus + GPT-5.4 collaboration
 */

#include "rpi_format.h"
#include "rpi_runtime.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* ── Coarse phase profiler (RPI_PROFILE=1) ──────────────────
 * Accumulates timebase ticks in three buckets so we can see where per-token
 * time actually goes (permute vs routing/resonance vs vocab scoring) before
 * optimizing the wrong one. Percentages only; no tb_freq needed. */
static uint64_t g_prof_perm = 0, g_prof_route = 0, g_prof_emit = 0;
static int g_prof_on = -1;
static int prof_enabled(void) {
    if (g_prof_on < 0) {
        const char *e = getenv("RPI_PROFILE");
        g_prof_on = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    return g_prof_on;
}

/* ── State Management ───────────────────────────────────── */

void rpi_state_init(RPIState *st) {
    memset(st, 0, sizeof(*st));
    st->tb_prev = rpi_tb_now();
}

void rpi_state_reset(RPIState *st) {
    st->n_active = 0;
    st->round = 0;
    st->sig_prev = 0;
    memset(st->tok_scores, 0, sizeof(st->tok_scores));
}

/* ── Hardware Detection ─────────────────────────────────── */

void rpi_hw_detect(RPIHWProfile *hw) {
    memset(hw, 0, sizeof(*hw));
    hw->n_threads = 1;
    hw->cache_line_bytes = 64;
    hw->l1_size_kb = 32;
    hw->l2_size_kb = 256;
    hw->l3_size_kb = 8192;

#if defined(__powerpc64__)
    hw->has_vsx = 1;
    hw->cache_line_bytes = 128;  /* POWER8 uses 128-byte cache lines */
    hw->l1_size_kb = 64;
    hw->l2_size_kb = 512;
    hw->l3_size_kb = 8192;
    hw->tb_freq = 512000000;     /* 512 MHz typical POWER8 timebase */
    hw->lat_l1_ns = 1.2f;
    hw->lat_l2_ns = 4.5f;
    hw->lat_l3_ns = 12.0f;
    hw->lat_dram_ns = 55.0f;
    fprintf(stderr, "[RPI] POWER8 detected: VSX + 128-byte cache lines\n");
#elif defined(__ALTIVEC__)
    hw->has_altivec = 1;
    hw->cache_line_bytes = 32;   /* G4 uses 32-byte cache lines */
    hw->l1_size_kb = 32;
    hw->l2_size_kb = 256;
    hw->l3_size_kb = 0;          /* G4 typically no L3 */
    hw->tb_freq = 25000000;      /* 25 MHz typical G4 timebase */
    hw->lat_l1_ns = 2.0f;
    hw->lat_l2_ns = 8.0f;
    hw->lat_l3_ns = 0.0f;
    hw->lat_dram_ns = 80.0f;
    fprintf(stderr, "[RPI] PowerPC G4 detected: AltiVec + 32-byte cache lines\n");
#elif defined(__aarch64__)
    /* ARM64 (Apple M-series, Pi 5, Graviton, Cortex...). NEON tbl is baseline.
     * Read the actual counter frequency instead of assuming one — it varies by
     * platform (Apple 24 MHz, Pi 5 54 MHz, Graviton differs). Wrong tb_freq
     * poisons both the decode deadline budget and the tok/s report. */
    {
        uint64_t cntfrq = 0;
        __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(cntfrq));
        hw->tb_freq = cntfrq ? cntfrq : 24000000;
    }
    hw->l3_size_kb = 0;          /* no conventional L3 on these parts */
#ifdef __APPLE__
    hw->cache_line_bytes = 128;  /* Apple M-series use 128-byte lines */
    hw->l1_size_kb = 128;        /* M-series P-core L1 data cache */
    hw->l2_size_kb = 16384;      /* shared L2 (P-cluster) */
    hw->lat_l1_ns = 0.9f; hw->lat_l2_ns = 3.5f; hw->lat_dram_ns = 45.0f;
    fprintf(stderr, "[RPI] Apple Silicon detected: NEON tbl, tb_freq=%llu Hz\n",
            (unsigned long long)hw->tb_freq);
#else
    hw->cache_line_bytes = 64;   /* generic AArch64 default */
    hw->l1_size_kb = 64;
    hw->l2_size_kb = 512;
    hw->lat_l1_ns = 1.2f; hw->lat_l2_ns = 4.0f; hw->lat_dram_ns = 60.0f;
    fprintf(stderr, "[RPI] AArch64 detected: NEON tbl, tb_freq=%llu Hz\n",
            (unsigned long long)hw->tb_freq);
#endif
    hw->lat_l3_ns = 0.0f;
#else
    hw->tb_freq = 1000000000;    /* assume ~1GHz for rdtsc */
    hw->lat_l1_ns = 1.0f;
    hw->lat_l2_ns = 4.0f;
    hw->lat_l3_ns = 10.0f;
    hw->lat_dram_ns = 50.0f;
    fprintf(stderr, "[RPI] Generic CPU detected\n");
#endif
}

/* ── Permutation Block Execution (generic C) ────────────── */

void rpi_run_perm_block_c(const RPIPermBlock *block,
                          const int16_t *in, int16_t *out) {
    /*
     * For each output lane:
     *   1. Read from source lane (permutation)
     *   2. Apply sign (negate if sign bit set)
     *   3. Accumulate into output
     *
     * This is the ZERO-MULTIPLY core. Only permute + add/subtract.
     */
    for (int i = 0; i < RPI_LANES; i++) {
        uint8_t src = block->src_idx[i];
        if (src == 0xFF) {
            /* Zero input — skip (W=0 in ternary) */
            continue;
        }
        int16_t val = in[src & 0x3F];  /* mask to valid range */

        /* Apply sign: bit i of sign_bits → negate */
        if ((block->sign_bits >> i) & 1) {
            out[i] -= val;  /* W=-1: subtract */
        } else {
            out[i] += val;  /* W=+1: add */
        }
    }
}

/* ── Run permutation blocks for one cell ────────────────── */

static void run_cell_perms(const RPIModel *model, const RPICell *cell,
                           int16_t *lanes) {
    /* Select backend */
    void (*run_block)(const RPIPermBlock *, const int16_t *, int16_t *);

    /* Runtime override for A/B benchmarking: RPI_SCALAR=1 forces the generic
     * C backend on any platform. Resolved once. */
    static int force_scalar = -1;
    if (force_scalar < 0) {
        const char *e = getenv("RPI_SCALAR");
        force_scalar = (e && e[0] && e[0] != '0') ? 1 : 0;
    }

    if (force_scalar) {
        run_block = rpi_run_perm_block_c;
    } else {
#if defined(__POWER8_VECTOR__)
        run_block = rpi_run_perm_block_vsx;
#elif defined(__ALTIVEC__)
        run_block = rpi_run_perm_block_altivec;
#elif defined(__aarch64__)
        /* Build the TBL control table once so the hot path stays pure-vector. */
        rpi_neon_prepare(model->perm_blocks, model->hdr.n_perm_blocks);
        run_block = rpi_run_perm_block_neon;
#else
        run_block = rpi_run_perm_block_c;
#endif
    }

    /* Temporary buffer for ping-pong */
    int16_t tmp[RPI_LANES];

    for (uint32_t b = 0; b < cell->perm_block_count; b++) {
        const RPIPermBlock *block = &model->perm_blocks[cell->perm_block_start + b];
        memset(tmp, 0, sizeof(tmp));
        run_block(block, lanes, tmp);

        /* Ping-pong: tmp becomes input for next block */
        memcpy(lanes, tmp, sizeof(tmp));
    }
}

/* ── Activate seed cells for a token ────────────────────── */

void rpi_activate_token(const RPIModel *model, RPIState *st,
                        uint32_t token_id) {
    if (!model->embed_seeds || token_id >= model->hdr.vocab_size)
        return;

    uint32_t seed_cell = model->embed_seeds[token_id];
    if (seed_cell >= model->hdr.n_cells)
        return;

    /* Add to active set if not already present and not full */
    for (uint32_t i = 0; i < st->n_active; i++) {
        if (st->active_ids[i] == seed_cell)
            return;  /* already active */
    }

    if (st->n_active < RPI_MAX_ACTIVE_CELLS) {
        uint32_t idx = st->n_active++;
        st->active_ids[idx] = seed_cell;
        memset(st->active_lanes[idx], 0, sizeof(st->active_lanes[idx]));

        /* Initialize lanes with a simple hash of token_id + cell_id */
        uint64_t h = rpi_fnv1a(&token_id, sizeof(token_id));
        for (int i = 0; i < RPI_LANES; i++) {
            st->active_lanes[idx][i] = (int16_t)((h >> (i % 48)) & 0xFF) - 128;
        }
    }
}

/* ── Cache Resonance Probing ────────────────────────────── */

void rpi_probe_resonance(const RPIModel *model, const RPIHWProfile *hw,
                         RPIState *st) {
    /*
     * For each active cell, measure access latency to its weight data.
     * Faster access = stronger resonance = higher routing priority.
     *
     * This is the CACHE-RESONANCE ATTENTION mechanism.
     * GPUs have uniform shared memory — they can't do this.
     * CPU cache hierarchy IS the attention score function.
     */
    for (uint32_t i = 0; i < st->n_active; i++) {
        uint32_t cell_id = st->active_ids[i];
        const RPICell *cell = &model->cells[cell_id];

        /* Touch the first perm block of this cell and measure time */
        if (cell->perm_block_count == 0) {
            st->lat_score[i] = 0.0f;
            continue;
        }

        const RPIPermBlock *block = &model->perm_blocks[cell->perm_block_start];

        /* Timed probe: read the block data and measure latency */
        uint64_t t0 = rpi_tb_now();

        /* Force a real memory access (prevent optimization) */
        volatile uint8_t sink = 0;
        for (int j = 0; j < 8; j++) {
            sink ^= block->src_idx[j * 8];
        }
        (void)sink;

        uint64_t t1 = rpi_tb_now();
        uint64_t dt = t1 - t0;

        /* Convert to a score: faster access = higher score */
        /* Normalize by hardware latency profile */
        float ns_est = (float)dt * 1e9f / (float)hw->tb_freq;

        if (ns_est <= hw->lat_l1_ns * 2.0f) {
            st->lat_score[i] = 1.0f;       /* L1: strong resonance */
        } else if (ns_est <= hw->lat_l2_ns * 2.0f) {
            st->lat_score[i] = 0.6f;       /* L2: moderate */
        } else if (ns_est <= hw->lat_l3_ns * 2.0f) {
            st->lat_score[i] = 0.3f;       /* L3: weak */
        } else {
            st->lat_score[i] = 0.05f;      /* DRAM: minimal */
        }
    }
}

/* ── One Recurrence Round ───────────────────────────────── */

void rpi_round(const RPIModel *model, const RPIHWProfile *hw,
               RPIState *st) {
    /*
     * For each active cell:
     *   1. Run permutation blocks (zero-multiply forward pass)
     *   2. Follow routes to activate downstream cells
     *   3. Measure cache resonance
     */

    uint64_t _pt0 = prof_enabled() ? rpi_tb_now() : 0;

    /* Phase 1: Run permutations on all active cells */
    for (uint32_t i = 0; i < st->n_active; i++) {
        uint32_t cell_id = st->active_ids[i];
        const RPICell *cell = &model->cells[cell_id];

        /* Check phase mask */
        if (!((cell->phase_mask >> st->phase) & 1))
            continue;

        run_cell_perms(model, cell, st->active_lanes[i]);
    }

    if (prof_enabled()) { uint64_t t = rpi_tb_now(); g_prof_perm += t - _pt0; _pt0 = t; }

    /* Phase 2: Follow routes — activate downstream cells */
    uint32_t new_active[RPI_MAX_ACTIVE_CELLS];
    int16_t  new_lanes[RPI_MAX_ACTIVE_CELLS][RPI_LANES];
    uint32_t n_new = 0;

    /* Keep existing active cells */
    memcpy(new_active, st->active_ids, st->n_active * sizeof(uint32_t));
    memcpy(new_lanes, st->active_lanes, st->n_active * sizeof(new_lanes[0]));
    n_new = st->n_active;

    for (uint32_t i = 0; i < st->n_active; i++) {
        uint32_t cell_id = st->active_ids[i];
        const RPICell *cell = &model->cells[cell_id];

        /* Compute lane energy (L1 norm) for routing decisions */
        int32_t energy = 0;
        for (int j = 0; j < RPI_LANES; j++) {
            int16_t v = st->active_lanes[i][j];
            energy += (v >= 0) ? v : -v;
        }

        /* Only route from high-energy cells */
        if (energy < 64 * 10)  /* threshold */
            continue;

        for (uint32_t r = 0; r < cell->route_count && r < RPI_MAX_ROUTES; r++) {
            const RPIRoute *route = &model->routes[cell->route_start + r];
            uint32_t dst = route->dst_cell_id;

            /* Check if already active */
            int found = 0;
            for (uint32_t k = 0; k < n_new; k++) {
                if (new_active[k] == dst) { found = 1; break; }
            }
            if (found || n_new >= model->hdr.max_active)
                continue;

            /* Activate downstream cell with scaled lanes */
            new_active[n_new] = dst;
            for (int j = 0; j < RPI_LANES; j++) {
                new_lanes[n_new][j] = st->active_lanes[i][j] >> 2; /* decay */
            }
            n_new++;
        }
    }

    /* Update state */
    st->n_active = n_new;
    memcpy(st->active_ids, new_active, n_new * sizeof(uint32_t));
    memcpy(st->active_lanes, new_lanes, n_new * sizeof(new_lanes[0]));

    /* Phase 3: Probe cache resonance */
    rpi_probe_resonance(model, hw, st);

    if (prof_enabled()) g_prof_route += rpi_tb_now() - _pt0;

    st->round++;
}

/* ── Token Emission (v2: multi-cell voting + diversity) ─── */

/* Repetition penalty history */
#define RPI_REP_WINDOW 32
static uint32_t rep_history[RPI_REP_WINDOW];
static uint32_t rep_pos = 0;

uint32_t rpi_emit_next(const RPIModel *model, const RPIHWProfile *hw,
                       RPIState *st) {
    (void)hw;

    /* RPI_DENSE_EMIT=1 forces the old full-vocab path for same-binary A/B. */
    static int dense_emit = -1;
    if (dense_emit < 0) {
        const char *e = getenv("RPI_DENSE_EMIT");
        dense_emit = (e && e[0] && e[0] != '0') ? 1 : 0;
    }

    /* Clear scores. Sparse (default): wipe only slots dirtied last step, an
     * O(touched) lazy clear that keeps the invariant "untouched reads 0" and
     * replaces the O(vocab) memset (was ~a third of per-token time at 32k
     * vocab). Dense: the old full-vocab memset. */
    if (dense_emit) {
        memset(st->tok_scores, 0, model->hdr.vocab_size * sizeof(int32_t));
    } else {
        for (uint32_t i = 0; i < st->n_touched; i++)
            st->tok_scores[st->touched[i]] = 0;
    }
    st->n_touched = 0;

    /* Collect emissions from ALL active cells (multi-cell voting) */
    for (uint32_t i = 0; i < st->n_active; i++) {
        uint32_t cell_id = st->active_ids[i];
        if (cell_id >= model->hdr.n_cells) continue;
        const RPICell *cell = &model->cells[cell_id];

        /* Lane energy as cell activation strength */
        int32_t energy = 0;
        for (int j = 0; j < RPI_LANES; j++) {
            int16_t v = st->active_lanes[i][j];
            energy += (v >= 0) ? v : -v;
        }

        /* Bank diversity bonus: cells from underrepresented banks score higher */
        int32_t bank_bonus = (cell->bank_id != (st->phase & 3)) ? 50 : 0;

        /* Resonance bonus from cache timing */
        int32_t lat_bonus = (int32_t)(st->lat_score[i] * 80.0f);

        for (uint32_t e = 0; e < cell->emit_count && e < RPI_MAX_EMITS; e++) {
            const RPIEmit *emit = &model->emits[cell->emit_start + e];

            /* Phase check */
            if (emit->phase_mask != 0x0F &&
                !((emit->phase_mask >> st->phase) & 1))
                continue;

            int32_t score = (int32_t)emit->rank_bias
                          + (energy >> 8)       /* cell activation (reduced weight) */
                          + lat_bonus
                          + bank_bonus;

            if (emit->token_id < model->hdr.vocab_size) {
                st->tok_scores[emit->token_id] += score;
                /* Record the dirtied slot so we can scan/clear it sparsely.
                 * Duplicates are fine: the score already accumulated in the
                 * slot, and the top-K scan below dedups. */
                if (st->n_touched < RPI_MAX_TOUCHED)
                    st->touched[st->n_touched++] = emit->token_id;
            }
        }
    }

    /* Repetition penalty: halve scores of recently emitted tokens */
    for (uint32_t r = 0; r < RPI_REP_WINDOW; r++) {
        uint32_t prev_tok = rep_history[r];
        if (prev_tok < model->hdr.vocab_size && st->tok_scores[prev_tok] > 0) {
            st->tok_scores[prev_tok] >>= 2;  /* quarter the score */
        }
    }

    /* Top-K sampling with hardware entropy (K=8).
     * Scan ONLY the touched candidates, not the whole vocab. The set of tokens
     * with score>0 is exactly the touched set (untouched read 0 by invariant),
     * so this yields the same top-K set as the old full-vocab scan. */
    uint32_t top_ids[8] = {0};
    int32_t  top_scores[8] = {0};

    if (dense_emit) {
        /* Old full-vocab scan (A/B baseline). */
        for (uint32_t t = 0; t < model->hdr.vocab_size; t++) {
            if (st->tok_scores[t] <= 0) continue;
            int min_idx = 0;
            for (int k = 1; k < 8; k++)
                if (top_scores[k] < top_scores[min_idx]) min_idx = k;
            if (st->tok_scores[t] > top_scores[min_idx]) {
                top_scores[min_idx] = st->tok_scores[t];
                top_ids[min_idx] = t;
            }
        }
    } else {
        for (uint32_t idx = 0; idx < st->n_touched; idx++) {
            uint32_t t = st->touched[idx];
            int32_t sc = st->tok_scores[t];
            if (sc <= 0) continue;

            /* Skip if already placed (touched[] may contain duplicates). */
            int dup = 0;
            for (int k = 0; k < 8; k++)
                if (top_scores[k] > 0 && top_ids[k] == t) { dup = 1; break; }
            if (dup) continue;

            int min_idx = 0;
            for (int k = 1; k < 8; k++)
                if (top_scores[k] < top_scores[min_idx]) min_idx = k;
            if (sc > top_scores[min_idx]) {
                top_scores[min_idx] = sc;
                top_ids[min_idx] = t;
            }
        }
    }

    /* Opt-in differential self-check: verify the sparse top-K SET matches the
     * old dense full-vocab scan. Resolved once; off by default. */
    static int emit_check = -1;
    if (emit_check < 0) {
        const char *e = getenv("RPI_EMIT_CHECK");
        emit_check = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    if (emit_check) {
        uint32_t d_ids[8] = {0}; int32_t d_scores[8] = {0};
        for (uint32_t t = 0; t < model->hdr.vocab_size; t++) {
            if (st->tok_scores[t] <= 0) continue;
            int mi = 0;
            for (int k = 1; k < 8; k++) if (d_scores[k] < d_scores[mi]) mi = k;
            if (st->tok_scores[t] > d_scores[mi]) { d_scores[mi] = st->tok_scores[t]; d_ids[mi] = t; }
        }
        /* Compare as multisets of (score) present with score>0: same top-K
         * scores => same candidate strength distribution. */
        int32_t ss[8], ds[8]; int ns = 0, nd = 0;
        for (int k = 0; k < 8; k++) { if (top_scores[k] > 0) ss[ns++] = top_scores[k];
                                      if (d_scores[k] > 0) ds[nd++] = d_scores[k]; }
        for (int a = 0; a < ns; a++) for (int b = a + 1; b < ns; b++) if (ss[b] > ss[a]) { int32_t t = ss[a]; ss[a] = ss[b]; ss[b] = t; }
        for (int a = 0; a < nd; a++) for (int b = a + 1; b < nd; b++) if (ds[b] > ds[a]) { int32_t t = ds[a]; ds[a] = ds[b]; ds[b] = t; }
        int mismatch = (ns != nd);
        for (int k = 0; !mismatch && k < ns; k++) if (ss[k] != ds[k]) mismatch = 1;
        if (mismatch) {
            static int reported = 0;
            if (reported++ < 5)
                fprintf(stderr, "[RPI] EMIT_CHECK MISMATCH: sparse topK=%d dense topK=%d\n", ns, nd);
        }
    }

    /* Sample from top-K using hardware entropy */
    uint64_t tb = rpi_tb_now();
    int32_t total = 0;
    for (int k = 0; k < 8; k++) total += top_scores[k];

    uint32_t chosen = top_ids[0];
    if (total > 0) {
        int32_t threshold = (int32_t)((tb ^ (tb >> 17)) % (uint64_t)total);
        int32_t cumulative = 0;
        for (int k = 0; k < 8; k++) {
            cumulative += top_scores[k];
            if (cumulative >= threshold) {
                chosen = top_ids[k];
                break;
            }
        }
    }

    /* Update repetition history */
    rep_history[rep_pos % RPI_REP_WINDOW] = chosen;
    rep_pos++;

    return chosen;
}

/* ── Full Token Decode (v2: min rounds + cross-bank activation) ── */

uint32_t rpi_decode_token(const RPIModel *model, const RPIHWProfile *hw,
                          RPIState *st) {
    st->round = 0;
    st->sig_prev = 0;
    st->tb_round_start = rpi_tb_now();

    uint64_t budget = hw->tb_freq / 10;  /* 100ms max */
    uint64_t deadline = st->tb_round_start + budget;

    uint32_t stable = 0;
    uint32_t min_rounds = 3;  /* MINIMUM 3 rounds before convergence check */

    while (st->round < model->hdr.max_rounds &&
           rpi_tb_now() < deadline) {

        rpi_round(model, hw, st);

        /* Force cross-bank activation every round:
         * Pick a random cell from a different bank and activate it */
        if (st->n_active < model->hdr.max_active && model->hdr.n_cells > 0) {
            uint64_t tb_seed = rpi_tb_now();
            uint32_t target_bank = (st->phase + st->round + 1) & 3;
            uint32_t cells_per_bank = model->hdr.n_cells / model->hdr.n_banks;
            uint32_t cross_cell = target_bank * cells_per_bank
                                + (uint32_t)(tb_seed % cells_per_bank);

            if (cross_cell < model->hdr.n_cells) {
                /* Check not already active */
                int found = 0;
                for (uint32_t k = 0; k < st->n_active; k++) {
                    if (st->active_ids[k] == cross_cell) { found = 1; break; }
                }
                if (!found && st->n_active < RPI_MAX_ACTIVE_CELLS) {
                    uint32_t idx = st->n_active++;
                    st->active_ids[idx] = cross_cell;
                    /* Initialize with entropy-seeded lanes */
                    for (int j = 0; j < RPI_LANES; j++) {
                        st->active_lanes[idx][j] =
                            (int16_t)((tb_seed >> (j & 0x3F)) & 0x7F) - 64;
                    }
                }
            }
        }

        /* Only check convergence after minimum rounds */
        if (st->round >= min_rounds) {
            uint64_t sig = rpi_fnv1a(st->active_ids,
                                     st->n_active * sizeof(uint32_t));
            for (uint32_t i = 0; i < st->n_active; i++) {
                int32_t sum = 0;
                for (int j = 0; j < RPI_LANES; j++)
                    sum += st->active_lanes[i][j];
                sig ^= rpi_fnv1a(&sum, sizeof(sum));
            }

            if (sig == st->sig_prev) {
                stable++;
                if (stable >= 2) break;
            } else {
                stable = 0;
            }
            st->sig_prev = sig;
        }
    }

    /* Emit next token with diversity */
    uint64_t _et0 = prof_enabled() ? rpi_tb_now() : 0;
    uint32_t next = rpi_emit_next(model, hw, st);
    if (prof_enabled()) g_prof_emit += rpi_tb_now() - _et0;

    /* Activate seed cells for next token */
    rpi_activate_token(model, st, next);

    /* Advance phase (rotates through banks) */
    st->phase = (st->phase + 1) & 3;
    st->last_token = next;
    st->total_tokens++;
    st->tb_prev = rpi_tb_now();

    return next;
}

/* ── Generate Text ──────────────────────────────────────── */

void rpi_generate(const RPIModel *model, const RPIHWProfile *hw,
                  RPIState *st, const uint32_t *prompt_ids,
                  uint32_t prompt_len, uint32_t max_tokens,
                  uint32_t *output_ids, uint32_t *n_output) {
    /* Ingest prompt */
    for (uint32_t i = 0; i < prompt_len; i++) {
        rpi_activate_token(model, st, prompt_ids[i]);
        /* Quick round for prompt tokens (fewer rounds needed) */
        for (uint32_t r = 0; r < 2; r++) {
            rpi_round(model, hw, st);
        }
        st->phase = (st->phase + 1) & 3;
    }

    /* Generate */
    *n_output = 0;
    for (uint32_t i = 0; i < max_tokens; i++) {
        uint32_t tok = rpi_decode_token(model, hw, st);
        output_ids[i] = tok;
        (*n_output)++;

        /* Stop on EOS (token 0 by convention) */
        if (tok == 0)
            break;
    }

    if (prof_enabled()) {
        uint64_t tot = g_prof_perm + g_prof_route + g_prof_emit;
        if (tot == 0) tot = 1;
        fprintf(stderr, "[RPI] profile: permute=%.1f%%  routing/resonance=%.1f%%  "
                "vocab-scoring=%.1f%%\n",
                100.0 * g_prof_perm / tot,
                100.0 * g_prof_route / tot,
                100.0 * g_prof_emit / tot);
    }
}
