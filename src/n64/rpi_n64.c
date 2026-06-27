/* SPDX-License-Identifier: AGPL-3.0-or-later
 * rpi_n64.c — RPI inference for N64 MIPS R4300i
 *
 * Zero-multiply inference engine for Legend of Elya.
 * Sophia speaks through permutation tables.
 * No floating point. No matrix multiply. Just lookup + accumulate.
 *
 * (c) 2026 Elyan Labs
 */

#include <stdint.h>
#include <string.h>

/* ── Compact structures for N64 (minimal RAM) ─────────── */

#define RPI_MAGIC       0x21495052
#define RPI_MAX_EMITS   32
#define RPI_MAX_ACTIVE  16  /* N64 has limited RAM */
#define RPI_REP_WINDOW  12

typedef struct {
    uint32_t magic, version, n_cells, n_perm, n_routes, n_emits;
    uint32_t vocab_size, lane_width, max_rounds, max_active, n_banks;
    uint32_t embed_offset, flags;
    uint8_t  reserved[76];
} RPIHeader;

typedef struct {
    uint32_t cell_id, bank_id, perm_start, perm_count;
    uint32_t route_start, route_count, emit_start, emit_count;
    uint16_t phase_mask, priority;
} RPICell;

typedef struct {
    uint32_t token_id;
    uint16_t rank_bias;
    uint8_t  phase_mask, reserved;
} RPIEmit;

typedef struct {
    uint32_t dst_cell_id;
    uint16_t token_class;
    uint8_t  boundary_type, weight;
} RPIRoute;

/* ── N64 RPI Model (loaded from ROM) ──────────────────── */

typedef struct {
    const RPIHeader *hdr;
    const RPICell   *cells;
    const RPIEmit   *emits;
    const RPIRoute  *routes;
    const uint32_t  *embed_seeds;
    uint32_t         n_cells;
    uint32_t         vocab_size;
} RPIN64Model;

/* ── Inference State ──────────────────────────────────── */

typedef struct {
    uint32_t last_token;
    uint32_t rep_history[RPI_REP_WINDOW];
    uint32_t rep_pos;
    uint32_t total_generated;
    /* Simple LFSR for randomness (no FPU needed) */
    uint32_t rng_state;
} RPIN64State;

/* ── LFSR pseudo-random (no multiply, no FPU) ─────────── */

static uint32_t rpi_rand(RPIN64State *st) {
    /* xorshift32 — zero multiplies */
    uint32_t x = st->rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    st->rng_state = x;
    return x;
}

/* ── Load model from ROM address ──────────────────────── */

void rpi_n64_load(RPIN64Model *model, const void *rom_addr) {
    const uint8_t *p = (const uint8_t *)rom_addr;

    model->hdr = (const RPIHeader *)p;
    model->n_cells = model->hdr->n_cells;
    model->vocab_size = model->hdr->vocab_size;

    p += 128;  /* header */
    p += model->hdr->n_banks * 32;  /* bank descriptors */

    model->cells = (const RPICell *)p;
    p += model->n_cells * sizeof(RPICell);

    p += model->hdr->n_perm * 72;  /* skip perm blocks (not used in Markov mode) */

    model->routes = (const RPIRoute *)p;
    p += model->hdr->n_routes * sizeof(RPIRoute);

    model->emits = (const RPIEmit *)p;
    p += model->hdr->n_emits * sizeof(RPIEmit);

    model->embed_seeds = (const uint32_t *)((const uint8_t *)rom_addr + model->hdr->embed_offset);
}

/* ── Initialize state ─────────────────────────────────── */

void rpi_n64_init(RPIN64State *st, uint32_t seed) {
    memset(st, 0, sizeof(*st));
    st->rng_state = seed ? seed : 0xDEADBEEF;
    st->last_token = 0;
}

/* ── Generate next token (zero multiply) ──────────────── */

uint32_t rpi_n64_next(const RPIN64Model *model, RPIN64State *st) {
    uint32_t cell_idx = st->last_token;
    if (cell_idx >= model->n_cells)
        cell_idx = cell_idx % model->n_cells;

    const RPICell *cell = &model->cells[cell_idx];

    /* Collect emissions */
    uint32_t best_ids[8];
    uint32_t best_scores[8];
    uint32_t n_candidates = 0;

    uint32_t emit_end = cell->emit_start + cell->emit_count;
    if (emit_end > model->hdr->n_emits)
        emit_end = model->hdr->n_emits;

    for (uint32_t e = cell->emit_start; e < emit_end && n_candidates < 8; e++) {
        const RPIEmit *emit = &model->emits[e];
        uint32_t tid = emit->token_id;
        uint32_t score = emit->rank_bias;

        /* Repetition penalty — check last N tokens */
        int is_repeat = 0;
        for (uint32_t r = 0; r < RPI_REP_WINDOW; r++) {
            if (st->rep_history[r] == tid) {
                is_repeat = 1;
                break;
            }
        }
        if (is_repeat) {
            score >>= 2;  /* quarter score for repeats */
        }

        if (score > 0) {
            best_ids[n_candidates] = tid;
            best_scores[n_candidates] = score;
            n_candidates++;
        }
    }

    if (n_candidates == 0) {
        /* Fallback: emit a common token */
        st->last_token = 0;
        return 0;
    }

    /* Weighted random selection (no multiply — use shifts + adds) */
    uint32_t total = 0;
    for (uint32_t i = 0; i < n_candidates; i++) {
        total += best_scores[i];
    }

    uint32_t r = rpi_rand(st) % total;
    uint32_t cumulative = 0;
    uint32_t chosen = best_ids[0];

    for (uint32_t i = 0; i < n_candidates; i++) {
        cumulative += best_scores[i];
        if (cumulative > r) {
            chosen = best_ids[i];
            break;
        }
    }

    /* Update state */
    st->rep_history[st->rep_pos % RPI_REP_WINDOW] = chosen;
    st->rep_pos++;
    st->last_token = chosen;
    st->total_generated++;

    return chosen;
}

/* ── Generate a string of tokens ──────────────────────── */

uint32_t rpi_n64_generate(const RPIN64Model *model, RPIN64State *st,
                          const uint32_t *prompt, uint32_t prompt_len,
                          uint32_t *output, uint32_t max_tokens) {
    /* Seed from last prompt token */
    if (prompt_len > 0) {
        st->last_token = prompt[prompt_len - 1];
    }

    uint32_t generated = 0;
    for (uint32_t i = 0; i < max_tokens; i++) {
        uint32_t tok = rpi_n64_next(model, st);
        output[i] = tok;
        generated++;

        /* Stop on EOS or end-of-turn marker */
        if (tok == 0)
            break;
    }

    return generated;
}
