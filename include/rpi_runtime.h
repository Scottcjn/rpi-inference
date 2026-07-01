/* SPDX-License-Identifier: MIT
 * rpi_runtime.h — Resonant Permutation Inference runtime
 *
 * Core inference loop:
 *   1. Activate seed cells for input token
 *   2. Run permutation rounds until convergence (hardware-timed)
 *   3. Collect emissions from active cells
 *   4. Select next token
 *
 * (c) 2026 Elyan Labs
 */

#ifndef RPI_RUNTIME_H
#define RPI_RUNTIME_H

#include "rpi_format.h"
#include <stddef.h>

#define RPI_MAX_ACTIVE_CELLS  128
#define RPI_MAX_VOCAB         65536

/* ── Hardware Profile ───────────────────────────────────── */
typedef struct {
    uint32_t n_threads;
    uint32_t n_numa_nodes;
    uint32_t cache_line_bytes;      /* typically 64 or 128 */
    uint32_t l1_size_kb;
    uint32_t l2_size_kb;
    uint32_t l3_size_kb;
    uint64_t tb_freq;               /* timebase ticks per second */
    uint32_t has_altivec;           /* G4/G5 AltiVec */
    uint32_t has_vsx;               /* POWER8 VSX */
    float    lat_l1_ns;             /* measured L1 latency */
    float    lat_l2_ns;
    float    lat_l3_ns;
    float    lat_dram_ns;
} RPIHWProfile;

/* ── Inference State ────────────────────────────────────── */
typedef struct {
    /* Active cell tracking */
    uint32_t active_ids[RPI_MAX_ACTIVE_CELLS];
    int16_t  active_lanes[RPI_MAX_ACTIVE_CELLS][RPI_LANES]; /* lane accumulators */
    uint32_t n_active;

    /* Recurrence state */
    uint32_t phase;                 /* current phase (0-3, maps to Dawn/Noon/Dusk/Night) */
    uint64_t sig_prev;              /* previous round signature for convergence */
    uint32_t round;                 /* current round within token */
    uint32_t total_tokens;          /* tokens generated so far */

    /* Token scoring */
    int32_t  tok_scores[RPI_MAX_VOCAB];
    uint32_t last_token;

    /* Timing */
    uint64_t tb_prev;              /* timebase at previous token */
    uint64_t tb_round_start;       /* timebase at round start */

    /* Cache resonance scores (per active cell) */
    float    lat_score[RPI_MAX_ACTIVE_CELLS];  /* measured latency score */

} RPIState;

/* ── Core API ───────────────────────────────────────────── */

/* Load model from file */
int rpi_model_load(RPIModel *model, const char *path);
void rpi_model_free(RPIModel *model);

/* Detect hardware capabilities */
void rpi_hw_detect(RPIHWProfile *hw);

/* Initialize inference state */
void rpi_state_init(RPIState *st);
void rpi_state_reset(RPIState *st);

/* Run one permutation round on all active cells */
void rpi_round(const RPIModel *model, const RPIHWProfile *hw,
               RPIState *st);

/* Activate seed cells for a token */
void rpi_activate_token(const RPIModel *model, RPIState *st,
                        uint32_t token_id);

/* Measure cache resonance for active cells */
void rpi_probe_resonance(const RPIModel *model, const RPIHWProfile *hw,
                         RPIState *st);

/* Collect emissions and score next token */
uint32_t rpi_emit_next(const RPIModel *model, const RPIHWProfile *hw,
                       RPIState *st);

/* Full decode: generate one token */
uint32_t rpi_decode_token(const RPIModel *model, const RPIHWProfile *hw,
                          RPIState *st);

/* Generate text */
void rpi_generate(const RPIModel *model, const RPIHWProfile *hw,
                  RPIState *st, const uint32_t *prompt_ids,
                  uint32_t prompt_len, uint32_t max_tokens,
                  uint32_t *output_ids, uint32_t *n_output);

/* ── Platform-specific permutation backends ─────────────── */

/* Generic C fallback */
void rpi_run_perm_block_c(const RPIPermBlock *block,
                          const int16_t *in, int16_t *out);

/* POWER8 VSX (vec_perm) */
#ifdef __POWER8_VECTOR__
void rpi_run_perm_block_vsx(const RPIPermBlock *block,
                            const int16_t *in, int16_t *out);
#endif

/* G4 AltiVec (vperm) */
#ifdef __ALTIVEC__
void rpi_run_perm_block_altivec(const RPIPermBlock *block,
                                const int16_t *in, int16_t *out);
#endif

/* ARM64 NEON (tbl) — AArch64 only (vqtbl4q_u8 is the 64-bit TBL form) */
#if defined(__aarch64__)
void rpi_run_perm_block_neon(const RPIPermBlock *block,
                             const int16_t *in, int16_t *out);
/* Precompute TBL control vectors for all blocks once (keeps the hot path
 * pure-vector). Safe to call repeatedly; a no-op for an already-built set. */
void rpi_neon_prepare(const RPIPermBlock *blocks, uint32_t n);
/* Invalidate the prepared table (call when the model is freed). */
void rpi_neon_reset(void);
#endif

/* ── Timebase helpers ───────────────────────────────────── */
static inline uint64_t rpi_tb_now(void) {
#if defined(__powerpc64__) || defined(__ppc64__)
    uint64_t tb;
    __asm__ volatile("mftb %0" : "=r"(tb));
    return tb;
#elif defined(__powerpc__) || defined(__ppc__)
    /* 32-bit PPC: read upper and lower timebase */
    uint32_t hi, lo, hi2;
    do {
        __asm__ volatile("mftbu %0" : "=r"(hi));
        __asm__ volatile("mftb  %0" : "=r"(lo));
        __asm__ volatile("mftbu %0" : "=r"(hi2));
    } while (hi != hi2);
    return ((uint64_t)hi << 32) | lo;
#elif defined(__x86_64__) || defined(_M_X64)
    uint32_t lo, hi;
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
#elif defined(__aarch64__)
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
#else
    return 0; /* fallback */
#endif
}

/* FNV-1a hash for convergence detection */
static inline uint64_t rpi_fnv1a(const void *data, size_t len) {
    uint64_t h = 0xcbf29ce484222325ULL;
    const uint8_t *p = (const uint8_t *)data;
    for (size_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

#endif /* RPI_RUNTIME_H */
