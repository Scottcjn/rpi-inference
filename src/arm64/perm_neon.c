/* SPDX-License-Identifier: MIT
 * perm_neon.c — ARM64 NEON permutation backend for RPI
 *
 * The AArch64 answer to POWER's vec_perm is TBL (vqtbl): a byte-wise
 * table lookup that gathers up to 16 output bytes from a register table
 * in a single instruction, returning 0 for any out-of-range index.
 *
 * Same zero-multiply idea as perm_vsx.c, done as a real hardware permute:
 *   - the 64-lane gather runs through vqtbl4q_u8 (TBL), not a C loop
 *   - the ternary sign is a conditional-negate-via-select (vbslq), never a
 *     multiply
 *   - accumulation is vaddq_s16
 *
 * KEY PERFORMANCE NOTE: a block's src_idx/sign_bits are constant for the
 * whole run, but turning them into TBL index vectors + a sign mask is 64
 * lanes of scalar work. Doing that per call makes the vector path LOSE to
 * the scalar loop (measured 0.46x). So we split:
 *   build_ctrl()  — scalar, run ONCE per block via rpi_neon_prepare()
 *   apply_ctrl()  — the hot path, pure vector, zero scalar setup
 * rpi_run_perm_block_neon() uses the prepared table when available and
 * falls back to an on-the-fly build otherwise (keeps direct callers correct).
 *
 * Unlocks the M-series Macs, Pi 5, and any AArch64 core. TBL is baseline on
 * AArch64, so no special -mfpu/-march flag is required for correctness.
 *
 * (c) 2026 Elyan Labs
 */

#include "rpi_format.h"
#include "rpi_runtime.h"

/* AArch64 only: vqtbl4q_u8 is the 64-bit TBL form, absent on 32-bit ARMv7. */
#if defined(__aarch64__)

#include <arm_neon.h>
#include <stdlib.h>
#include <string.h>

/*
 * Contract (identical to rpi_run_perm_block_c):
 *   for each output lane i in 0..63:
 *     s = block->src_idx[i]
 *     if (s == 0xFF)             out[i] += 0        (ternary W=0, skip)
 *     else if (sign bit i set)   out[i] -= in[s & 0x3F]   (W=-1)
 *     else                       out[i] += in[s & 0x3F]   (W=+1)
 *
 * Lanes are int16, so 64 lanes = 128 bytes = eight q-registers.
 *
 * TBL detail: vqtbl4q_u8 indexes a 64-byte table (four q-registers = 32
 * int16 lanes) and returns 0 for indices >= 64. The source spans 64 lanes
 * (128 bytes), so we split it into two groups:
 *     group A = input lanes  0..31  (bytes  0..63)
 *     group B = input lanes 32..63  (bytes 64..127)
 * For each output byte we point exactly one group at the wanted byte and the
 * other out of range (0xFF -> 0). OR-merging the two TBL results is exact,
 * because the inactive group contributes all-zero bytes. A 16-bit gather is
 * two byte gathers: input lane s reads bytes (2s, 2s+1); a W=0 lane points
 * both bytes out of range so the halfword is 0.
 */

/* Compiled control for one block: 8 chunks of 8 int16 lanes. */
typedef struct {
    uint8_t  idxA[8][16];   /* TBL byte indices into group A per chunk */
    uint8_t  idxB[8][16];   /* TBL byte indices into group B per chunk */
    uint16_t mask[8][8];    /* per-lane negate mask (0xFFFF = negate)   */
} NeonCtrl;

static void build_ctrl(const RPIPermBlock *block, NeonCtrl *c) {
    for (int chunk = 0; chunk < 8; chunk++) {
        const int base = chunk * 8;
        const uint8_t sbyte = (uint8_t)((block->sign_bits >> base) & 0xFFu);
        for (int e = 0; e < 8; e++) {
            const int lane = base + e;
            const uint8_t s = block->src_idx[lane];
            const int bo = e * 2;  /* byte offset of this halfword in the chunk */

            if (s == 0xFF) {                 /* W=0: gather zero */
                c->idxA[chunk][bo] = c->idxA[chunk][bo + 1] = 0xFF;
                c->idxB[chunk][bo] = c->idxB[chunk][bo + 1] = 0xFF;
            } else {
                const uint8_t sl = s & 0x3F;
                if (sl < 32) {
                    const uint8_t b = (uint8_t)(sl * 2);
                    c->idxA[chunk][bo] = b; c->idxA[chunk][bo + 1] = (uint8_t)(b + 1);
                    c->idxB[chunk][bo] = c->idxB[chunk][bo + 1] = 0xFF;
                } else {
                    const uint8_t b = (uint8_t)((sl - 32) * 2);
                    c->idxB[chunk][bo] = b; c->idxB[chunk][bo + 1] = (uint8_t)(b + 1);
                    c->idxA[chunk][bo] = c->idxA[chunk][bo + 1] = 0xFF;
                }
            }
            c->mask[chunk][e] = ((sbyte >> e) & 1u) ? 0xFFFFu : 0x0000u;
        }
    }
}

/* Hot path: pure vector, no scalar setup. */
static inline void apply_ctrl(const NeonCtrl *c, const int16_t *in, int16_t *out) {
    const uint8_t *inb = (const uint8_t *)in;
    uint8x16x4_t grpA = vld1q_u8_x4(inb);        /* input lanes  0..31 */
    uint8x16x4_t grpB = vld1q_u8_x4(inb + 64);   /* input lanes 32..63 */

    for (int chunk = 0; chunk < 8; chunk++) {
        const int base = chunk * 8;
        uint8x16_t vidxA = vld1q_u8(c->idxA[chunk]);
        uint8x16_t vidxB = vld1q_u8(c->idxB[chunk]);

        /* The permute: two TBLs + OR-merge = gathered halfwords. */
        uint8x16_t ga = vqtbl4q_u8(grpA, vidxA);
        uint8x16_t gb = vqtbl4q_u8(grpB, vidxB);
        int16x8_t gathered = vreinterpretq_s16_u8(vorrq_u8(ga, gb));

        /* Ternary sign: select +gathered / -gathered. No multiply. */
        int16x8_t neg  = vnegq_s16(gathered);
        uint16x8_t msk = vld1q_u16(c->mask[chunk]);
        int16x8_t sv   = vbslq_s16(msk, neg, gathered);

        int16x8_t acc = vld1q_s16(out + base);
        acc = vaddq_s16(acc, sv);
        vst1q_s16(out + base, acc);
    }
}

/* ── Prepared-control table (built once per model) ──────────── */
static const RPIPermBlock *g_base = NULL;
static NeonCtrl *g_ctrl = NULL;
static uint32_t g_n = 0;

/* Invalidate the prepared table. MUST be called when the block array is freed
 * (see rpi_model_free): otherwise a new model malloc'd at the same address with
 * the same n_perm_blocks would match the pointer-identity check below and reuse
 * stale control vectors (an ABA hazard). */
void rpi_neon_reset(void) {
    g_base = NULL;   /* invalidate BEFORE free so no lookup indexes freed memory */
    g_n = 0;
    free(g_ctrl);
    g_ctrl = NULL;
}

/* Precompute control vectors for a contiguous block array. Called once after
 * model load (see decode.c). Idempotent for the same base pointer.
 * NOTE: uses process-global state; the engine is single-threaded. If inference
 * is ever parallelised, this table must move into model-owned state or be
 * guarded — see the tri-brain review notes. */
void rpi_neon_prepare(const RPIPermBlock *blocks, uint32_t n) {
    if (blocks == g_base && n == g_n && g_ctrl) return;  /* already built */
    rpi_neon_reset();                                    /* invalidate then rebuild */
    if (!blocks || n == 0) return;                       /* nothing to prepare */
    g_ctrl = (NeonCtrl *)malloc((size_t)n * sizeof(NeonCtrl));
    if (!g_ctrl) return;                                 /* fall back to on-the-fly */
    for (uint32_t i = 0; i < n; i++) build_ctrl(&blocks[i], &g_ctrl[i]);
    g_base = blocks;
    g_n = n;
}

void rpi_run_perm_block_neon(const RPIPermBlock *block,
                             const int16_t *in, int16_t *out) {
    /* Fast path: use the prepared control if this block is in the table. */
    if (g_ctrl && g_base && block >= g_base && block < g_base + g_n) {
        apply_ctrl(&g_ctrl[block - g_base], in, out);
        return;
    }
    /* Fallback: build once on the stack (direct callers, e.g. unit tests). */
    NeonCtrl c;
    build_ctrl(block, &c);
    apply_ctrl(&c, in, out);
}

#endif /* __aarch64__ */
