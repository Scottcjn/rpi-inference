/* SPDX-License-Identifier: MIT
 * test_perm_neon.c — differential correctness + microbenchmark for the
 * ARM64 NEON permutation backend.
 *
 * Correctness: run many random RPIPermBlocks through an independent scalar
 * reference and through rpi_run_perm_block_neon, assert bit-identical output.
 * This isolates kernel correctness from the engine's timing-driven
 * nondeterminism, and proves the NEON path stays a faithful zero-multiply
 * permute + add/subtract.
 *
 * Benchmark: time the same fixed block set through the scalar reference and
 * through NEON, report ns/block and speedup. A kernel-level number, not an
 * end-to-end tok/s guess.
 *
 * Build (on the M2):
 *   clang -O3 -mcpu=apple-m2 -I include \
 *       tools/test_perm_neon.c src/arm64/perm_neon.c -o test_perm && ./test_perm
 *
 * (c) 2026 Elyan Labs
 */

#include "rpi_format.h"
#include "rpi_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Independent scalar reference — a second implementation of the contract,
 * intentionally NOT the shipped generic C, so the test also checks the spec. */
static void perm_ref(const RPIPermBlock *b, const int16_t *in, int16_t *out) {
    for (int i = 0; i < RPI_LANES; i++) {
        uint8_t s = b->src_idx[i];
        if (s == 0xFF) continue;
        int16_t v = in[s & 0x3F];
        if ((b->sign_bits >> i) & 1) out[i] -= v;
        else                         out[i] += v;
    }
}

static int16_t rnd_lane(void) { return (int16_t)((rand() % 8001) - 4000); }
static int16_t rnd_seed(void) { return (int16_t)((rand() % 2001) - 1000); }

static void fill_block(RPIPermBlock *b) {
    for (int i = 0; i < RPI_LANES; i++) {
        int r = rand() % 80;              /* ~20% land on 0xFF (W=0 skip) */
        b->src_idx[i] = (r >= 64) ? 0xFF : (uint8_t)r;
    }
    b->sign_bits = ((uint64_t)rand() << 33) ^ ((uint64_t)rand() << 11) ^ rand();
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
    srand(0xE1A11AB);

    /* ── Correctness: 200k random blocks ─────────────────────── */
    const int N = 200000;
    int fails = 0;
    for (int t = 0; t < N; t++) {
        RPIPermBlock b;
        int16_t in[RPI_LANES], o_ref[RPI_LANES], o_neon[RPI_LANES];
        fill_block(&b);
        for (int i = 0; i < RPI_LANES; i++) in[i] = rnd_lane();
        for (int i = 0; i < RPI_LANES; i++) {
            int16_t seed = rnd_seed();      /* pre-seed the += accumulator */
            o_ref[i] = seed;
            o_neon[i] = seed;
        }
        perm_ref(&b, in, o_ref);
        rpi_run_perm_block_neon(&b, in, o_neon);
        if (memcmp(o_ref, o_neon, sizeof(o_ref)) != 0) {
            if (fails < 3) {
                printf("MISMATCH at block %d\n", t);
                for (int i = 0; i < RPI_LANES; i++)
                    if (o_ref[i] != o_neon[i])
                        printf("  lane %2d: ref=%6d neon=%6d src=%3u sign=%d\n",
                               i, o_ref[i], o_neon[i], b.src_idx[i],
                               (int)((b.sign_bits >> i) & 1));
            }
            fails++;
        }
    }
    printf("[correctness] %d random blocks: %s (%d mismatches)\n",
           N, fails ? "FAIL" : "PASS", fails);

    /* ── Write-variant: out = ±in[src] (no accumulate), incl. IN-PLACE ── */
    int wfails = 0;
    for (int t = 0; t < 50000; t++) {
        RPIPermBlock b;
        int16_t in[RPI_LANES], exp[RPI_LANES], o[RPI_LANES], ip[RPI_LANES];
        fill_block(&b);
        for (int i = 0; i < RPI_LANES; i++) in[i] = rnd_lane();
        for (int i = 0; i < RPI_LANES; i++) {
            uint8_t s = b.src_idx[i];
            if (s == 0xFF) exp[i] = 0;
            else {
                int16_t v = in[s & 0x3F];
                exp[i] = (int16_t)(((b.sign_bits >> i) & 1) ? -v : v);
            }
        }
        for (int i = 0; i < RPI_LANES; i++) o[i] = rnd_seed(); /* garbage pre-fill */
        rpi_run_perm_block_neon_write(&b, in, o);
        if (memcmp(exp, o, sizeof(exp)) != 0) wfails++;
        /* In-place: out aliases in. Must still equal exp. */
        memcpy(ip, in, sizeof(ip));
        rpi_run_perm_block_neon_write(&b, ip, ip);
        if (memcmp(exp, ip, sizeof(exp)) != 0) wfails++;
    }
    printf("[correctness] write-variant + in-place, 50000 blocks: %s (%d mismatches)\n",
           wfails ? "FAIL" : "PASS", wfails);
    fails += wfails;

    /* ── Fixed contiguous block set for the prepared fast path ─ */
    const int NB = 4096;      /* working set of blocks */
    const int ITERS = 20000;  /* repeat count */
    RPIPermBlock *blocks = malloc(sizeof(RPIPermBlock) * NB);
    int16_t in[RPI_LANES], out[RPI_LANES];
    for (int i = 0; i < NB; i++) fill_block(&blocks[i]);
    for (int i = 0; i < RPI_LANES; i++) in[i] = rnd_lane();

    /* Correctness of the prepared path against the reference. */
    rpi_neon_prepare(blocks, NB);
    int prep_fails = 0;
    for (int i = 0; i < NB; i++) {
        int16_t a[RPI_LANES], b[RPI_LANES];
        for (int j = 0; j < RPI_LANES; j++) a[j] = b[j] = rnd_seed();
        perm_ref(&blocks[i], in, a);
        rpi_run_perm_block_neon(&blocks[i], in, b);   /* uses prepared table */
        if (memcmp(a, b, sizeof(a)) != 0) prep_fails++;
    }
    printf("[correctness] prepared path, %d blocks: %s (%d mismatches)\n",
           NB, prep_fails ? "FAIL" : "PASS", prep_fails);
    fails += prep_fails;

    volatile int16_t sink = 0;

    double t0 = now_s();
    for (int it = 0; it < ITERS; it++) {
        memset(out, 0, sizeof(out));
        for (int i = 0; i < NB; i++) perm_ref(&blocks[i], in, out);
        sink ^= out[it & (RPI_LANES - 1)];
    }
    double t_ref = now_s() - t0;

    /* Prepared NEON (control vectors already built): the shippable hot path. */
    t0 = now_s();
    for (int it = 0; it < ITERS; it++) {
        memset(out, 0, sizeof(out));
        for (int i = 0; i < NB; i++) rpi_run_perm_block_neon(&blocks[i], in, out);
        sink ^= out[it & (RPI_LANES - 1)];
    }
    double t_neon = now_s() - t0;

    double calls = (double)NB * ITERS;
    printf("[benchmark] %.0f block-calls each\n", calls);
    printf("  scalar ref     : %.3f s  (%.2f ns/block)\n", t_ref, t_ref * 1e9 / calls);
    printf("  NEON prepared  : %.3f s  (%.2f ns/block)\n", t_neon, t_neon * 1e9 / calls);
    printf("  speedup        : %.2fx\n", t_ref / t_neon);
    printf("  (sink=%d)\n", (int)sink);

    free(blocks);
    return fails ? 1 : 0;
}
