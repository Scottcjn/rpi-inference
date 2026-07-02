/* SPDX-License-Identifier: MIT
 * rpi_speculative.c — RPI n-gram drafting inside a real llama.cpp verify loop.
 *
 * True speculative decoding with logit teacher-forcing: the RPI drafter
 * (context-copy / prompt-lookup — the matmul-free permutation move) proposes
 * k tokens, the verifier scores ALL of them in ONE llama_decode batch, the
 * longest argmax-matching prefix is accepted, the verifier's own next token
 * is taken as a bonus, and the KV cache is rolled back past the rejection
 * point. Greedy (temperature 0) verification, so accepted output equals
 * what the verifier alone would produce — the draft can only change speed,
 * never content. One honest caveat, shared with llama.cpp's own speculative
 * mode: batched and single-token decodes use different kernel/reduction
 * orders, so logits can differ by ULPs and a near-tie argmax can flip
 * (observed once ~90 tokens into a JSON prompt on Metal; both continuations
 * coherent, the flipped token was the verifier's own pick, not a draft).
 * Byte-identity holds wherever the backend is batch-invariant.
 *
 * Baseline is the SAME binary with --no-draft (plain autoregressive), so the
 * A/B is one flag on one code path.
 *
 * BACKEND-AGNOSTIC: this file contains no Metal/CUDA/HIP code — the GPU
 * backend is whatever the libllama you link was built with. Build once per
 * backend (or use `make rpi-spec LLAMA_BUILD=...`):
 *   cc -O2 -o rpi-spec tools/rpi_speculative.c \
 *      -I $LLAMA/include -I $LLAMA/ggml/include \
 *      -L $LLAMA/<build-dir>/bin -lllama -Wl,-rpath,$LLAMA/<build-dir>/bin
 * where <build-dir> is a Metal, CUDA, Vulkan, or HIP/ROCm build of llama.cpp.
 * On multi-GPU/Vulkan boxes select the device with GGML_VK_VISIBLE_DEVICES.
 *
 * Measured (log-structured prompt / prose prompt, wall-clock vs --no-draft,
 * identity verified per backend):
 *   Metal  Apple M2            1.54x / 0.93x   (TinyLlama-1.1B)
 *   CUDA   RTX 4070 Laptop     2.34x / 1.01x   (TinyLlama-1.1B)
 *   CUDA   RTX 4070 Laptop     2.71x / 1.01x   (Qwen2.5-7B — bigger verifier,
 *                                               bigger win, as the economics say)
 *   Vulkan AMD Radeon 780M     1.95x / 1.04x   (TinyLlama-1.1B)
 *   CUDA   Tesla M40 (2015!)   1.89x / 0.91x   (TinyLlama-1.1B; 7B loses on
 *                                               Maxwell — batch too costly)
 *   CUDA   Tesla V100-SXM2     3.63x / 0.93x   (TinyLlama-1.1B, 746 tok/s)
 *   CUDA   Tesla V100-SXM2     1.98x / 0.91x   (Qwen2.5-7B, card shared with
 *                                               a live server)
 * Backend-ratio check (same RTX 4070, two backends): CUDA 2.34x vs Vulkan
 * 1.92x with near-identical baselines — the ratio roughly holds and the
 * native backend extracts ~20% more, so a native HIP build should beat the
 * Vulkan numbers above on the same AMD silicon. ROCm build of llama.cpp:
 *   cmake -B build-hip -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx90a  # MI250
 *   (gfx942 for MI300X); then make rpi-spec LLAMA_BUILD=build-hip
 * NOTE: CUDA enumerates devices fastest-first by default, so nvidia-smi index
 * != CUDA index on mixed boxes (CUDA_VISIBLE_DEVICES=1 picked the M40 on a
 * V100+M40 machine).
 *
 * (c) 2026 Elyan Labs
 */

#include "llama.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ── RPI n-gram drafter (mirrors src/common/decode.c ngram_copy_next) ──
 * CANONICAL COPY LIVES IN src/common/decode.c (ngram_copy_next); if either
 * side changes, change both or extract a shared header — this function is the
 * measurement oracle, and drift silently invalidates the numbers.
 * Longest suffix of the token history that occurred earlier;
 * most recent match wins; the continuation is the draft token. Pure table
 * permutation — no model, no multiply. */
#define NGRAM_MAX 8
/* Confidence gate: only draft from a match of >= 3 tokens. Unigram/bigram
 * matches fire constantly on prose ("the" occurred before) and draft junk at
 * ~15% acceptance — measured as a net wall-clock LOSS (0.42-0.51x), because
 * a (k+1)-token batch decode costs more than a 1-token decode. Long matches
 * are the procedural-copy signature; short ones are noise. */
#define NGRAM_MIN 3

static llama_token ngram_next(const llama_token *hist, int T) {
    if (T < NGRAM_MIN + 1) return -1;
    int nmax = (T - 1 < NGRAM_MAX) ? T - 1 : NGRAM_MAX;
    for (int n = nmax; n >= NGRAM_MIN; n--) {
        for (int p = T - n - 1; p >= 0; p--) {
            int ok = 1;
            for (int j = 0; j < n; j++)
                if (hist[p + j] != hist[T - n + j]) { ok = 0; break; }
            if (ok) return hist[p + n];
        }
    }
    return -1;
}

/* ── helpers ─────────────────────────────────────────────── */
static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static llama_token argmax_ith(struct llama_context *ctx, int i, int n_vocab) {
    const float *lg = llama_get_logits_ith(ctx, i);
    int best = 0;
    for (int t = 1; t < n_vocab; t++)
        if (lg[t] > lg[best]) best = t;
    return best;
}

/* decode a run of tokens starting at position pos; logits on every token if
 * want_all, else only on the last. returns 0 on success. */
static int decode_run(struct llama_context *ctx, const llama_token *toks,
                      int n, int pos, int want_all) {
    llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; i++) {
        b.token[i]     = toks[i];
        b.pos[i]       = pos + i;
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = 0;
        b.logits[i]    = want_all ? 1 : (i == n - 1);
    }
    b.n_tokens = n;
    int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    return rc;
}

int main(int argc, char **argv) {
    const char *model_path = NULL, *prompt = "";
    int n_gen = 128, draft_k = 8, use_draft = 1, show_text = 0;
    int n_ctx_arg = 4096;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) n_gen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-k") && i + 1 < argc) draft_k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-c") && i + 1 < argc) n_ctx_arg = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-draft")) use_draft = 0;
        else if (!strcmp(argv[i], "-t")) show_text = 1;
        else { fprintf(stderr, "usage: %s -m model.gguf -p prompt [-n N] "
                       "[-k draft_k] [-c ctx] [--no-draft] [-t]\n", argv[0]); return 1; }
    }
    if (!model_path) { fprintf(stderr, "need -m model.gguf\n"); return 1; }
    if (draft_k < 1) draft_k = 1;
    if (draft_k > 63) draft_k = 63;   /* d[64]/run[65] capacity — -k 65 was a
                                         stack overflow (tri-brain catch) */
    if (n_ctx_arg < 512) n_ctx_arg = 512;
    if (n_ctx_arg > 32768) n_ctx_arg = 32768;
    if (n_gen < 1) n_gen = 1;
    if (n_gen > n_ctx_arg - 96) n_gen = n_ctx_arg - 96;  /* leave prompt room */

    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    struct llama_model *model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const llama_token eos = llama_vocab_eos(vocab);

    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (unsigned)n_ctx_arg; cp.n_batch = (unsigned)n_ctx_arg;
    struct llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context init failed\n"); return 1; }
    llama_memory_t mem = llama_get_memory(ctx);

    /* history buffer: prompt + generated. cap == n_ctx (cp.n_ctx below is
     * set from this) so the loop guard and the KV limit cannot drift apart. */
    const int cap = n_ctx_arg;
    llama_token *hist = malloc(sizeof(llama_token) * cap);
    if (!hist) { fprintf(stderr, "oom\n"); return 1; }
    /* saved logits row: on a rejection we keep the last ACCEPTED position's
     * logits here instead of paying a whole re-decode for them. */
    float *saved_logits = malloc(sizeof(float) * n_vocab);
    if (!saved_logits) { fprintf(stderr, "oom\n"); return 1; }
    int have_saved = 0;
    int T = llama_tokenize(vocab, prompt, (int)strlen(prompt),
                           hist, cap, /*add_special=*/1, /*parse_special=*/1);
    if (T <= 0) { fprintf(stderr, "tokenize failed (%d)\n", T); return 1; }

    /* ingest prompt, logits on last */
    if (decode_run(ctx, hist, T, 0, 0)) { fprintf(stderr, "prompt decode failed\n"); return 1; }

    /* stats */
    int generated = 0, accepted_draft = 0, fwd_calls = 1, rounds = 0;

    /* Adaptive draft window. With the n>=3 confidence gate, drafts only fire
     * on strong copy evidence, so start at full k (a k=2 warm-up throttled
     * the procedural win: LOG 1.54x -> 1.28x measured); shrink to what a
     * partial round actually accepted, regrow on full acceptance. */
    int k_cur = draft_k;

    double t0 = now_s();
    while (generated < n_gen && T + draft_k + 1 < cap) {
        rounds++;
        llama_token g0;
        if (have_saved) {                 /* logits kept from a rejected round */
            int best = 0;
            for (int t = 1; t < n_vocab; t++)
                if (saved_logits[t] > saved_logits[best]) best = t;
            g0 = best;
            have_saved = 0;
        } else {
            g0 = argmax_ith(ctx, -1, n_vocab);   /* verifier's pick */
        }

        /* build the RPI draft from history (stop at EOS / no match) */
        llama_token d[64];
        int kd = 0;
        if (use_draft) {
            while (kd < k_cur) {
                /* draft continues from history + already-drafted tokens */
                hist[T + kd] = (kd == 0) ? g0 : d[kd - 1];  /* scratch append */
                llama_token nx = ngram_next(hist, T + kd + 1);
                if (nx < 0 || nx == eos) break;
                d[kd++] = nx;
            }
        }

        /* emit the verifier's token (always correct at temp 0) */
        hist[T] = g0; T++; generated++;
        if (show_text) {
            char buf[256];
            int L = llama_token_to_piece(vocab, g0, buf, sizeof buf, 0, 1);
            if (L > 0) fwrite(buf, 1, L, stdout);
        }
        if (g0 == eos || generated >= n_gen) break;

        if (kd == 0) {
            /* nothing procedural to copy — plain autoregressive step */
            if (decode_run(ctx, &g0, 1, T - 1, 0)) break;
            fwd_calls++;
            continue;
        }

        /* ONE forward pass: g0 followed by the kd draft tokens, all logits.
         * After it, logits[i] = verifier's distribution given ...g0,d[0..i-1],
         * i.e. its greedy check for draft slot i. */
        llama_token run[65];
        run[0] = g0;
        memcpy(run + 1, d, kd * sizeof(llama_token));
        if (decode_run(ctx, run, kd + 1, T - 1, 1)) break;
        fwd_calls++;

        int a = 0;  /* accepted draft tokens */
        while (a < kd) {
            llama_token v = argmax_ith(ctx, a, n_vocab);
            if (v != d[a]) break;
            hist[T] = d[a]; T++; generated++; accepted_draft++;
            if (show_text) {
                char buf[256];
                int L = llama_token_to_piece(vocab, d[a], buf, sizeof buf, 0, 1);
                if (L > 0) fwrite(buf, 1, L, stdout);
            }
            a++;
            if (d[a - 1] == eos || generated >= n_gen) break;
        }

        /* On rejection: SAVE the last accepted position's logits (batch index
         * a — after g0,d[0..a-1]) for the next round's g0, then roll the KV
         * back past the rejected tail. Costs a 128 KB memcpy instead of a
         * whole re-decode; without this, low-acceptance prompts did more
         * forward passes than tokens (measured tokens_per_fwd 0.85 < 1). */
        if (a < kd) {
            memcpy(saved_logits, llama_get_logits_ith(ctx, a),
                   sizeof(float) * n_vocab);
            have_saved = 1;
            llama_memory_seq_rm(mem, 0, T, -1);
        }
        /* if a == kd, batch index kd is the fresh head; argmax_ith(ctx,-1)
         * at the next loop head reads exactly that row. Correct as-is. */

        /* adapt the window to observed acceptance */
        if (a == kd) {
            k_cur = (k_cur + 2 > draft_k) ? draft_k : k_cur + 2;
        } else {
            k_cur = (a > 1) ? a : 1;
        }
    }
    if (generated < n_gen && T + draft_k + 1 >= cap)
        fprintf(stderr, "[rpi-spec] warning: stopped at context cap "
                "(%d tokens generated of %d requested)\n", generated, n_gen);
    double dt = now_s() - t0;

    if (show_text) printf("\n");
    fprintf(stderr, "mode=%s k=%d tokens=%d time=%.3fs tok/s=%.1f "
            "fwd_calls=%d rounds=%d accepted_draft=%d draft_share=%.2f "
            "tokens_per_fwd=%.2f\n",
            use_draft ? "spec" : "baseline", draft_k, generated, dt,
            generated / (dt > 0 ? dt : 1), fwd_calls, rounds, accepted_draft,
            generated > 0 ? (double)accepted_draft / generated : 0,
            fwd_calls > 0 ? (double)generated / fwd_calls : 0);

    free(hist);
    free(saved_logits);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
