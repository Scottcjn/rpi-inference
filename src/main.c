/* SPDX-License-Identifier: AGPL-3.0-or-later
 * main.c — RPI inference CLI (the "llama-cli" of permutation inference)
 *
 * Usage:
 *   rpi-cli -m model.rpi -p "Hello world" -n 50
 *
 * (c) 2026 Elyan Labs — Resonant Permutation Inference
 */

#include "rpi_format.h"
#include "rpi_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <pthread.h>

/* One independent generation stream (its own RPIState + output buffer; the
 * model and NEON control table are shared read-only). */
typedef struct {
    const RPIModel *model;
    const RPIHWProfile *hw;
    const uint32_t *prompt_ids;
    uint32_t prompt_len;
    uint32_t max_tokens;
    RPIState *st;           /* heap: RPIState is ~280 KB, too big for a
                               default pthread stack */
    uint32_t *output_ids;
    uint32_t n_output;
} StreamJob;

static void *stream_worker(void *arg) {
    StreamJob *j = (StreamJob *)arg;
    rpi_generate(j->model, j->hw, j->st, j->prompt_ids, j->prompt_len,
                 j->max_tokens, j->output_ids, &j->n_output);
    return NULL;
}

static void print_banner(void) {
    fprintf(stderr, "\n");
    fprintf(stderr, "  ╦═╗╔═╗╦\n");
    fprintf(stderr, "  ╠╦╝╠═╝║   Resonant Permutation Inference\n");
    fprintf(stderr, "  ╩╚═╩  ╩   Zero-multiply. Cache-resonant. Hardware-timed.\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "  (c) 2026 Elyan Labs — Claude Opus + GPT-5.4\n");
    fprintf(stderr, "  \"Inference is reordering, not arithmetic.\"\n");
    fprintf(stderr, "\n");
}

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s [options]\n", prog);
    fprintf(stderr, "  -m <path>     Model file (.rpi)\n");
    fprintf(stderr, "  -p <text>     Prompt text\n");
    fprintf(stderr, "  -n <int>      Max tokens to generate (default: 64)\n");
    fprintf(stderr, "  -j <int>      Parallel independent streams (default: 1;\n");
    fprintf(stderr, "                each generates -n tokens; aggregate tok/s reported.\n");
    fprintf(stderr, "                RPI_PROFILE is single-stream only.)\n");
    fprintf(stderr, "  -v            Verbose (show timing per token)\n");
    fprintf(stderr, "  -h            Help\n");
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *prompt = NULL;
    uint32_t max_tokens = 64;
    int verbose = 0;

    int opt;
    int n_streams = 1;
    while ((opt = getopt(argc, argv, "m:p:n:j:vh")) != -1) {
        switch (opt) {
            case 'm': model_path = optarg; break;
            case 'p': prompt = optarg; break;
            case 'n': max_tokens = atoi(optarg); break;
            case 'j': n_streams = atoi(optarg); break;
            case 'v': verbose = 1; break;
            case 'h': usage(argv[0]); return 0;
            default:  usage(argv[0]); return 1;
        }
    }

    if (!model_path) {
        fprintf(stderr, "Error: -m model.rpi required\n");
        usage(argv[0]);
        return 1;
    }

    print_banner();

    /* Detect hardware */
    RPIHWProfile hw;
    rpi_hw_detect(&hw);

    /* Load model */
    RPIModel model;
    fprintf(stderr, "[RPI] Loading %s...\n", model_path);
    if (rpi_model_load(&model, model_path) != 0) {
        return 1;
    }

    /* Initialize state */
    RPIState st;
    rpi_state_init(&st);

    /* Tokenize prompt (simple: each byte is a token for now) */
    /* TODO: proper tokenizer integration */
    uint32_t prompt_ids[4096];
    uint32_t prompt_len = 0;

    if (prompt) {
        fprintf(stderr, "[RPI] Prompt: \"%s\"\n", prompt);
        for (const char *p = prompt; *p && prompt_len < 4096; p++) {
            prompt_ids[prompt_len++] = (uint32_t)(uint8_t)*p;
        }
    }

    if (max_tokens == 0) max_tokens = 1;

    /* ── Multi-stream mode (-j N): N independent generation streams ──
     * The model + NEON control table are shared read-only (the table is built
     * at load time, before any thread exists). Each stream gets its own heap
     * RPIState and output buffer. A 1-token warmup runs on the main thread
     * first so every lazily-resolved toggle static is settled before threads
     * spawn. */
    if (n_streams > 1) {
        if (n_streams > 64) n_streams = 64;

        /* Warmup: resolve lazy statics single-threaded. */
        {
            RPIState *wst = (RPIState *)calloc(1, sizeof(RPIState));
            uint32_t wtok, wn = 0;
            if (wst) {
                rpi_state_init(wst);
                rpi_generate(&model, &hw, wst, prompt_ids, prompt_len, 1, &wtok, &wn);
                free(wst);
            }
        }

        StreamJob *jobs = (StreamJob *)calloc((size_t)n_streams, sizeof(StreamJob));
        pthread_t *tids = (pthread_t *)calloc((size_t)n_streams, sizeof(pthread_t));
        int alloc_fail = (!jobs || !tids);
        for (int s = 0; !alloc_fail && s < n_streams; s++) {
            jobs[s].model = &model;
            jobs[s].hw = &hw;
            jobs[s].prompt_ids = prompt_ids;
            jobs[s].prompt_len = prompt_len;
            jobs[s].max_tokens = max_tokens;
            jobs[s].st = (RPIState *)calloc(1, sizeof(RPIState));
            jobs[s].output_ids = (uint32_t *)malloc((size_t)max_tokens * sizeof(uint32_t));
            if (!jobs[s].st || !jobs[s].output_ids) { alloc_fail = 1; break; }
            rpi_state_init(jobs[s].st);
        }
        if (alloc_fail) {
            fprintf(stderr, "Error: allocation failed for %d streams\n", n_streams);
            rpi_model_free(&model);
            return 1;
        }

        fprintf(stderr, "[RPI] Generating %u tokens x %d streams...\n\n",
                max_tokens, n_streams);

        uint64_t mt0 = rpi_tb_now();
        for (int s = 0; s < n_streams; s++)
            pthread_create(&tids[s], NULL, stream_worker, &jobs[s]);
        for (int s = 0; s < n_streams; s++)
            pthread_join(tids[s], NULL);
        uint64_t mt1 = rpi_tb_now();

        double el = (double)(mt1 - mt0) / (double)hw.tb_freq;
        uint64_t total_tok = 0;
        for (int s = 0; s < n_streams; s++) total_tok += jobs[s].n_output;

        /* stdout: stream 0's tokens (keeps the single-stream contract). */
        for (uint32_t i = 0; i < jobs[0].n_output; i++)
            printf("%u%s", jobs[0].output_ids[i],
                   (i + 1 < jobs[0].n_output) ? " " : "");
        printf("\n");

        fprintf(stderr, "\n[RPI] %d streams: %llu total tokens in %.3f s\n",
                n_streams, (unsigned long long)total_tok, el);
        fprintf(stderr, "[RPI] Aggregate: %.1f tok/s  (%.1f tok/s per stream)\n",
                (el > 0) ? total_tok / el : 0,
                (el > 0) ? total_tok / el / n_streams : 0);

        for (int s = 0; s < n_streams; s++) { free(jobs[s].st); free(jobs[s].output_ids); }
        free(jobs); free(tids);
        rpi_model_free(&model);
        return 0;
    }

    /* Generate. Heap-allocate sized to the request so large -n cannot overflow
     * a fixed stack buffer (the old uint32_t output_ids[8192] crashed for
     * -n > 8192). */
    uint32_t *output_ids = (uint32_t *)malloc((size_t)max_tokens * sizeof(uint32_t));
    if (!output_ids) {
        fprintf(stderr, "Error: cannot allocate output buffer for %u tokens\n", max_tokens);
        rpi_model_free(&model);
        return 1;
    }
    uint32_t n_output = 0;

    fprintf(stderr, "[RPI] Generating %u tokens...\n\n", max_tokens);

    uint64_t t_start = rpi_tb_now();

    rpi_generate(&model, &hw, &st, prompt_ids, prompt_len,
                 max_tokens, output_ids, &n_output);

    uint64_t t_end = rpi_tb_now();
    double elapsed_s = (double)(t_end - t_start) / (double)hw.tb_freq;
    double tok_per_s = (elapsed_s > 0) ? (double)n_output / elapsed_s : 0;

    /* Output token IDs as space-separated integers */
    for (uint32_t i = 0; i < n_output; i++) {
        printf("%u", output_ids[i]);
        if (i < n_output - 1) printf(" ");
    }
    printf("\n");

    /* Stats */
    fprintf(stderr, "\n[RPI] Generated %u tokens in %.3f s (%.1f tok/s)\n",
            n_output, elapsed_s, tok_per_s);
    fprintf(stderr, "[RPI] Avg rounds/token: %.1f\n",
            (double)st.round / (double)(n_output > 0 ? n_output : 1));
    fprintf(stderr, "[RPI] Active cells: %u / %u max\n",
            st.n_active, model.hdr.max_active);

    free(output_ids);
    rpi_model_free(&model);
    return 0;
}
