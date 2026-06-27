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
    fprintf(stderr, "  -v            Verbose (show timing per token)\n");
    fprintf(stderr, "  -h            Help\n");
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *prompt = NULL;
    uint32_t max_tokens = 64;
    int verbose = 0;

    int opt;
    while ((opt = getopt(argc, argv, "m:p:n:vh")) != -1) {
        switch (opt) {
            case 'm': model_path = optarg; break;
            case 'p': prompt = optarg; break;
            case 'n': max_tokens = atoi(optarg); break;
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

    /* Generate */
    uint32_t output_ids[8192];
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

    rpi_model_free(&model);
    return 0;
}
