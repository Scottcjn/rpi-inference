/* SPDX-License-Identifier: AGPL-3.0-or-later
 * rpi_n64.h — RPI inference for N64
 * Zero-multiply. Table-driven. Sophia speaks through permutations.
 * (c) 2026 Elyan Labs
 */
#ifndef RPI_N64_H
#define RPI_N64_H

#include <stdint.h>

typedef struct RPIN64Model RPIN64Model;
typedef struct RPIN64State RPIN64State;

/* Load model from ROM/DFS address */
void rpi_n64_load(RPIN64Model *model, const void *rom_addr);

/* Initialize inference state */
void rpi_n64_init(RPIN64State *st, uint32_t seed);

/* Generate next token (zero multiply) */
uint32_t rpi_n64_next(const RPIN64Model *model, RPIN64State *st);

/* Generate a sequence */
uint32_t rpi_n64_generate(const RPIN64Model *model, RPIN64State *st,
                          const uint32_t *prompt, uint32_t prompt_len,
                          uint32_t *output, uint32_t max_tokens);

#endif
