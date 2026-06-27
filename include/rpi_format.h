/* SPDX-License-Identifier: AGPL-3.0-or-later
 * rpi_format.h — Resonant Permutation Inference model format
 *
 * .rpi file layout:
 *   RPIHeader (128 bytes)
 *   RPIBankDesc[n_banks] (4 banks: LEX, SYN, DISC, MEM)
 *   RPICell[n_cells]
 *   RPIPermBlock[n_perm_blocks]  (64-lane ternary permute micro-ops)
 *   RPIRoute[n_routes]          (sparse transition table)
 *   RPIEmit[n_emits]            (token emission table)
 *   uint8_t embedding_data[]    (token → cell seed mappings)
 *
 * (c) 2026 Elyan Labs — Claude Opus + GPT-5.4 collaboration
 */

#ifndef RPI_FORMAT_H
#define RPI_FORMAT_H

#include <stdint.h>
#include <stddef.h>

#define RPI_MAGIC       0x21495052  /* "RPI!" little-endian */
#define RPI_VERSION     1
#define RPI_MAX_BANKS   4
#define RPI_LANES       64          /* permutation width */
#define RPI_MAX_ROUTES  16          /* max destinations per cell */
#define RPI_MAX_EMITS   8           /* max token emissions per cell */

/* Bank IDs */
#define RPI_BANK_LEX    0   /* Vocabulary / early layers */
#define RPI_BANK_SYN    1   /* Syntax / mid layers */
#define RPI_BANK_DISC   2   /* Discourse / late layers */
#define RPI_BANK_MEM    3   /* Long-term memory / context */

/* ── File Header ────────────────────────────────────────── */
typedef struct {
    uint32_t magic;             /* RPI_MAGIC */
    uint32_t version;           /* RPI_VERSION */
    uint32_t n_cells;           /* total cells across all banks */
    uint32_t n_perm_blocks;     /* total permutation blocks */
    uint32_t n_routes;          /* total route entries */
    uint32_t n_emits;           /* total emit entries */
    uint32_t vocab_size;        /* number of tokens */
    uint32_t lane_width;        /* RPI_LANES (64) */
    uint32_t max_rounds;        /* max recurrence rounds per token */
    uint32_t max_active;        /* max simultaneously active cells */
    uint32_t n_banks;           /* RPI_MAX_BANKS (4) */
    uint32_t embed_offset;      /* byte offset to embedding data */
    uint32_t flags;             /* bit 0: big-endian, bit 1: has NUMA hints */
    uint8_t  reserved[76];      /* pad to 128 bytes */
} RPIHeader;

/* ── Bank Descriptor ────────────────────────────────────── */
typedef struct {
    uint32_t bank_id;           /* RPI_BANK_LEX/SYN/DISC/MEM */
    uint32_t cell_start;        /* first cell index in this bank */
    uint32_t cell_count;        /* number of cells */
    uint32_t numa_hint;         /* preferred NUMA node (-1 = any) */
    uint32_t cache_color_start; /* preferred cache set offset */
    uint32_t page_color_mask;   /* page coloring mask for placement */
    uint8_t  reserved[8];
} RPIBankDesc;

/* ── Permutation Block (64-lane ternary micro-op) ───────── */
typedef struct {
    /*
     * Each block encodes a 64→64 ternary permutation:
     *   For each output lane i:
     *     src_idx[i]  = which input lane to read (0-63, or 0xFF = zero)
     *     sign_bits   = bit i: 0 = add, 1 = subtract
     *
     * Execution: out[i] = sign(i) * in[src_idx[i]]
     * This is ONE vec_perm + conditional negate. Zero multiplies.
     */
    uint8_t  src_idx[RPI_LANES];    /* source lane indices */
    uint64_t sign_bits;              /* bit per lane: 0=+, 1=- */
} RPIPermBlock;

/* ── Cell ───────────────────────────────────────────────── */
typedef struct {
    uint32_t cell_id;
    uint32_t bank_id;           /* which bank this cell belongs to */
    uint32_t perm_block_start;  /* first RPIPermBlock index */
    uint32_t perm_block_count;  /* how many sequential perm blocks */
    uint32_t route_start;       /* first RPIRoute index */
    uint32_t route_count;       /* number of outgoing routes */
    uint32_t emit_start;        /* first RPIEmit index */
    uint32_t emit_count;        /* number of token emissions */
    uint16_t phase_mask;        /* which phases this cell is active in */
    uint16_t priority;          /* scheduling priority (higher = earlier) */
} RPICell;

/* ── Route (sparse transition) ──────────────────────────── */
typedef struct {
    uint32_t dst_cell_id;       /* destination cell */
    uint16_t token_class;       /* token class that triggers this route */
    uint8_t  boundary_type;     /* 0=none, 1=word, 2=clause, 3=sentence */
    uint8_t  weight;            /* route strength (0-255) */
} RPIRoute;

/* ── Emit (token emission) ──────────────────────────────── */
typedef struct {
    uint32_t token_id;          /* vocabulary token */
    uint16_t rank_bias;         /* base score for this emission */
    uint8_t  phase_mask;        /* which phases allow this emission */
    uint8_t  reserved;
} RPIEmit;

/* ── Model (loaded in memory) ───────────────────────────── */
typedef struct {
    RPIHeader       hdr;
    RPIBankDesc    *banks;
    RPICell        *cells;
    RPIPermBlock   *perm_blocks;
    RPIRoute       *routes;
    RPIEmit        *emits;
    uint32_t       *embed_seeds;    /* token_id → seed cell mapping */
    void           *raw;            /* mmap'd file */
    size_t          raw_size;
} RPIModel;

#endif /* RPI_FORMAT_H */
