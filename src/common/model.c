/* SPDX-License-Identifier: AGPL-3.0-or-later
 * model.c — RPI model loader
 * (c) 2026 Elyan Labs
 */

#include "rpi_format.h"
#include "rpi_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

int rpi_model_load(RPIModel *model, const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "[RPI] Cannot open %s\n", path);
        return -1;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        fprintf(stderr, "[RPI] Cannot stat %s\n", path);
        close(fd);
        return -1;
    }
    model->raw_size = st.st_size;

    /* A file shorter than the fixed header makes the memcpy below read past
     * the end of the mapping. */
    if (model->raw_size < sizeof(RPIHeader)) {
        fprintf(stderr, "[RPI] File too small: %zu bytes (need >= %zu)\n",
                model->raw_size, sizeof(RPIHeader));
        close(fd);
        return -1;
    }

    model->raw = mmap(NULL, model->raw_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    if (model->raw == MAP_FAILED) {
        fprintf(stderr, "[RPI] mmap failed for %s\n", path);
        return -1;
    }

    /* Parse header */
    memcpy(&model->hdr, model->raw, sizeof(RPIHeader));

    if (model->hdr.magic != RPI_MAGIC) {
        fprintf(stderr, "[RPI] Bad magic: 0x%08x (expected 0x%08x)\n",
                model->hdr.magic, RPI_MAGIC);
        munmap(model->raw, model->raw_size);
        return -1;
    }

    if (model->hdr.version != RPI_VERSION) {
        fprintf(stderr, "[RPI] Unsupported version: %u\n", model->hdr.version);
        munmap(model->raw, model->raw_size);
        return -1;
    }

    /* Validate that every declared section actually fits inside the mapped
     * file before we walk section base pointers off the header counts. The
     * counts are untrusted (a .rpi is an external artifact — see
     * tools/dual_brain_router.py, which scp's models onto remote hosts), so
     * without this a truncated or crafted file makes the pointers below run
     * past the end of the mapping and later cell/route/emit/embed indexing
     * reads out of bounds -> segfault (DoS) or OOB read. Accumulate in 64-bit
     * so the uint32 counts cannot overflow the running total. */
    uint64_t need = (uint64_t)sizeof(RPIHeader)
        + (uint64_t)model->hdr.n_banks       * sizeof(RPIBankDesc)
        + (uint64_t)model->hdr.n_cells       * 36
        + (uint64_t)model->hdr.n_perm_blocks * sizeof(RPIPermBlock)
        + (uint64_t)model->hdr.n_routes      * sizeof(RPIRoute)
        + (uint64_t)model->hdr.n_emits       * sizeof(RPIEmit);
    if (need > (uint64_t)model->raw_size) {
        fprintf(stderr, "[RPI] Corrupt/truncated model: sections need %llu "
                "bytes but file is %zu\n",
                (unsigned long long)need, model->raw_size);
        munmap(model->raw, model->raw_size);
        return -1;
    }

    if (model->hdr.embed_offset > 0 &&
        (uint64_t)model->hdr.embed_offset
            + (uint64_t)model->hdr.vocab_size * sizeof(uint32_t)
            > (uint64_t)model->raw_size) {
        fprintf(stderr, "[RPI] Corrupt model: embedding data out of bounds\n");
        munmap(model->raw, model->raw_size);
        return -1;
    }

    /* Point into mmap'd data */
    uint8_t *ptr = (uint8_t *)model->raw + sizeof(RPIHeader);

    model->banks = (RPIBankDesc *)ptr;
    ptr += model->hdr.n_banks * sizeof(RPIBankDesc);

    model->cells = (RPICell *)ptr;
    ptr += model->hdr.n_cells * 36;

    model->perm_blocks = (RPIPermBlock *)ptr;
    ptr += model->hdr.n_perm_blocks * sizeof(RPIPermBlock);

    model->routes = (RPIRoute *)ptr;
    ptr += model->hdr.n_routes * sizeof(RPIRoute);

    model->emits = (RPIEmit *)ptr;
    ptr += model->hdr.n_emits * sizeof(RPIEmit);

    if (model->hdr.embed_offset > 0) {
        model->embed_seeds = (uint32_t *)((uint8_t *)model->raw + model->hdr.embed_offset);
    } else {
        model->embed_seeds = NULL;
    }

    fprintf(stderr, "[RPI] Loaded: %u cells, %u perm blocks, %u routes, "
            "%u emits, vocab %u\n",
            model->hdr.n_cells, model->hdr.n_perm_blocks,
            model->hdr.n_routes, model->hdr.n_emits,
            model->hdr.vocab_size);

#if defined(__aarch64__)
    /* Build the NEON control table NOW, on the loading thread, so concurrent
     * generation streams only ever READ it. The lazy call in run_cell_perms
     * then always hits the already-built fast path. */
    rpi_neon_prepare(model->perm_blocks, model->hdr.n_perm_blocks);
#endif

    return 0;
}

void rpi_model_free(RPIModel *model) {
#if defined(__aarch64__)
    /* Drop any NEON control table keyed on this model's block array before the
     * backing memory goes away (prevents a stale-pointer ABA match on reload). */
    rpi_neon_reset();
#endif
    if (model->raw && model->raw != MAP_FAILED) {
        munmap(model->raw, model->raw_size);
    }
    memset(model, 0, sizeof(*model));
}
