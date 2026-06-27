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
    fstat(fd, &st);
    model->raw_size = st.st_size;

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

    return 0;
}

void rpi_model_free(RPIModel *model) {
    if (model->raw && model->raw != MAP_FAILED) {
        munmap(model->raw, model->raw_size);
    }
    memset(model, 0, sizeof(*model));
}
