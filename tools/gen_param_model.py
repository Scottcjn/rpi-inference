#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Parametrized .rpi generator for controlled benchmarks.

Usage: gen_param_model.py <out.rpi> <n_cells> <vocab_size> [perm_per_cell=2]

Varies ONE structural axis at a time so the NEON-vs-scalar speedup can be
attributed to working-set residency vs per-token vocab-scoring (Amdahl).
Same structure/seed as gen_test_model.py; corrected embed offset (cell=36B).
"""
import struct, random, os, sys

RPI_MAGIC = 0x21495052
RPI_VERSION = 1
RPI_LANES = 64
N_BANKS = 4
N_ROUTES_PER = 4
N_EMITS_PER = 4
CELL_SIZE = 36           # 8xu32 + 2xu16 (packed '<', no padding)
PERM_SIZE = RPI_LANES + 8

def write_rpi(path, n_cells, vocab, perm_per_cell):
    n_perm = n_cells * perm_per_cell
    n_routes = n_cells * N_ROUTES_PER
    n_emits = n_cells * N_EMITS_PER
    cells_per_bank = max(1, n_cells // N_BANKS)

    with open(path, 'wb') as f:
        embed_offset = (128 + N_BANKS * 32 + n_cells * CELL_SIZE +
                        n_perm * PERM_SIZE + n_routes * 8 + n_emits * 8)
        f.write(struct.pack('<IIIIIIIIIIII I 76s',
            RPI_MAGIC, RPI_VERSION, n_cells, n_perm, n_routes, n_emits,
            vocab, RPI_LANES, 8, 32, N_BANKS, embed_offset, 0, b'\x00' * 76))

        for b in range(N_BANKS):
            f.write(struct.pack('<IIIIII 8s', b, b * cells_per_bank,
                cells_per_bank, b, b * 64, 0xFF, b'\x00' * 8))

        perm_idx = route_idx = emit_idx = 0
        for c in range(n_cells):
            f.write(struct.pack('<IIIIIIII HH', c, min(c // cells_per_bank, N_BANKS - 1),
                perm_idx, perm_per_cell, route_idx, N_ROUTES_PER,
                emit_idx, N_EMITS_PER, 0x0F, c % 16))
            perm_idx += perm_per_cell
            route_idx += N_ROUTES_PER
            emit_idx += N_EMITS_PER

        random.seed(42)
        for _ in range(n_perm):
            src = bytearray(RPI_LANES)
            for i in range(RPI_LANES):
                if random.random() < 0.33:
                    src[i] = 0xFF
                else:
                    src[i] = (i + random.randint(-8, 8)) % RPI_LANES
            f.write(bytes(src))
            f.write(struct.pack('<Q', random.getrandbits(RPI_LANES)))

        for c in range(n_cells):
            for r in range(N_ROUTES_PER):
                f.write(struct.pack('<I HBB', (c + r + 1) % n_cells, r, 0,
                    128 + random.randint(0, 127)))

        for c in range(n_cells):
            for e in range(N_EMITS_PER):
                f.write(struct.pack('<I HBB', (c * 2 + e) % vocab, 100 + c % 50, 0x0F, 0))

        for t in range(vocab):
            f.write(struct.pack('<I', t % n_cells))

    sz = os.path.getsize(path)
    print(f"{path}: {sz/1024:.0f} KB | {n_cells} cells, {n_perm} perm, vocab {vocab}")

if __name__ == "__main__":
    out = sys.argv[1]
    n_cells = int(sys.argv[2])
    vocab = int(sys.argv[3])
    ppc = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    write_rpi(out, n_cells, vocab, ppc)
