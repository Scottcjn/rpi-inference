#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generate a test .rpi model file for validating the inference engine."""

import struct
import random
import os

RPI_MAGIC = 0x21495052  # "RPI!"
RPI_VERSION = 1
RPI_LANES = 64
N_BANKS = 4
N_CELLS = 128       # 32 per bank
N_PERM_BLOCKS = 256  # 2 per cell
N_ROUTES_PER = 4
N_EMITS_PER = 4
VOCAB_SIZE = 256     # byte-level for testing
CELL_SIZE = struct.calcsize('<IIIIIIII HH')

def write_rpi(path):
    n_routes = N_CELLS * N_ROUTES_PER
    n_emits = N_CELLS * N_EMITS_PER

    with open(path, 'wb') as f:
        # Header (128 bytes)
        embed_offset = (128 +                           # header
                        N_BANKS * 32 +                   # banks
                        N_CELLS * CELL_SIZE +            # cells
                        N_PERM_BLOCKS * (RPI_LANES + 8) + # perm blocks
                        n_routes * 8 +                   # routes
                        n_emits * 8)                     # emits

        hdr = struct.pack('<IIIIIIIIIIII I 76s',
            RPI_MAGIC, RPI_VERSION,
            N_CELLS, N_PERM_BLOCKS, n_routes, n_emits,
            VOCAB_SIZE, RPI_LANES,
            8,      # max_rounds
            32,     # max_active
            N_BANKS,
            embed_offset,
            0,      # flags
            b'\x00' * 76)
        f.write(hdr)

        # Bank descriptors (32 bytes each)
        cells_per_bank = N_CELLS // N_BANKS
        for b in range(N_BANKS):
            bank = struct.pack('<IIIIII 8s',
                b,                          # bank_id
                b * cells_per_bank,         # cell_start
                cells_per_bank,             # cell_count
                b if b < 4 else 0xFFFFFFFF, # numa_hint
                b * 64,                     # cache_color_start
                0xFF,                       # page_color_mask
                b'\x00' * 8)
            f.write(bank)

        # Cells
        perm_idx = 0
        route_idx = 0
        emit_idx = 0
        for c in range(N_CELLS):
            cell = struct.pack('<IIIIIIII HH',
                c,                          # cell_id
                c // cells_per_bank,        # bank_id
                perm_idx,                   # perm_block_start
                2,                          # perm_block_count
                route_idx,                  # route_start
                N_ROUTES_PER,               # route_count
                emit_idx,                   # emit_start
                N_EMITS_PER,                # emit_count
                0x0F,                       # phase_mask (all phases)
                c % 16)                     # priority
            f.write(cell)
            perm_idx += 2
            route_idx += N_ROUTES_PER
            emit_idx += N_EMITS_PER

        # Perm blocks (RPI_LANES + 8 bytes each)
        random.seed(42)  # reproducible
        for _ in range(N_PERM_BLOCKS):
            # Generate structured permutation (not random — has patterns)
            src_idx = bytearray(RPI_LANES)
            for i in range(RPI_LANES):
                if random.random() < 0.33:
                    src_idx[i] = 0xFF  # zero (ternary W=0)
                else:
                    # Route from nearby lanes (local connectivity)
                    offset = random.randint(-8, 8)
                    src_idx[i] = (i + offset) % RPI_LANES

            # Sign bits: ~50% negative
            sign_bits = random.getrandbits(RPI_LANES)

            f.write(bytes(src_idx))
            f.write(struct.pack('<Q', sign_bits))

        # Routes
        for c in range(N_CELLS):
            for r in range(N_ROUTES_PER):
                dst = (c + r + 1) % N_CELLS  # route to next cells
                route = struct.pack('<I HBB',
                    dst,                    # dst_cell_id
                    r,                      # token_class
                    0,                      # boundary_type
                    128 + random.randint(0, 127))  # weight
                f.write(route)

        # Emits
        for c in range(N_CELLS):
            for e in range(N_EMITS_PER):
                # Each cell emits tokens related to its position
                tok = (c * 2 + e) % VOCAB_SIZE
                emit = struct.pack('<I HBB',
                    tok,                    # token_id
                    100 + c % 50,           # rank_bias
                    0x0F,                   # phase_mask (all)
                    0)                      # reserved
                f.write(emit)

        # Embedding seeds: token → cell mapping
        for t in range(VOCAB_SIZE):
            seed_cell = t % N_CELLS
            f.write(struct.pack('<I', seed_cell))

    size = os.path.getsize(path)
    print(f"Generated {path}: {size} bytes ({size/1024:.1f} KB)")
    print(f"  {N_CELLS} cells, {N_PERM_BLOCKS} perm blocks, "
          f"{N_CELLS * N_ROUTES_PER} routes, {N_CELLS * N_EMITS_PER} emits")
    print(f"  Vocab: {VOCAB_SIZE} (byte-level)")

if __name__ == "__main__":
    write_rpi("test_model.rpi")
