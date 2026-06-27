#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
distill_to_rpi.py — Distill TinyLlama 1.1B into RPI format
Resonant Permutation Inference model compiler

Pipeline:
  1. Run teacher on WikiText-2, collect hidden states at layer quartiles
  2. Cluster hidden states into cells (128 per bank, 4 banks = 512 cells)
  3. Fit ternary permutation blocks per cell
  4. Learn transition routes from teacher dynamics
  5. Learn token emission tables from teacher logits
  6. Write .rpi binary

(c) 2026 Elyan Labs — Claude Opus + GPT-5.4
"""

import struct
import os
import sys
import json
import numpy as np
from collections import defaultdict

# ── Constants matching rpi_format.h ──────────────────────────
RPI_MAGIC = 0x21495052
RPI_VERSION = 1
RPI_LANES = 64
N_BANKS = 4
CELLS_PER_BANK = 128
N_CELLS = N_BANKS * CELLS_PER_BANK  # 512
PERM_BLOCKS_PER_CELL = 2
MAX_ROUTES = 16
MAX_EMITS = 8
MAX_ROUNDS = 8
MAX_ACTIVE = 32

# Layer quartile indices for TinyLlama (22 layers)
QUARTILE_LAYERS = [0, 5, 11, 16, 21]
BANK_NAMES = ["LEX", "SYN", "DISC", "MEM"]


def collect_teacher_states(model, tokenizer, texts, seq_len=128, batch_size=4):
    """Run teacher and collect hidden states at quartile layers."""
    import torch

    print("  Collecting teacher hidden states...")
    all_states = {q: [] for q in range(N_BANKS)}  # 4 banks
    all_logits = []
    all_tokens = []

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for i in range(0, min(200, len(texts)), batch_size):
        batch = texts[i:i+batch_size]
        tok = tokenizer(batch, return_tensors='pt', truncation=True,
                        max_length=seq_len, padding='max_length')
        ids = tok.input_ids.cuda()

        with torch.no_grad():
            out = model(ids, output_hidden_states=True)

        # Map quartile layers to banks
        # Bank 0 (LEX): layers 0-5 (early) → use layer 0 output
        # Bank 1 (SYN): layers 5-11 (mid) → use layer 5 output
        # Bank 2 (DISC): layers 11-16 (late) → use layer 11 output
        # Bank 3 (MEM): layers 16-21 (final) → use layer 16 output
        for b, layer_idx in enumerate(QUARTILE_LAYERS[:N_BANKS]):
            h = out.hidden_states[layer_idx + 1].cpu().float().numpy()
            # Reshape to [n_tokens, dim] and take mean over seq dimension
            all_states[b].append(h.reshape(-1, h.shape[-1]))

        # Collect logits for emission learning
        logits = out.logits.cpu().float().numpy()
        all_logits.append(logits.reshape(-1, logits.shape[-1]))
        all_tokens.append(ids.cpu().numpy().reshape(-1))

        del out
        torch.cuda.empty_cache()

        if (i // batch_size + 1) % 10 == 0:
            print(f"    Batch {i // batch_size + 1}/{200 // batch_size}")

    # Concatenate
    for b in range(N_BANKS):
        all_states[b] = np.concatenate(all_states[b], axis=0)
        print(f"    Bank {BANK_NAMES[b]}: {all_states[b].shape}")

    all_logits = np.concatenate(all_logits, axis=0)
    all_tokens = np.concatenate(all_tokens, axis=0)

    return all_states, all_logits, all_tokens


def cluster_into_cells(states, n_clusters=CELLS_PER_BANK):
    """K-means clustering of hidden states into cells."""
    from sklearn.cluster import MiniBatchKMeans

    print(f"    Clustering {states.shape[0]} vectors into {n_clusters} cells...")
    km = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1000,
                         random_state=42, n_init=3)
    labels = km.fit_predict(states)
    centroids = km.cluster_centers_

    return labels, centroids


def fit_perm_block(centroid, dim=2048):
    """Fit a 64-lane ternary permutation block from a centroid vector.

    Strategy: select the top-64 most important dimensions,
    create a permutation that routes them, with sign from the centroid sign.
    """
    # Select top RPI_LANES dimensions by magnitude
    abs_vals = np.abs(centroid[:min(dim, len(centroid))])
    top_indices = np.argsort(abs_vals)[-RPI_LANES:]

    src_idx = np.full(RPI_LANES, 0xFF, dtype=np.uint8)  # default: zero
    sign_bits = 0

    for lane, src in enumerate(top_indices):
        src_idx[lane] = src % RPI_LANES  # map to lane space
        if centroid[src] < 0:
            sign_bits |= (1 << lane)  # negative sign

    # Sparsify: ~33% zeros (matching ternary sparsity)
    threshold_idx = int(RPI_LANES * 0.33)
    weakest = np.argsort(abs_vals[top_indices])[:threshold_idx]
    for w in weakest:
        src_idx[w] = 0xFF  # zero out weak connections

    return src_idx, sign_bits


def learn_routes(labels_per_bank, n_cells_per_bank=CELLS_PER_BANK):
    """Learn transition routes between cells from sequential activation patterns."""
    routes = defaultdict(lambda: defaultdict(int))

    for bank_labels in labels_per_bank:
        for i in range(len(bank_labels) - 1):
            src = bank_labels[i]
            dst = bank_labels[i + 1]
            if src != dst:
                routes[src][dst] += 1

    # Keep top MAX_ROUTES per cell
    cell_routes = {}
    for src, dsts in routes.items():
        sorted_dsts = sorted(dsts.items(), key=lambda x: -x[1])[:MAX_ROUTES]
        cell_routes[src] = sorted_dsts

    return cell_routes


def learn_emits(labels, tokens, logits, vocab_size, n_cells=CELLS_PER_BANK):
    """Learn which tokens each cell emits from teacher logits."""
    cell_token_scores = defaultdict(lambda: defaultdict(float))

    for i in range(min(len(labels), len(logits))):
        cell = labels[i]
        # Top tokens from teacher logits at this position
        top_k = min(32, logits.shape[1])
        top_tokens = np.argpartition(logits[i], -top_k)[-top_k:]
        for t in top_tokens:
            cell_token_scores[cell][t] += logits[i][t]

    # Keep top MAX_EMITS per cell
    cell_emits = {}
    for cell, scores in cell_token_scores.items():
        sorted_tokens = sorted(scores.items(), key=lambda x: -x[1])[:MAX_EMITS]
        cell_emits[cell] = sorted_tokens

    return cell_emits


def learn_embeddings(labels, tokens, n_cells=CELLS_PER_BANK):
    """Learn token → seed cell mapping."""
    token_cell_counts = defaultdict(lambda: defaultdict(int))

    for i in range(len(labels)):
        token_cell_counts[tokens[i]][labels[i]] += 1

    # Most common cell per token
    embed_map = {}
    for tok, cells in token_cell_counts.items():
        embed_map[tok] = max(cells, key=cells.get)

    return embed_map


def write_rpi(path, cells_data, perm_blocks_data, routes_data, emits_data,
              embed_map, vocab_size):
    """Write the .rpi binary file."""
    n_cells = len(cells_data)
    n_perm_blocks = sum(c['n_perms'] for c in cells_data)
    n_routes = sum(len(r) for r in routes_data.values())
    n_emits = sum(len(e) for e in emits_data.values())

    print(f"  Writing {path}:")
    print(f"    {n_cells} cells, {n_perm_blocks} perm blocks")
    print(f"    {n_routes} routes, {n_emits} emits")
    print(f"    vocab: {vocab_size}")

    with open(path, 'wb') as f:
        # Calculate embed offset
        embed_offset = (128 +                                   # header
                        N_BANKS * 32 +                           # banks
                        n_cells * 28 +                           # cells (packed)
                        n_perm_blocks * (RPI_LANES + 8) +        # perm blocks
                        n_routes * 8 +                           # routes
                        n_emits * 8)                             # emits

        # Header (128 bytes)
        hdr = struct.pack('<IIIIIIIIIIII I 76s',
            RPI_MAGIC, RPI_VERSION,
            n_cells, n_perm_blocks, n_routes, n_emits,
            vocab_size, RPI_LANES, MAX_ROUNDS, MAX_ACTIVE, N_BANKS,
            embed_offset, 0, b'\x00' * 76)
        f.write(hdr)

        # Bank descriptors
        for b in range(N_BANKS):
            start = b * CELLS_PER_BANK
            count = min(CELLS_PER_BANK, n_cells - start)
            bank = struct.pack('<IIIIII 8s',
                b, start, count, b, b * 64, 0xFF, b'\x00' * 8)
            f.write(bank)

        # Cells
        perm_idx = 0
        route_idx = 0
        emit_idx = 0
        for c in cells_data:
            n_r = len(routes_data.get(c['id'], []))
            n_e = len(emits_data.get(c['id'], []))
            cell = struct.pack('<IIIIIIII HH',
                c['id'], c['bank'], perm_idx, c['n_perms'],
                route_idx, n_r, emit_idx, n_e,
                0x0F, c['id'] % 16)
            f.write(cell)
            perm_idx += c['n_perms']
            route_idx += n_r
            emit_idx += n_e

        # Perm blocks
        for c in cells_data:
            for pb in c['perm_blocks']:
                f.write(bytes(pb['src_idx']))
                f.write(struct.pack('<Q', pb['sign_bits']))

        # Routes
        for c in cells_data:
            for dst, weight in routes_data.get(c['id'], []):
                f.write(struct.pack('<I HBB', dst, 0, 0, min(255, weight)))

        # Emits
        for c in cells_data:
            for tok, score in emits_data.get(c['id'], []):
                rank_bias = min(65535, max(0, int(score)))
                f.write(struct.pack('<I HBB', tok, rank_bias, 0x0F, 0))

        # Embeddings
        for t in range(vocab_size):
            seed = embed_map.get(t, 0)
            f.write(struct.pack('<I', seed))

    size = os.path.getsize(path)
    print(f"    Size: {size:,} bytes ({size/1024/1024:.1f} MB)")


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print("=" * 60)
    print("  RPI Distillation — TinyLlama → Resonant Permutation")
    print("  (c) 2026 Elyan Labs")
    print("=" * 60)

    # Load teacher
    print("\n[1/6] Loading TinyLlama...")
    tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    model = AutoModelForCausalLM.from_pretrained(
        'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        dtype=torch.float16).cuda().eval()

    # Load data
    print("\n[2/6] Loading WikiText-2...")
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    texts = [t for t in ds['text'] if len(t.strip()) > 100]

    vocab_size = tokenizer.vocab_size
    print(f"  Vocab: {vocab_size}")

    # Collect hidden states
    print("\n[3/6] Collecting teacher hidden states...")
    states, logits, tokens = collect_teacher_states(model, tokenizer, texts)
    del model
    torch.cuda.empty_cache()

    # Cluster into cells
    print("\n[4/6] Clustering into cells...")
    try:
        from sklearn.cluster import MiniBatchKMeans
    except ImportError:
        print("  Installing scikit-learn...")
        os.system("pip3 install --break-system-packages scikit-learn")
        from sklearn.cluster import MiniBatchKMeans

    labels_per_bank = {}
    centroids_per_bank = {}
    for b in range(N_BANKS):
        print(f"  Bank {BANK_NAMES[b]}:")
        # Reduce dimensionality for clustering (2048 → 64 via PCA-like projection)
        bank_states = states[b]
        # Simple: take every 32nd dimension (2048/64 = 32)
        reduced = bank_states[:, ::32][:, :RPI_LANES]
        labels, centroids = cluster_into_cells(reduced, CELLS_PER_BANK)
        # Offset cell IDs by bank
        labels_per_bank[b] = labels + b * CELLS_PER_BANK
        centroids_per_bank[b] = centroids

    # Fit permutation blocks
    print("\n[5/6] Fitting permutation blocks + routes + emits...")
    cells_data = []
    all_routes = {}
    all_emits = {}
    all_labels = np.concatenate([labels_per_bank[b] for b in range(N_BANKS)])

    for b in range(N_BANKS):
        for c in range(CELLS_PER_BANK):
            cell_id = b * CELLS_PER_BANK + c
            centroid = centroids_per_bank[b][c]

            # Fit 2 perm blocks per cell
            perms = []
            for p in range(PERM_BLOCKS_PER_CELL):
                # Vary the permutation slightly for each block
                shifted = np.roll(centroid, p * 3)
                src_idx, sign_bits = fit_perm_block(shifted, RPI_LANES)
                perms.append({'src_idx': src_idx, 'sign_bits': sign_bits})

            cells_data.append({
                'id': cell_id,
                'bank': b,
                'n_perms': PERM_BLOCKS_PER_CELL,
                'perm_blocks': perms,
            })

    # Learn routes from sequential patterns
    routes_per_bank = learn_routes(
        [labels_per_bank[b] for b in range(N_BANKS)])
    # Merge into global routes with bank offsets
    for src, dsts in routes_per_bank.items():
        all_routes[src] = dsts

    # Learn emits from teacher logits (use bank 3 / MEM labels)
    emits = learn_emits(labels_per_bank[3], tokens,
                        logits, vocab_size)
    for cell, toks in emits.items():
        all_emits[cell] = toks

    # Learn embeddings
    embed_map = learn_embeddings(labels_per_bank[0], tokens)

    # Write .rpi file
    print("\n[6/6] Writing RPI model...")
    output_path = os.path.join(os.path.dirname(__file__), '..', 'tinyllama.rpi')
    write_rpi(output_path, cells_data, {}, all_routes, all_emits,
              embed_map, vocab_size)

    print("\n" + "=" * 60)
    print("  DISTILLATION COMPLETE")
    print(f"  Model: {output_path}")
    print(f"  Test: ./rpi-cli -m tinyllama.rpi -p 'hello' -n 50")
    print("=" * 60)


if __name__ == "__main__":
    main()
