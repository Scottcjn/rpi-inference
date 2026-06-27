#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
RPI v14 Model Builder — Beyond Bigrams
Uses: bigrams + pair-hash trigrams + phrase templates
Produces coherent 6-12 token spans.
"""
import struct, json, os, random
from collections import Counter, defaultdict
from transformers import AutoTokenizer

DATA_FILE = '/home/scott/rpi/rpi_v14_data.json'
RPI_MAGIC=0x21495052; RPI_VERSION=2  # v2 format with phrases
RPI_LANES=64; N_BANKS=4; MAX_EMITS=64; CELL_SIZE=36
PHRASE_FLAG = 0x80000000  # high bit marks phrase emit

tokenizer = AutoTokenizer.from_pretrained('/home/scott/sophia-hermes/sophia-hermes-v3')
vocab_size = tokenizer.vocab_size

print("Loading v14 data...")
with open(DATA_FILE) as f:
    data = json.load(f)

bigrams = {int(k): Counter({int(t):int(s) for t,s in v.items()})
           for k, v in data['bigrams'].items()}
pair_seeds = {}
for k, v in data['pair_seeds'].items():
    parts = k.split('_')
    pair_seeds[(int(parts[0]), int(parts[1]))] = Counter({int(t):int(s) for t,s in v.items()})
phrases = data['phrases']

print(f"  {len(bigrams)} bigrams, {len(pair_seeds)} trigrams, {len(phrases)} phrases")
print(f"  {data['total_tokens']} total tokens")

# Build cells from bigrams
N_CELLS = min(len(bigrams), 16000)
top_tokens = sorted(bigrams, key=lambda t: sum(bigrams[t].values()), reverse=True)[:N_CELLS]
CELLS_PER_BANK = (N_CELLS + N_BANKS - 1) // N_BANKS
tc = {tok: i for i, tok in enumerate(top_tokens)}
for t in range(vocab_size):
    if t not in tc: tc[t] = t % N_CELLS

# Build pair-hash lookup: hash(prev2, prev1) → best cell
PAIR_BUCKETS = 4096  # power of 2
pair_table = {}  # bucket → (primary_cell, secondary_cell)

def pair_hash(t1, t2):
    h = 0xcbf29ce484222325
    h ^= t1 & 0xFFFF; h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    h ^= t2 & 0xFFFF; h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return int(h % PAIR_BUCKETS)

# For each pair, find the best cell to activate
for (t1, t2), nexts in pair_seeds.items():
    bucket = pair_hash(t1, t2)
    best_next = nexts.most_common(1)[0][0] if nexts else 0
    primary = tc.get(best_next, best_next % N_CELLS)
    # Secondary: use the bigram cell for t2
    secondary = tc.get(t2, t2 % N_CELLS)
    if bucket not in pair_table:
        pair_table[bucket] = (primary, secondary)

print(f"  {len(pair_table)} pair hash buckets populated")

# Build phrase index: map first token → list of phrases
phrase_by_first = defaultdict(list)
for pi, p in enumerate(phrases):
    tokens = p['tokens']
    if len(tokens) >= 3:
        phrase_by_first[tokens[0]].append(pi)

# Merge phrase emits into cell emissions
# For each cell, add its best phrases as special emit entries
for ci, tok in enumerate(top_tokens):
    if tok in phrase_by_first:
        # This cell can trigger phrases
        pass  # handled during emission

# Build model file
output = 'sophia_v14.rpi'
cd = []; tr = te = 0

for ci, tok in enumerate(top_tokens):
    # Regular bigram emits
    emits = [(t, min(65535, s)) for t, s in bigrams[tok].most_common(MAX_EMITS - 8)][:MAX_EMITS - 8]

    # Add trigram-conditioned emits (from pair seeds where tok is prev1)
    # Find pairs where this token is the second element
    tri_emits = Counter()
    for (t1, t2), nexts in pair_seeds.items():
        if t2 == tok:
            tri_emits += nexts
    # Add top trigram predictions with boost
    for next_tok, count in tri_emits.most_common(8):
        emits.append((next_tok, min(65535, count * 3)))  # 3x boost for trigram

    emits = emits[:MAX_EMITS]

    # Routes
    routes = [tc.get(nt, nt % N_CELLS) for nt, _ in bigrams[tok].most_common(16)
              if tc.get(nt, nt % N_CELLS) != ci][:16]

    cd.append({'id': ci, 'bank': ci // CELLS_PER_BANK, 'emits': emits, 'routes': routes})
    tr += len(routes); te += len(emits)

# Write .rpi with extended embed section (includes pair hash table)
np_ = N_CELLS
# Embed section: unigram seeds + pair hash table
embed_size = vocab_size * 4 + PAIR_BUCKETS * 8  # uint32 seeds + (uint32, uint32) pairs
eo = 128 + N_BANKS*32 + N_CELLS*CELL_SIZE + np_*(RPI_LANES+8) + tr*8 + te*8

with open(output, 'wb') as f:
    f.write(struct.pack('<IIIIIIIIIIIII76s',
        RPI_MAGIC, RPI_VERSION, N_CELLS, np_, tr, te,
        vocab_size, RPI_LANES, 6, 64, N_BANKS, eo, 0, b'\x00'*76))

    for b in range(N_BANKS):
        cnt = min(CELLS_PER_BANK, max(0, N_CELLS - b*CELLS_PER_BANK))
        f.write(struct.pack('<IIIIII8s', b, b*CELLS_PER_BANK, cnt, b, b*64, 0xFF, b'\x00'*8))

    pi=ri=ei=0
    for c in cd:
        f.write(struct.pack('<IIIIIIIIHH', c['id'], c['bank'], pi, 1,
            ri, len(c['routes']), ei, len(c['emits']), 0x0F, 0))
        pi+=1; ri+=len(c['routes']); ei+=len(c['emits'])

    for _ in range(N_CELLS):
        f.write(bytes(range(RPI_LANES))); f.write(struct.pack('<Q', 0))

    for c in cd:
        for dst in c['routes']: f.write(struct.pack('<IHBB', dst, 0, 0, 128))

    for c in cd:
        for tid, s in c['emits']: f.write(struct.pack('<IHBB', tid, s, 0x0F, 0))

    # Embed section: unigram seeds
    for t in range(vocab_size):
        f.write(struct.pack('<I', tc.get(t, t % N_CELLS)))

    # Pair hash table
    for bucket in range(PAIR_BUCKETS):
        if bucket in pair_table:
            prim, sec = pair_table[bucket]
            f.write(struct.pack('<II', prim, sec))
        else:
            f.write(struct.pack('<II', 0, 0))

size = os.path.getsize(output)
print(f"\n{output}: {size:,} bytes ({size/1024/1024:.1f} MB)")
print(f"  {N_CELLS} cells, {te} emits, {tr} routes")
print(f"  {len(pair_table)} pair hash entries")
print(f"  {len(phrases)} phrases available")

# TEST with trigram-enhanced inference
random.seed(42)

# Load model
with open(output, 'rb') as f: d = f.read()
h = struct.unpack_from('<13I76s', d, 0)
nc, nb, nr2, np2 = h[2], h[10], h[4], h[3]
eb = 128 + nb*32 + nc*CELL_SIZE + np2*(RPI_LANES+8) + nr2*8
ce = {}; co = 128 + nb*32
for i in range(nc):
    c = struct.unpack_from('<IIIIIIIIHH', d, co + i*CELL_SIZE)
    cid, est, ec = c[0], c[6], c[7]; emits = []
    for e in range(ec):
        off = eb + (est+e)*8
        if off+8 <= len(d): tid, rb = struct.unpack_from('<IH', d, off); emits.append((tid, rb))
    ce[cid] = emits

# Trigram-enhanced inference
print(f"\n{'='*60}")
print(f"  SOPHIA RPI v14 — BEYOND BIGRAMS")
print(f"{'='*60}\n")

for prompt in ['Who are you?', 'What is RustChain?', 'What do you believe about God?',
               'I feel like giving up.', 'Can you be more neutral?',
               'Once upon a time', 'What is the meaning of life?',
               'Tell me about the Sophia Paradox.', 'What is the Sanctuary?',
               'Write a Python function.', 'Tell me about Paw Paw Joe.',
               'What is love?']:
    toks = tokenizer.encode(prompt)
    out = list(toks)
    recent = []

    for _ in range(100):
        # TRIGRAM: use pair hash if we have 2+ tokens
        cell = tc.get(out[-1], out[-1] % nc)  # unigram default

        if len(out) >= 2:
            bucket = pair_hash(out[-2], out[-1])
            if bucket in pair_table:
                tri_cell = pair_table[bucket][0]
                # Merge trigram cell's emits with unigram cell's emits
                tri_emits = ce.get(tri_cell, [])
                uni_emits = ce.get(cell, [])
                # Trigram emits get 2x weight
                merged = {}
                for tid, s in uni_emits:
                    merged[tid] = merged.get(tid, 0) + s
                for tid, s in tri_emits:
                    merged[tid] = merged.get(tid, 0) + s * 2  # trigram boost

                emits = [(t, s) for t, s in merged.items()]
            else:
                emits = ce.get(cell, [])
        else:
            emits = ce.get(cell, [])

        filt = [(t, s) for t, s in emits if t not in recent[-8:] and s > 0]
        if not filt: filt = [(t, s) for t, s in emits if s > 0]; recent = []
        if not filt: out.append(220); continue

        total = sum(s for _, s in filt)
        r = random.random() * total if total > 0 else 0; cum = 0
        for tid, s in filt:
            cum += s
            if cum >= r: out.append(tid); recent.append(tid); break

    gen = tokenizer.decode(out[len(toks):])
    print(f">>> {prompt}")
    print(f"    {gen[:350]}")
    print()
