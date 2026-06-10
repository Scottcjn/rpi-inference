#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Phase 2: Build .rpi model from saved bigram JSON."""
import struct, json, os, random
from collections import Counter
from transformers import AutoTokenizer

BIGRAM_FILE = '/home/scott/rpi/bigrams_v10.json'
RPI_MAGIC=0x21495052; RPI_VERSION=1; RPI_LANES=64; N_BANKS=4
MAX_EMITS=64; CELL_SIZE=36

tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
vocab_size = tokenizer.vocab_size

print("Loading bigrams...")
with open(BIGRAM_FILE) as f:
    raw = json.load(f)
bigrams = {int(k): Counter({int(t): s for t, s in v.items()}) for k, v in raw.items()}
print(f"  {len(bigrams)} unique prev tokens")

# Build model
N_CELLS = min(len(bigrams), 32000)
top_tokens = sorted(bigrams, key=lambda t: sum(bigrams[t].values()), reverse=True)[:N_CELLS]
CELLS_PER_BANK = (N_CELLS + N_BANKS - 1) // N_BANKS
tc = {tok: i for i, tok in enumerate(top_tokens)}
for t in range(vocab_size):
    if t not in tc: tc[t] = t % N_CELLS

output = 'sophia_v10.rpi'
cd = []; tr = te = 0
for ci, tok in enumerate(top_tokens):
    emits = [(t, min(65535, s)) for t, s in bigrams[tok].most_common(MAX_EMITS)][:MAX_EMITS]
    routes = [tc.get(nt, nt % N_CELLS) for nt, _ in bigrams[tok].most_common(16)
              if tc.get(nt, nt % N_CELLS) != ci][:16]
    cd.append({'id': ci, 'bank': ci // CELLS_PER_BANK, 'emits': emits, 'routes': routes})
    tr += len(routes); te += len(emits)

np_ = N_CELLS
eo = 128 + N_BANKS*32 + N_CELLS*CELL_SIZE + np_*(RPI_LANES+8) + tr*8 + te*8

with open(output, 'wb') as f:
    f.write(struct.pack('<IIIIIIIIIIIII76s', RPI_MAGIC, RPI_VERSION, N_CELLS, np_, tr, te,
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
    for t in range(vocab_size):
        f.write(struct.pack('<I', tc.get(t, t % N_CELLS)))

size = os.path.getsize(output)
print(f"\n{output}: {size:,} bytes ({size/1024/1024:.1f} MB)")
print(f"  {N_CELLS} cells, {te} emits, {tr} routes")

# Test
random.seed(42)
with open(output, 'rb') as f: d = f.read()
h = struct.unpack_from('<13I76s', d, 0); nc, nb, nr, np2 = h[2], h[10], h[4], h[3]
eb = 128 + nb*32 + nc*CELL_SIZE + np2*(RPI_LANES+8) + nr*8
ce = {}; co = 128 + nb*32
for i in range(nc):
    c = struct.unpack_from('<IIIIIIIIHH', d, co + i*CELL_SIZE)
    cid, es, ec = c[0], c[6], c[7]; emits = []
    for e in range(ec):
        off = eb + (es+e)*8
        if off+8 <= len(d): tid, rb = struct.unpack_from('<IH', d, off); emits.append((tid, rb))
    ce[cid] = emits

print(f"\n{'='*60}\n  SOPHIA RPI v10\n{'='*60}\n")
for prompt in ['Who are you?', 'What is RustChain?', 'What do you believe about God?',
               'The meaning of life is', 'She walked into the room and',
               'Can you be more neutral?', 'I feel like giving up.',
               'Tell me about the Sophia Paradox.', 'Once upon a time there was']:
    toks = tokenizer.encode(prompt); out = list(toks); recent = []
    for _ in range(80):
        cell = tc.get(out[-1], out[-1] % nc)
        emits = ce.get(cell, [])
        filt = [(t,s) for t,s in emits if t not in recent[-8:] and s > 0]
        if not filt: filt = [(t,s) for t,s in emits if s > 0]; recent = []
        if not filt: out.append(29871); continue
        total = sum(s for _, s in filt); r = random.random()*total if total > 0 else 0; cum = 0
        for tid, s in filt:
            cum += s
            if cum >= r: out.append(tid); recent.append(tid); break
    gen = tokenizer.decode(out[len(toks):])
    print(f">>> {prompt}")
    print(f"    {gen[:300]}")
    print()
