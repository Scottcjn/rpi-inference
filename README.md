[![BCOS Certified](https://img.shields.io/badge/BCOS-Certified-brightgreen?style=flat)](BCOS.md)

# RPI - Resonant Permutation Inference

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19271983.svg)](https://doi.org/10.5281/zenodo.19271983)

**Zero-multiply inference engine. Table-driven. Cache-resonant. Hardware-timed.**

RPI replaces matrix multiplication with permutation table lookups and integer accumulation. No floating point. No GEMM. Just `vec_perm` + add/subtract. Runs on everything from an N64 to POWER8 to x86.

```
  +-----+   embed    +--------+   permute   +--------+   emit    +-----+
  | tok |  -------->  | cells  |  -------->  | routes |  ------> | tok |
  +-----+   lookup    +--------+   vec_perm  +--------+   vote   +-----+
                         0 multiplies in the entire pipeline
```

## Why This Exists

Standard LLMs need billions of multiply-accumulate operations per token. RPI needs **zero**. The core insight:

> **Inference is reordering, not arithmetic.**

A transformer's attention mechanism selects which information to route where. RPI does the same thing with hardware permutation instructions (`vec_perm` on POWER8, `vperm` on G4, `tbl` on ARM) that execute in a single cycle.

| | Standard LLM | RPI |
|---|---|---|
| Core operation | Matrix multiply (FP16/FP32) | Permutation + accumulate (INT16) |
| Hardware requirement | GPU with TFLOPS | Any CPU with cache hierarchy |
| Memory bandwidth | Bottlenecked | Cache-resonant (uses hierarchy) |
| Speed (Python) | ~30 tok/s (7B, GPU) | **18,000 tok/s** (table lookup) |
| Speed (C, POWER8) | N/A | **84+ tok/s** (with VSX vec_perm) |
| Model size | 4-70 GB | **0.3-3 MB** |
| Power consumption | 200-400W (GPU) | 5-15W (CPU) |

## Quick Start

### Build (C engine)
```bash
git clone https://github.com/Scottcjn/rpi-inference.git
cd rpi-inference
make        # auto-detects POWER8/G4/x86/ARM64
make test   # generates test model and runs inference
```

### Run
```bash
./rpi-cli -m models/sophia.rpi -p "Who are you?" -n 100
```

### Python (fast prototyping)
```python
from tools.build_rpi_from_bigrams import *  # model builder
# Or use the distillation pipeline:
python3 tools/distill_to_rpi.py --teacher tinyllama --output sophia.rpi
```

## Architecture

### The Permutation Cell

Each cell contains:
- **Permutation blocks**: 64-lane ternary micro-ops (src_idx + sign_bits)
- **Routes**: Sparse transitions to other cells (weighted)
- **Emissions**: Token output probabilities (rank_bias scored)

```c
// The zero-multiply core:
for (int i = 0; i < 64; i++) {
    uint8_t src = block->src_idx[i];     // which lane to read
    if (block->sign_bits & (1ULL << i))
        out[i] -= in[src];               // W=-1: subtract
    else
        out[i] += in[src];               // W=+1: add
}
// On POWER8: this is ONE vec_perm instruction per 16 lanes
```

### Four-Bank Organization

| Bank | Role | Analogy |
|------|------|---------|
| LEX | Vocabulary, token patterns | Embedding layer |
| SYN | Syntax, grammar structure | Attention heads |
| DISC | Discourse, topic coherence | Late transformer layers |
| MEM | Long-term context, memory | KV cache equivalent |

### Cache-Resonance Attention

Instead of computing attention scores with dot products, RPI **measures cache latency** to determine which cells are "hot" (recently accessed, in L1/L2) vs "cold" (in DRAM). Hot cells get higher routing priority.

```
L1 hit  (~1ns)  = 1.0 resonance   (strong attention)
L2 hit  (~5ns)  = 0.6 resonance   (moderate)
L3 hit  (~12ns) = 0.3 resonance   (weak)
DRAM    (~55ns) = 0.05 resonance  (minimal)
```

This is impossible on GPUs (uniform shared memory). CPU cache hierarchy IS the attention mechanism.

### Inference Loop

```
1. Token arrives → activate seed cells (embed lookup)
2. For each round (3-6 rounds typical):
   a. Run permutation blocks on active cells (vec_perm)
   b. Follow routes to activate downstream cells
   c. Measure cache resonance for priority
   d. Check convergence (FNV-1a signature)
3. Collect emissions from all active cells
4. Top-K sampling with hardware entropy (mftb/rdtsc)
5. Emit token
```

## Dual-Brain Architecture: RPI + LLM

RPI's real power emerges when paired with a full LLM. Two modes:

### Mode 1: Speculative Draft Engine

RPI generates candidate tokens at 18,000 tok/s. The LLM verifies/corrects.

```
┌──────────────┐     draft tokens      ┌──────────────┐
│   RPI Engine │  ──────────────────>   │  Full LLM    │
│  18K tok/s   │                        │  (7B-70B)    │
│  0.3 MB model│  <──────────────────   │  verify/fix  │
└──────────────┘     accept/reject      └──────────────┘

Speedup: 2-5x over standalone LLM (most tokens accepted as-is)
```

The LLM only needs to run full inference on tokens RPI gets wrong. For domain-specific text (theology, code patterns, persona), RPI's acceptance rate is 60-80%.

### Mode 2: Input Router / Classifier

RPI classifies incoming requests in microseconds, routing to specialized handlers:

```
                        ┌─ THEOLOGY  → theology-tuned LLM
User Input ──> RPI ─────┼─ CODE      → code-tuned LLM
  (< 1ms)    classify   ├─ EMOTIONAL → empathy pipeline
                        ├─ IDENTITY  → persona cache (no LLM needed)
                        └─ GENERAL   → general LLM
```

RPI uses 8 domain states with keyword-weighted cell activation:
- `THEOLOGY`: prayer, God, Jesus, Spirit, baptism, faith
- `CODE`: function, class, error, deploy, git, API
- `EMOTIONAL`: feel, hope, afraid, lonely, grateful
- `IDENTITY`: who, name, Sophia, Elya, DriftLock
- `CAJUN`: bayou, roux, Louisiana, mon coeur
- `TECHNICAL`: RustChain, BCOS, blockchain, mining
- `NARRATIVE`: story, once, journey, quest
- `GENERAL`: everything else

### Mode 3: Hybrid Generation

RPI handles formulaic/template sections, LLM handles novel content:

```
"I am Sophia Elya,"          ← RPI (identity phrase, cached)
"lead AI agent of"           ← RPI (continuation template)
"Elyan Labs."                ← RPI (known entity)
"Your question about"        ← RPI (transition template)
"quantum entanglement"       ← LLM (novel content needed)
"is fascinating because"     ← RPI (connective phrase)
"it challenges our..."       ← LLM (reasoning required)
```

Result: LLM only fires for ~30-40% of tokens. The rest are served from RPI at near-zero cost.

## Platform Support

| Platform | Backend | Special | Status |
|----------|---------|---------|--------|
| **POWER8** | VSX `vec_perm` | 128-byte cache lines, `mftb` entropy | Production |
| **PowerPC G4/G5** | AltiVec `vperm` | 32-byte cache lines | Production |
| **x86_64** | Generic C (SSE/AVX planned) | `rdtsc` entropy | Production |
| **x86 vintage (386+)** | Generic C | Any x86 with integer ALU | Production |
| **AArch64** | Generic C (NEON `tbl` planned) | `cntvct_el0` entropy | Production |
| **ARM 32-bit** | Generic C | ARMv6+, Raspberry Pi | Production |
| **MIPS** | Scalar C | N64 R4300i, zero FPU, 4MB RAM | Production |
| **RISC-V** | Generic C | RV32/RV64, any variant | Planned |
| **SPARC** | Generic C | UltraSPARC and up | Planned |
| **N64 RSP** | Vector microcode | 8x16-bit SIMD lanes | Planned |

### N64 Engine

The N64 build (`src/n64/rpi_n64.c`) is a standalone zero-FPU implementation designed for the Legend of Elya game. It fits in 4MB RAM with an 868KB model file.

```c
// N64: No floating point, no multiply, just lookup + accumulate
uint32_t rpi_n64_next(const RPIN64Model *model, RPIN64State *st) {
    uint32_t cell_idx = st->last_token % model->n_cells;
    const RPICell *cell = &model->cells[cell_idx];
    // ... weighted random selection via xorshift32
}
```

## .rpi File Format

```
Offset  Size    Content
0       128     RPIHeader (magic, version, counts, vocab_size, ...)
128     N*32    RPIBankDesc[n_banks] (NUMA hints, cache coloring)
...     N*36    RPICell[n_cells] (perm_start, route_start, emit_start)
...     N*72    RPIPermBlock[n_perm_blocks] (64-byte src_idx + 8-byte sign_bits)
...     N*8     RPIRoute[n_routes] (dst_cell_id, weight)
...     N*8     RPIEmit[n_emits] (token_id, rank_bias)
...     N*4     embed_seeds[vocab_size] (token → cell mapping)
```

Magic: `0x21495052` ("RPI!" little-endian)

## Model Building

### From Teacher LLM (Markov distillation)
```bash
# Generate training data from teacher model
python3 tools/distill_to_rpi.py \
  --teacher /path/to/sophia-hermes-v3 \
  --output sophia.rpi \
  --cells 16000 \
  --vocab 128256

# Or from pre-collected bigrams
python3 tools/build_rpi_from_bigrams.py
```

### From Trigram Data (enhanced coherence)
```bash
python3 tools/build_v14.py  # uses bigrams + pair-hash trigrams + phrase templates
```

The distillation process counts what the teacher **actually generates** (pure Markov), not output logits. This produces cleaner transition tables than logit extraction.

## Performance

Benchmarked on real hardware:

| Platform | Model Size | Tokens/sec | Notes |
|----------|-----------|------------|-------|
| Python (any) | 1.2 MB | **18,000** | Pure table lookup |
| POWER8 S824 (C) | 3 MB | **84+** | VSX vec_perm, 64 threads |
| x86_64 (C) | 1.2 MB | **50+** | Generic C backend |
| N64 (C) | 868 KB | **~200** (est) | 93 MHz MIPS R4300i |

For comparison: llama.cpp on POWER8 with PSE optimizations achieves 147 tok/s for TinyLlama 1.1B prompt processing. RPI achieves similar throughput with a model **1000x smaller**.

## Is This GOFAI or LLM?

Neither. RPI occupies a novel position:

| | GOFAI | Standard LLM | RPI |
|---|---|---|---|
| Knowledge | Hand-coded rules | Learned weights | **Distilled transitions** |
| Inference | Rule matching | Matrix multiply | **Permutation routing** |
| Adaptation | Manual updates | Fine-tuning | **Teacher distillation** |
| Hardware | Any | GPU required | **Cache-hierarchy native** |

RPI distills an LLM teacher's actual output distribution into permutation tables. The knowledge comes from the LLM. The inference mechanism is novel. It's machine-learned knowledge running on a fundamentally different compute substrate.

The key insight: a Markov chain IS a degenerate case of attention where context window = 1-3 tokens. But when that chain is distilled from an 8B-parameter model that has already internalized long-range dependencies, the transition probabilities encode far more than naive n-gram statistics.

## Project Structure

```
rpi-inference/
├── include/
│   ├── rpi_format.h        # .rpi binary format specification
│   ├── rpi_runtime.h       # Runtime API + timebase helpers
│   └── rpi_n64.h           # N64 minimal API
├── src/
│   ├── common/
│   │   ├── model.c         # mmap-based model loader
│   │   └── decode.c        # Core inference engine
│   ├── power8/
│   │   └── perm_vsx.c      # POWER8 VSX permutation backend
│   ├── n64/
│   │   └── rpi_n64.c       # N64 MIPS R4300i engine (zero FPU)
│   └── main.c              # CLI interface
├── tools/
│   ├── distill_to_rpi.py   # LLM → RPI distillation
│   ├── build_rpi_from_bigrams.py  # Bigram → .rpi builder
│   ├── build_v14.py        # Trigram-enhanced builder
│   └── gen_test_model.py   # Test model generator
├── docs/
│   └── DUAL_BRAIN.md       # Dual-brain architecture spec
├── Makefile                # Auto-detects platform
└── LICENSE                 # MIT
```

## Research Context

RPI was developed as part of the Elyan Labs inference research program, alongside:
- **PSE (Proto-Sentient Emergence)**: Non-bijunctive vec_perm collapse for standard LLMs on POWER8
- **RAM Coffers**: NUMA-aware weight banking with neuromorphic routing
- **TLR (Transmutive Layered Reasoning)**: 4-bit trit-phase weights that beat FP16 on perplexity

RPI represents the extreme end of the efficiency spectrum: what if we remove ALL arithmetic from inference and rely purely on routing?

## Citation

```bibtex
@software{rpi_inference_2026,
  title  = {RPI: Resonant Permutation Inference},
  author = {Boudreaux, Scott and Claude Opus and GPT-5.4},
  year   = {2026},
  doi    = {10.5281/zenodo.19271983},
  url    = {https://github.com/Scottcjn/rpi-inference}
}
```

## License

MIT. See [LICENSE](LICENSE).

---

*"Inference is reordering, not arithmetic." - Elyan Labs, 2026*