# RPI Performance Comparison

This guide gives reviewers and Raspberry Pi users a repeatable way to compare
RPI throughput across ARM and x86 systems without mixing hardware, model, or
prompt differences.

## Current Published Baseline

The project README currently reports these baseline results:

| Platform | Backend | Model size | Throughput | Notes |
|---|---|---:|---:|---|
| Any platform with Python | Python table lookup | 1.2 MB | 18,000 tok/s | Pure table lookup prototype |
| POWER8 S824 | C with VSX `vec_perm` | 3 MB | 84+ tok/s | 64 threads |
| x86_64 | Generic C backend | 1.2 MB | 50+ tok/s | Generic portable backend |
| Nintendo 64 | C, estimated | 868 KB | ~200 tok/s | MIPS R4300i estimate |

These numbers are useful as a published baseline, but a Pi-vs-x86 comparison
should be measured with the same model, prompt, token count, and build flags.

## Reproducible Comparison Matrix

Use this matrix when reporting new benchmark runs:

| System | CPU | RAM | OS | Backend | Model | Prompt tokens | Generated tokens | Median tok/s | p95 latency/token |
|---|---|---:|---|---|---|---:|---:|---:|---:|
| Raspberry Pi 4 | fill in | fill in | fill in | Generic C | fill in | fill in | fill in | fill in | fill in |
| Raspberry Pi 5 | fill in | fill in | fill in | Generic C | fill in | fill in | fill in | fill in | fill in |
| x86_64 baseline | fill in | fill in | fill in | Generic C | fill in | fill in | fill in | fill in | fill in |

Keep the x86 row boring on purpose: a common laptop or low-power mini PC is
more useful for Pi viability comparisons than a GPU workstation.

## Benchmark Recipe

Build once per system:

```bash
git clone https://github.com/Scottcjn/rpi-inference.git
cd rpi-inference
make clean
make
make test
```

Generate or reuse the same test model on every system:

```bash
python3 tools/gen_test_model.py --output models/benchmark.rpi
```

Run the same prompt and generation length on each system:

```bash
./rpi-cli \
  -m models/benchmark.rpi \
  -p "Explain zero-multiply inference in one paragraph." \
  -n 128
```

For a fair comparison:

- Run at least five trials and report the median.
- Keep the model file, prompt, and generated token count identical.
- Record CPU governor or power mode, especially on Raspberry Pi boards.
- Note whether the system is thermally throttling.
- Run without unrelated background workloads.

## Reporting Template

Paste this block into a benchmark report or issue comment:

```text
System:
CPU:
RAM:
OS/kernel:
Compiler:
Backend:
Model file:
Model size:
Prompt:
Generated tokens:
Trials:
Median tok/s:
p95 latency/token:
Thermal throttling observed:
Notes:
```

## Why This Matters

RPI is designed for CPU and cache hierarchy efficiency, so the useful question
is not whether a Raspberry Pi can beat a high-end workstation. The useful
question is whether a Pi-class device can meet a specific local inference
latency or throughput target at a much lower power and deployment cost.
