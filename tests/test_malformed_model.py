#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Regression test: the loader must reject malformed .rpi files instead of
walking section pointers off header-declared counts and crashing.

A .rpi is an untrusted external artifact. rpi_model_load() only checked magic
and version, then trusted n_cells/n_perm_blocks/n_routes/n_emits/embed_offset,
so a truncated or crafted file drove the CLI to a segfault (OOB read).

Each case runs the real CLI. A clean rejection is a NON-crash exit (the loader
prints an error and returns -1). A crash is a negative return code (killed by a
signal, e.g. -11 SIGSEGV) or the shell's 128+signal convention.

Run: python3 tests/test_malformed_model.py   (builds rpi-cli via make if needed)
"""
import os
import struct
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI = os.path.join(ROOT, "rpi-cli")

RPI_MAGIC = 0x21495052
RPI_VERSION = 1


def make_header(n_cells, n_perm, n_routes, n_emits, vocab, n_banks, embed_offset):
    return struct.pack(
        "<IIIIIIIIIIII I 76s",
        RPI_MAGIC, RPI_VERSION,
        n_cells, n_perm, n_routes, n_emits,
        vocab, 64, 8, 32, n_banks, embed_offset, 0,
        b"\x00" * 76,
    )


def run_cli(path):
    r = subprocess.run(
        [CLI, "-m", path, "-p", "hi", "-n", "4"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30,
    )
    return r.returncode


def check(name, data):
    with tempfile.NamedTemporaryFile(suffix=".rpi", delete=False) as tf:
        tf.write(data)
        path = tf.name
    try:
        rc = run_cli(path)
    finally:
        os.unlink(path)
    crashed = rc < 0 or rc >= 128  # killed by signal
    status = "CRASH" if crashed else "rejected cleanly"
    print(f"  {name}: rc={rc} -> {status}")
    return not crashed


def main():
    if not os.path.exists(CLI):
        subprocess.run(["make", "rpi-cli"], cwd=ROOT, check=True)

    cases = [
        # valid magic+version, but body is absent and counts are huge
        ("oversized-counts (128B file claims 1e6 of each section)",
         make_header(1_000_000, 1_000_000, 1_000_000, 1_000_000,
                     256, 4, 64_000_000)),
        # embed_offset points 64MB into a 128-byte mapping
        ("embed_offset out of bounds",
         make_header(0, 0, 0, 0, 256, 0, 64_000_000)),
        # file shorter than the fixed 128-byte header
        ("truncated below header size",
         make_header(0, 0, 0, 0, 0, 0, 0)[:64]),
    ]

    ok = True
    print("Malformed .rpi rejection test:")
    for name, data in cases:
        ok = check(name, data) and ok

    if ok:
        print("PASS: loader rejected every malformed file without crashing")
        return 0
    print("FAIL: loader crashed on a malformed file (OOB read)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
