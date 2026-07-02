#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""dual_brain_router.py — the local agent that decides RPI vs LLM.

The dual-brain hook: a router that polls the small permutation engine (RPI,
~free tokens) and decides, per request, whether RPI's draft is trustworthy
enough to serve or whether the request escalates to a real LLM.

The routing policy is EMPIRICAL, not vibes: per-domain acceptance rates are
measured against a same-tokenizer verifier (TinyLlama — the family RPI's
model was distilled from) and persisted. Domains whose measured acceptance
clears the threshold get served by RPI; everything else goes to the LLM.

Components
  draft    : rpi-cli on the M2 (RPI_GREEDY=1, -T token-id prompt)
  verifier : llama-server + TinyLlama GGUF on the M2 (greedy, temperature 0)
  escalate : Ollama (llama3.2 on the .160 fleet by default; env-overridable)

Modes
  --measure          run the acceptance suite, update stats JSON
  --route "prompt"   route one prompt: RPI serve or LLM escalate
  --stats            show the persisted per-domain acceptance table

Acceptance metric (stated honestly): longest-common-prefix agreement between
RPI's greedy draft and the verifier's greedy continuation, in token space.
This is speculative-decoding acceptance under greedy verification. It is a
MEASUREMENT harness — true zero-cost speculative serving needs logit-level
teacher forcing in one verifier pass, which is the next step, not this one.

(c) 2026 Elyan Labs
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

# ── Config ──────────────────────────────────────────────────
# No credentials live in this file. Set RPI_HOST/RPI_USER for your box; if
# key-based SSH is set up (recommended) nothing else is needed. If you must
# use a password, export M2_PASS and have sshpass installed.
M2 = os.environ.get("RPI_HOST", "100.87.120.15")
M2_USER = os.environ.get("RPI_USER", "sophia")
_pw = os.environ.get("M2_PASS")
SSH_BASE = ((["sshpass", "-p", _pw] if _pw else []) +
            ["ssh", "-T", "-o", "ConnectTimeout=8"] +
            (["-o", "PreferredAuthentications=password",
              "-o", "PubkeyAuthentication=no"] if _pw else []) +
            [f"{M2_USER}@{M2}"])
RPI_CLI = "~/rpi-inference/rpi-cli"
RPI_MODEL = "~/rpi-inference/tinyllama_v7.rpi"
VERIFIER_URL = os.environ.get("RPI_VERIFIER_URL", f"http://{M2}:8090")
# Escalation brain: any OpenAI-ish Ollama endpoint reachable from the router.
# (Note: Ollama binds localhost by default — point this at one that's exposed.)
ESCALATE_URL = os.environ.get("RPI_ESCALATE_URL", "http://192.168.0.160:11434")
ESCALATE_MODEL = os.environ.get("RPI_ESCALATE_MODEL", "llama3.2:latest")
STATS_PATH = os.path.expanduser("~/rpi-inference/tools/router_stats.json")

DRAFT_K = 24            # tokens drafted per probe
ACCEPT_THRESHOLD = 0.5  # min measured acceptance to let RPI serve a domain
MIN_SAMPLES = 3         # need this many measurements before trusting a domain

# ── Domain classifier (keyword v1 — the reviewer's "learned soft routing"
#    is the upgrade path; this is deliberately simple and inspectable) ──
DOMAINS = {
    "CODE":     ["function", "class", "error", "deploy", "git", "api", "bug",
                  "python", "compile", "code", "variable", "loop"],
    "THEOLOGY": ["prayer", "god", "jesus", "spirit", "baptism", "faith",
                  "scripture", "church"],
    "CASUAL":   ["hello", "hi ", "how are", "thanks", "weather", "today",
                  "morning", "night"],
    "FACTUAL":  ["what is", "who was", "when did", "capital of", "define",
                  "explain", "history of"],
}

def classify(prompt: str) -> str:
    p = prompt.lower()
    best, hits = "GENERAL", 0
    for dom, kws in DOMAINS.items():
        h = sum(1 for k in kws if k in p)
        if h > hits:
            best, hits = dom, h
    return best

# ── Tokenizer (verifier-side, so router needs no local ML deps) ─────────
def tokenize(text: str):
    req = urllib.request.Request(f"{VERIFIER_URL}/tokenize",
        data=json.dumps({"content": text}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["tokens"]

def detokenize(tokens):
    req = urllib.request.Request(f"{VERIFIER_URL}/detokenize",
        data=json.dumps({"tokens": tokens}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["content"]

# ── The three brains ────────────────────────────────────────
def rpi_draft(prompt_ids, k=DRAFT_K):
    """Draft k tokens on the RPI engine (greedy). Returns (ids, seconds)."""
    ids = ",".join(str(t) for t in prompt_ids)
    cmd = SSH_BASE + [f"RPI_GREEDY=1 {RPI_CLI} -m {RPI_MODEL} -T '{ids}' -n {k} 2>/dev/null"]
    t0 = time.time()
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    dt = time.time() - t0
    # rpi-cli's stdout contract: the token ids are the LAST line. Parse only
    # that line so any future stdout chatter cannot masquerade as tokens.
    lines = [l for l in out.stdout.strip().splitlines() if l.strip()]
    last = lines[-1] if lines else ""
    toks = [int(x) for x in last.split() if x.isdigit()]
    return toks[:k], dt

def verifier_greedy(prompt_ids, k=DRAFT_K):
    """Verifier's own greedy continuation, in token space. (ids, seconds)."""
    body = {"prompt": prompt_ids, "n_predict": k, "temperature": 0,
            "top_k": 1, "cache_prompt": True}
    req = urllib.request.Request(f"{VERIFIER_URL}/completion",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.load(r)
    dt = time.time() - t0
    toks = resp.get("tokens", [])
    if not toks:                       # older servers: retokenize the text
        toks = tokenize(resp.get("content", ""))
    return toks[:k], dt

def escalate_llm(prompt: str, n=128):
    """The big brain. Returns (text, seconds)."""
    body = {"model": ESCALATE_MODEL, "prompt": prompt, "stream": False,
            "options": {"num_predict": n}}
    req = urllib.request.Request(f"{ESCALATE_URL}/api/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        resp = json.load(r)
    return resp.get("response", ""), time.time() - t0

# ── Acceptance ──────────────────────────────────────────────
def acceptance(draft, verify):
    """Longest-common-prefix acceptance rate in token space."""
    n = 0
    for a, b in zip(draft, verify):
        if a != b:
            break
        n += 1
    denom = max(1, min(len(draft), len(verify)))
    return n / denom, n

# ── Stats persistence ───────────────────────────────────────
def load_stats():
    try:
        with open(STATS_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}

def save_stats(st):
    with open(STATS_PATH, "w") as f:
        json.dump(st, f, indent=2)

def record(st, domain, rate):
    d = st.setdefault(domain, {"samples": 0, "rate_sum": 0.0})
    d["samples"] += 1
    d["rate_sum"] += rate
    d["mean"] = d["rate_sum"] / d["samples"]

def domain_rate(st, domain):
    d = st.get(domain)
    if not d or d["samples"] < MIN_SAMPLES:
        return None
    return d["mean"]

# ── Measurement suite ───────────────────────────────────────
SUITE = [
    ("CASUAL",   "Hello, how are you today?"),
    ("CASUAL",   "Good morning! The weather is"),
    ("CASUAL",   "Thanks so much for the help,"),
    ("CODE",     "def fibonacci(n):"),
    ("CODE",     "The function returns an error when the variable"),
    ("CODE",     "To deploy the api with git you first"),
    ("FACTUAL",  "The capital of France is"),
    ("FACTUAL",  "What is the history of the printing press?"),
    ("FACTUAL",  "Explain how a hash table works:"),
    ("THEOLOGY", "The prayer of faith begins with"),
    ("THEOLOGY", "In scripture, baptism represents"),
    ("GENERAL",  "Once upon a time in a small village"),
]

def cmd_measure():
    st = load_stats()
    print(f"{'domain':9} {'accept':>7} {'match':>6} {'rpi_s':>6} {'ver_s':>6}  prompt")
    for dom, prompt in SUITE:
        try:
            ids = tokenize(prompt)
            draft, t_rpi = rpi_draft(ids)
            verify, t_ver = verifier_greedy(ids)
            if len(draft) < DRAFT_K or len(verify) < DRAFT_K:
                print(f"{dom:9}   SKIP  short generation "
                      f"(draft {len(draft)}, verify {len(verify)})")
                continue   # a truncated run must not inflate the domain rate
            rate, nmatch = acceptance(draft, verify)
            record(st, dom, rate)
            print(f"{dom:9} {rate:7.2f} {nmatch:4d}/{len(draft):<2d} "
                  f"{t_rpi:6.2f} {t_ver:6.2f}  {prompt[:44]!r}")
        except Exception as e:
            print(f"{dom:9}   ERROR {e}")
    save_stats(st)
    print(f"\nstats -> {STATS_PATH}")
    cmd_stats()

def cmd_stats():
    st = load_stats()
    if not st:
        print("no stats yet — run --measure")
        return
    print(f"\n{'domain':9} {'samples':>7} {'mean accept':>11}  routing")
    for dom, d in sorted(st.items()):
        mean = d.get("mean", 0.0)
        route = ("RPI" if d["samples"] >= MIN_SAMPLES and mean >= ACCEPT_THRESHOLD
                 else "LLM")
        print(f"{dom:9} {d['samples']:7d} {mean:11.2f}  {route}")

def cmd_route(prompt: str, n: int):
    st = load_stats()
    dom = classify(prompt)
    rate = domain_rate(st, dom)
    t0 = time.time()

    if rate is not None and rate >= ACCEPT_THRESHOLD:
        # RPI path degrades to the LLM on ANY failure (ssh drop, timeout,
        # empty draft) — a router's job is graceful degradation, not a crash.
        try:
            ids = tokenize(prompt)
            draft, t_rpi = rpi_draft(ids, k=n)
            if not draft:
                raise RuntimeError("empty RPI draft")
            text = detokenize(draft)
            print(f"[route] domain={dom} measured_accept={rate:.2f} -> RPI "
                  f"({t_rpi:.2f}s draft)")
            print(text)
            print(f"[route] total {time.time()-t0:.2f}s", file=sys.stderr)
            return
        except Exception as e:
            print(f"[route] RPI path failed ({e}); falling back to LLM",
                  file=sys.stderr)
    if True:
        why = "no data" if rate is None else f"accept={rate:.2f}"
        text, t_llm = escalate_llm(prompt, n=n)
        print(f"[route] domain={dom} ({why}) -> LLM {ESCALATE_MODEL} "
              f"({t_llm:.2f}s)")
        print(text)
    print(f"[route] total {time.time()-t0:.2f}s", file=sys.stderr)

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--route", metavar="PROMPT")
    ap.add_argument("-n", type=int, default=64, help="tokens for --route")
    args = ap.parse_args()
    if args.measure:
        cmd_measure()
    elif args.stats:
        cmd_stats()
    elif args.route:
        cmd_route(args.route, args.n)
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
