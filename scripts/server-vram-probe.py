#!/usr/bin/env python3

"""Measure real device VRAM against llama.cpp's own accounting, as a function of
prefill depth.

Motivation: on some backends (notably SYCL) a significant amount of device memory
is allocated outside llama.cpp's memory breakdown, so `--fit` over-commits and a
deep prefill can OOM even though the projection said it would fit.  See
docs/development/sycl-fit-unaccounted-vram.md.

VRAM is read from `llamacpp:vram_total_bytes` / `llamacpp:vram_free_bytes` on the
server's /metrics endpoint.  Those come from `ggml_backend_dev_memory`, i.e. the
real device-wide figure that `--list-devices` prints -- not a tally of ggml
buffers -- which is what makes unaccounted allocations visible.

Subcommands:

  compare    used VRAM at one or more prefill depths, for N models.  Use with
             presets that pin --ctx-size, to A/B e.g. KV cache types.
  ladder     walk a depth ladder as fractions of each model's *fitted* n_ctx.
             Use with --fit presets (--ctx-size unset) to see how the
             unaccounted term grows with depth.
  replicate  repeat one measurement, interleaved across models, for error bars.

Examples:

  # A/B two pinned-context presets at 7k tokens, 3 replicates each
  scripts/server-vram-probe.py --url http://host:8080 \
      replicate --depth 6900 --reps 3 model-f16 model-q4

  # depth ladder on two --fit presets
  scripts/server-vram-probe.py --url http://host:8080 \
      ladder model-max-f16 model-max-q8

Caveats this script handles for you, both of which silently corrupt results:

  * `sycl::ext::intel::info::device::free_memory` degrades to `free = total`
    without ZES_ENABLE_SYSMAN=1.  We check and refuse to report in that case.
  * In router mode the previous child's VRAM is still resident for a while after
    a model swap; without a settle delay readings scatter by hundreds of MiB.
    Default settle is 15s, which gave bit-identical replicates on an Arc B70.

One caveat this script cannot handle for you: growth-over-depth figures are only
meaningful against a *freshly loaded* child.  A router will reuse an already
loaded model, and its scratch pool is already at its high-water mark, so a
`ladder` re-run on the currently-loaded model reports +0.0 growth.  Either pass
two or more models (they alternate, so each gets a fresh load) or force a swap
first.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

MIB = 1024 * 1024

# Calibrated for FILLER below; only affects how close we land to a requested
# depth, never the measurement itself (we report the server's prompt_tokens).
CHARS_PER_TOKEN = 5.7

FILLER = (
    "The quick brown fox jumps over the lazy dog while seventeen diligent "
    "engineers carefully measure the memory footprint of a quantized key "
    "value cache on a graphics processor. "
)

# Kept as a constant prefix so a growing suffix still hits the prompt cache,
# which makes a ladder incremental rather than re-prefilling from scratch.
INSTR = "Summarise the following text in one word.\n\n"


class Server:
    def __init__(self, url, settle, timeout):
        self.url = url.rstrip("/")
        self.settle = settle
        self.timeout = timeout

    def _post(self, path, payload=None):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(
            self.url + path, data=data,
            headers={"Content-Type": "application/json"} if data else {})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return r.read().decode()

    def metrics(self):
        """Parse llamacpp:* gauges. Assumes a single active model (router mode
        proxies only the loaded child)."""
        out = {}
        for line in self._post("/metrics").splitlines():
            if not line.startswith("llamacpp:"):
                continue
            name, _, rest = line.partition("{")
            if rest:
                _, _, val = rest.partition("} ")
            else:
                name, _, val = line.partition(" ")
            try:
                out[name] = float(val)
            except ValueError:
                pass
        return out

    def snapshot(self):
        m = self.metrics()
        total = m.get("llamacpp:vram_total_bytes")
        free = m.get("llamacpp:vram_free_bytes")
        if total is None or free is None:
            raise SystemExit(
                "error: llamacpp:vram_{total,free}_bytes absent -- start the "
                "server with --metrics")
        if total == free:
            raise SystemExit(
                "error: free == total, so the backend is not reporting real "
                "free memory (SYCL needs ZES_ENABLE_SYSMAN=1). Readings would "
                "be meaningless; refusing to continue.")
        return {
            "used": total - free,
            "free": free,
            "total": total,
            "kv": (m.get("llamacpp:kv_cache_k_bytes", 0.0)
                   + m.get("llamacpp:kv_cache_v_bytes", 0.0)),
            "kv_k": m.get("llamacpp:kv_cache_k_bytes", 0.0),
            "kv_v": m.get("llamacpp:kv_cache_v_bytes", 0.0),
            "cells": m.get("llamacpp:kv_cache_cells", 0.0),
            "tokens": m.get("llamacpp:kv_cache_tokens", 0.0),
        }

    def chat(self, model, content, max_tokens, cache_prompt=True):
        return json.loads(self._post("/v1/chat/completions", {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "cache_prompt": cache_prompt,
        }))

    def load(self, model):
        """Force the router to load `model`, then settle so the previous
        child's VRAM has actually been released before we measure."""
        self.chat(model, "Say ok.", 4)
        time.sleep(self.settle)
        return self.snapshot()

    def prefill(self, model, target_tokens, cache_prompt=True):
        r = self.chat(model, build_prompt(target_tokens), 8, cache_prompt)
        time.sleep(self.settle)
        return r.get("usage", {}).get("prompt_tokens"), self.snapshot()


def build_prompt(target_tokens):
    chars = int(target_tokens * CHARS_PER_TOKEN)
    reps = chars // len(FILLER) + 1
    return INSTR + (FILLER * reps)[:chars]


def fmt(s, base=None):
    extra = "" if base is None else f"  (+{(s['used'] - base) / MIB:7.1f})"
    return (f"used={s['used'] / MIB:9.1f}  free={s['free'] / MIB:8.1f}  "
            f"kv={s['kv'] / MIB:8.1f}  tok={s['tokens']:7.0f}{extra}")


def cmd_compare(srv, args):
    results = []
    for model in args.models:
        print(f"\n=== {model} ===", flush=True)
        at_load = srv.load(model)
        print(f"  {'load':>12}  {fmt(at_load)}", flush=True)
        base = at_load["used"]
        rungs = []
        for depth in args.depths:
            t0 = time.time()
            ptok, s = srv.prefill(model, depth)
            print(f"  {ptok:>12}  {fmt(s, base)}  [{time.time() - t0:.0f}s]",
                  flush=True)
            rungs.append({"ptok": ptok, **s})
        results.append({"model": model, "load": {"ptok": 0, **at_load},
                        "rungs": rungs})

    # With --depths '' this degenerates to a load-only sweep, which is what you
    # want for reading load-time state (e.g. against shutdown memory breakdowns).
    print(f"\n{'model':<30} {'cells':>8} {'kv':>9} {'used':>10} {'free':>10} "
          f"{'ptok':>8}")
    for r in results:
        last = r["rungs"][-1] if r["rungs"] else r["load"]
        print(f"{r['model']:<30} {last['cells']:8.0f} {last['kv'] / MIB:9.1f} "
              f"{last['used'] / MIB:10.1f} {last['free'] / MIB:10.1f} "
              f"{str(last['ptok']):>8}")
    return results


def cmd_ladder(srv, args):
    results = []
    for model in args.models:
        print(f"\n=== {model} ===", flush=True)
        at_load = srv.load(model)
        n_ctx = int(at_load["cells"])
        if n_ctx == 0:
            raise SystemExit(f"error: {model} reported 0 KV cells")
        print(f"  fitted n_ctx = {n_ctx}   {fmt(at_load)}", flush=True)
        base, rungs = at_load["used"], []
        for frac in args.fracs:
            t0 = time.time()
            try:
                ptok, s = srv.prefill(model, int(n_ctx * frac))
            except (urllib.error.HTTPError, urllib.error.URLError) as e:
                # An OOM at depth is a result, not a crash of the harness.
                print(f"  {frac:>5.0%}  FAILED {type(e).__name__}: {e}",
                      flush=True)
                rungs.append({"frac": frac, "failed": str(e)})
                break
            print(f"  {frac:>5.0%}  {fmt(s, base)}  [{time.time() - t0:.0f}s]",
                  flush=True)
            rungs.append({"frac": frac, "ptok": ptok, **s})
        results.append({"model": model, "n_ctx": n_ctx, "load": at_load,
                        "rungs": rungs})

    print(f"\n{'model':<26} {'n_ctx':>8} {'kv':>9} {'free@load':>10} "
          f"{'free@deep':>10} {'growth':>9} {'KiB/tok':>8}")
    for r in results:
        ok = [x for x in r["rungs"] if "failed" not in x]
        if not ok:
            print(f"{r['model']:<26} {r['n_ctx']:>8}  (no successful rungs)")
            continue
        last = ok[-1]
        growth = last["used"] - r["load"]["used"]
        per_tok = growth / last["ptok"] / 1024 if last["ptok"] else 0.0
        print(f"{r['model']:<26} {r['n_ctx']:>8} {last['kv'] / MIB:9.1f} "
              f"{r['load']['free'] / MIB:10.1f} {last['free'] / MIB:10.1f} "
              f"{growth / MIB:+9.1f} {per_tok:8.2f}")
    return results


def cmd_replicate(srv, args):
    data = {m: [] for m in args.models}
    kv = {}
    for rep in range(args.reps):
        for model in args.models:            # interleaved, not blocked
            srv.load(model)
            ptok, s = srv.prefill(model, args.depth, cache_prompt=False)
            data[model].append(s["used"])
            kv[model] = s["kv"]
            print(f"  rep{rep + 1} {model:<26} {fmt(s)}  ptok={ptok}",
                  flush=True)

    print(f"\n{'model':<26} {'kv':>9} {'mean used':>10} {'stdev':>8} "
          f"{'min':>10} {'max':>10}")
    means = {}
    for model in args.models:
        v = [x / MIB for x in data[model]]
        means[model] = sum(v) / len(v)
        if len(v) > 1:
            mu = means[model]
            sd = (sum((x - mu) ** 2 for x in v) / (len(v) - 1)) ** 0.5
        else:
            sd = 0.0
        print(f"{model:<26} {kv[model] / MIB:9.1f} {means[model]:10.1f} "
              f"{sd:8.1f} {min(v):10.1f} {max(v):10.1f}")

    if len(args.models) == 2:
        a, b = args.models
        nominal = (kv[a] - kv[b]) / MIB
        actual = means[a] - means[b]
        print(f"\n  nominal KV saving {a} -> {b}: {nominal:9.1f} MiB")
        print(f"  measured VRAM saving:            {actual:9.1f} MiB")
        if nominal:
            print(f"  shortfall:                       "
                  f"{nominal - actual:9.1f} MiB "
                  f"({100 * (nominal - actual) / nominal:.1f}% of nominal)")
    return {"used": data, "kv": kv}


def int_list(s):
    return [int(x) for x in s.split(",") if x]


def float_list(s):
    return [float(x) for x in s.split(",") if x]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080",
                    help="server base URL (default: %(default)s)")
    ap.add_argument("--settle", type=float, default=15.0,
                    help="seconds to wait before each reading, so a swapped-out "
                         "model's VRAM is released (default: %(default)s)")
    ap.add_argument("--timeout", type=float, default=3600.0,
                    help="HTTP timeout in seconds (default: %(default)s)")
    ap.add_argument("--json", metavar="PATH", help="also write results as JSON")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # NB: comma-separated rather than nargs="+", which would swallow the
    # trailing positional model names.
    p = sub.add_parser("compare", help="used VRAM at fixed prefill depths")
    p.add_argument("--depths", type=int_list, default=[8000, 16000, 30000],
                   metavar="N,N,...", help="prefill depths in tokens")
    p.add_argument("models", nargs="+")
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("ladder", help="depth ladder over each model's fitted n_ctx")
    p.add_argument("--fracs", type=float_list,
                   default=[0.10, 0.35, 0.60, 0.85, 0.97],
                   metavar="F,F,...", help="fractions of fitted n_ctx")
    p.add_argument("models", nargs="+")
    p.set_defaults(fn=cmd_ladder)

    p = sub.add_parser("replicate", help="repeat one measurement for error bars")
    p.add_argument("--depth", type=int, required=True)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("models", nargs="+")
    p.set_defaults(fn=cmd_replicate)

    args = ap.parse_args()
    srv = Server(args.url, args.settle, args.timeout)
    res = args.fn(srv, args)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
