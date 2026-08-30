#!/usr/bin/env python3

"""Walk a llama-server router's configured models, find each one's context limit,
fill it, and report prefill/generation throughput and device memory against depth.

Answers three questions per model:

  * what context did it actually get?  (--fit chooses this; it is not the preset value)
  * does it survive its own context being filled?
  * how much do prefill and generation slow down with depth, and how much VRAM is left?

The last one matters on hardware whose decode is memory-latency-bound, where
generation throughput falls off as the KV cache grows -- the point of reporting a
ratio between the shallowest and deepest rung rather than a single number.

By default only models the router was *configured* with are tested (`source ==
"preset"`), not everything it discovered in the HuggingFace cache, since the cache
typically holds far more than you want to sit through.

    scripts/server-context-bench.py --url http://host:8080
    scripts/server-context-bench.py --url http://host:8080 --fracs 0.1,0.5,0.95
    scripts/server-context-bench.py --url http://host:8080 --models my-preset

Each rung re-prefills from scratch (cache_prompt=false) so the prefill rate is
honest rather than measuring a cache hit. That is the slow part: budget roughly
(sum of fracs) x n_ctx tokens of prefill per model.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

MIB = 1024 * 1024

FILLER = ("The archives of the Meridian Institute contain extensive records of "
          "atmospheric measurements taken across four continents over six decades. ")

# Planted mid-prompt so that a wrong answer indicates the deep KV cache is not being
# read back correctly, rather than merely that the model is being vague.
FACT_VALUE = "8317"
FACT = f"\n\nIMPORTANT FACT: the calibration constant is {FACT_VALUE}.\n\n"
QUESTION = ("\n\nIgnore the filler text. Answer with just the number: what is the "
            "calibration constant stated above?")


def http_json(url, payload=None, timeout=3600):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def http_text(url, timeout=120):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode()


class Server:
    def __init__(self, url, timeout):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def models(self, include_cache=False):
        d = http_json(self.url + "/models", timeout=60)
        entries = d["data"] if isinstance(d, dict) and "data" in d else d
        return [m for m in entries
                if include_cache or m.get("source") == "preset"]

    def metrics(self, model=None):
        """In router mode /metrics requires the model name and answers 400 without it.
        Any failure yields no gauges, which callers treat as "not exported"."""
        url = self.url + "/metrics"
        if model:
            url += "?" + urllib.parse.urlencode({"model": model})
        out = {}
        try:
            text = http_text(url)
        except Exception:
            return out
        for line in text.splitlines():
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

    def state(self, model=None):
        """KV cache and VRAM gauges, or None where the build does not export them.

        Only the timings are essential; the gauges are a convenience, and upstream
        llama-server exports neither the KV cache nor the VRAM series. Everything here
        is therefore optional, so the script still works against a stock build."""
        m = self.metrics(model)
        total = m.get("llamacpp:vram_total_bytes")
        free = m.get("llamacpp:vram_free_bytes")
        k = m.get("llamacpp:kv_cache_k_bytes")
        v = m.get("llamacpp:kv_cache_v_bytes")
        cells = m.get("llamacpp:kv_cache_cells")
        return {
            "cells": int(cells) if cells else None,
            "kv_mib": (k + v) / MIB if k is not None and v is not None else None,
            "vram_used_mib": (total - free) / MIB if total and free else None,
            "vram_free_mib": free / MIB if free is not None else None,
            "vram_total_mib": total / MIB if total else None,
        }

    def chat(self, model, content, max_tokens, cache_prompt=False):
        return http_json(self.url + "/v1/chat/completions", {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "cache_prompt": cache_prompt,
        }, timeout=self.timeout)


def answer_text(resp):
    m = resp["choices"][0]["message"]
    return ((m.get("content") or "") + " " + (m.get("reasoning_content") or "")).strip()


def build_prompt(target_tokens, chars_per_token, with_fact):
    chars = max(1, int(target_tokens * chars_per_token))
    reps = chars // len(FILLER) + 1
    body = (FILLER * reps)[:chars]
    if not with_fact:
        return body
    mid = len(body) // 2
    return body[:mid] + FACT + body[mid:] + QUESTION


def bench_model(srv, model, fracs, gen_tokens, recall_at_deepest, settle, abs_depths=None):
    print(f"\n=== {model} ===", flush=True)
    t0 = time.time()
    try:
        srv.chat(model, "Say ok.", 4)
    except Exception as e:
        print(f"  load FAILED: {type(e).__name__}: {e}")
        return {"model": model, "error": f"load: {e}"}
    # A router swaps models by replacing the child process, and the outgoing child's
    # device memory is not released instantly. Reading VRAM immediately after a load
    # therefore reports the two models' usage summed -- which shows up as an alarming
    # near-zero free figure. Wait before the first reading.
    time.sleep(settle)
    st = srv.state(model)

    # Calibrate chars/token against this model's tokenizer rather than assuming.
    probe = srv.chat(model, FILLER * 20, 1)
    ptok = probe.get("usage", {}).get("prompt_tokens") or 1
    cpt = (len(FILLER) * 20) / ptok

    n_ctx = st["cells"]
    if n_ctx is None and not abs_depths:
        print("  n_ctx unknown (no kv_cache_cells gauge) and --depths not given; skipping. "
              "Pass absolute --depths to test a build without the KV cache metrics.")
        return {"model": model, "error": "n_ctx unknown, need --depths"}

    ctx_s  = f"{n_ctx:,}" if n_ctx else "unknown"
    kv_s   = f"{st['kv_mib']:.0f} MiB" if st["kv_mib"] is not None else "n/a"
    vram_s = (f"{st['vram_used_mib']:.0f}/{st['vram_total_mib']:.0f} MiB used, "
              f"{st['vram_free_mib']:.0f} free") if st["vram_used_mib"] is not None else "vram n/a"
    print(f"  n_ctx {ctx_s}  kv {kv_s}  {vram_s}   (load {time.time()-t0:.0f}s, "
          f"{cpt:.2f} chars/token)", flush=True)
    print(f"  {'depth':>9} {'tokens':>9} {'prefill t/s':>12} {'gen t/s':>9} "
          f"{'vram used':>10} {'free':>8}  check", flush=True)

    # Absolute depths make models with different fitted n_ctx comparable, which
    # fractions do not: anything that costs context (a draft model, a projector) would
    # otherwise be measured at fewer tokens than the config it is being compared with.
    steps = abs_depths if abs_depths else [int(n_ctx * f) for f in fracs]
    rungs = []
    for i, want in enumerate(steps):
        # leave room for the generated tokens inside the context
        cap = (n_ctx - gen_tokens - 64) if n_ctx else None
        target = max(16, min(want, cap)) if cap else max(16, want)
        if cap and want > cap:
            print(f"  {want:>9,}  skipped: exceeds this model's n_ctx ({n_ctx:,})",
                  flush=True)
            continue
        frac = target / n_ctx if n_ctx else 0.0
        deepest = (i == len(steps) - 1)
        want_fact = recall_at_deepest and deepest
        # a recall check needs room to reason before answering
        mt = max(gen_tokens, 600) if want_fact else gen_tokens
        prompt = build_prompt(target, cpt, want_fact)
        try:
            r = srv.chat(model, prompt, mt)
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:120]
            print(f"  {frac:>8.0%} {target:>9,}  FAILED HTTP {e.code}: {body}", flush=True)
            rungs.append({"frac": frac, "error": f"HTTP {e.code}: {body}"})
            break
        except Exception as e:
            print(f"  {frac:>8.0%} {target:>9,}  FAILED {type(e).__name__}: {e}", flush=True)
            rungs.append({"frac": frac, "error": str(e)})
            break

        t = r.get("timings", {}) or {}
        st = srv.state(model)
        check = ""
        if want_fact:
            check = "recall OK" if FACT_VALUE in answer_text(r) else "RECALL WRONG"
        rungs.append({
            "frac": frac,
            "tokens": r.get("usage", {}).get("prompt_tokens"),
            "prefill_tps": t.get("prompt_per_second"),
            "gen_tps": t.get("predicted_per_second"),
            "vram_used_mib": st["vram_used_mib"],
            "vram_free_mib": st["vram_free_mib"],
            "check": check,
        })
        r_ = rungs[-1]
        used = f"{r_['vram_used_mib']:.0f}" if r_["vram_used_mib"] is not None else "-"
        freem = f"{r_['vram_free_mib']:.0f}" if r_["vram_free_mib"] is not None else "-"
        depth_s = f"{frac:>8.0%}" if n_ctx else f"{'':>8}"
        print(f"  {depth_s} {r_['tokens'] or 0:>9,} "
              f"{r_['prefill_tps'] or 0:>12.1f} {r_['gen_tps'] or 0:>9.2f} "
              f"{used:>10} {freem:>8}  {check}",
              flush=True)

    ok = [r for r in rungs if "error" not in r]
    result = {"model": model, "n_ctx": n_ctx, "kv_mib": st["kv_mib"], "rungs": rungs}
    if len(ok) >= 2 and ok[0].get("gen_tps") and ok[-1].get("gen_tps"):
        result["gen_falloff"] = ok[-1]["gen_tps"] / ok[0]["gen_tps"]
        if ok[0].get("prefill_tps") and ok[-1].get("prefill_tps"):
            result["prefill_falloff"] = ok[-1]["prefill_tps"] / ok[0]["prefill_tps"]
        print(f"  falloff shallow->deep: generation x{result['gen_falloff']:.2f}"
              + (f", prefill x{result['prefill_falloff']:.2f}"
                 if "prefill_falloff" in result else ""), flush=True)
    return result


def float_list(s):
    return [float(x) for x in s.split(",") if x]


def int_list(s):
    return [int(x) for x in s.split(",") if x]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--fracs", type=float_list, default=[0.005, 0.10, 0.50, 0.95],
                    metavar="F,F,...", help="depth fractions of the fitted n_ctx. The "
                    "first rung is deliberately near-empty: generation throughput is at "
                    "its highest there, and omitting it understates the falloff")
    ap.add_argument("--depths", type=int_list, metavar="N,N,...",
                    help="absolute token depths instead of fractions. Use this to compare "
                         "models whose fitted n_ctx differs, so each is measured at the "
                         "same token counts")
    ap.add_argument("--gen", type=int, default=32,
                    help="tokens to generate per rung, for the generation rate")
    ap.add_argument("--models", nargs="*", help="only these model ids")
    ap.add_argument("--include-cache", action="store_true",
                    help="also test models discovered in the HF cache, not just presets")
    ap.add_argument("--no-recall-check", action="store_true",
                    help="skip the planted-fact check at the deepest rung")
    ap.add_argument("--settle", type=float, default=15.0,
                    help="seconds to wait after a model load before reading VRAM, so a "
                         "swapped-out model has released its memory")
    ap.add_argument("--timeout", type=float, default=7200.0)
    ap.add_argument("--json", metavar="PATH")
    args = ap.parse_args()

    srv = Server(args.url, args.timeout)
    entries = srv.models(args.include_cache)
    ids = [m["id"] for m in entries]
    if args.models:
        ids = [i for i in ids if i in args.models]
        missing = set(args.models) - set(ids)
        if missing:
            print(f"warning: not found or not configured: {', '.join(sorted(missing))}")
    if not ids:
        sys.exit("no models to test (configured models are those with source=preset)")

    print(f"testing {len(ids)} model(s): {', '.join(ids)}")
    results = [bench_model(srv, m, args.fracs, args.gen,
                           not args.no_recall_check, args.settle, args.depths)
               for m in ids]

    print(f"\n=== summary ===")
    print(f"{'model':<32} {'n_ctx':>9} {'kv MiB':>8} {'prefill t/s':>12} "
          f"{'gen t/s':>9} {'gen falloff':>12} {'free MiB':>9}  status")
    for r in results:
        if "error" in r:
            print(f"{r['model']:<32} {'-':>9} {'-':>8} {'-':>12} {'-':>9} {'-':>12} "
                  f"{'-':>9}  {r['error']}")
            continue
        ok = [x for x in r["rungs"] if "error" not in x]
        bad = [x for x in r["rungs"] if "error" in x]
        deep = ok[-1] if ok else {}
        status = "ok"
        if bad:
            status = "FAILED at %.0f%%: %s" % (bad[0]["frac"] * 100, bad[0]["error"][:40])
        elif any(x.get("check") == "RECALL WRONG" for x in ok):
            status = "RECALL WRONG"
        ctx_s  = f"{r['n_ctx']:,}" if r.get("n_ctx") else "-"
        kv_s   = f"{r['kv_mib']:.0f}" if r.get("kv_mib") is not None else "-"
        free_s = (f"{deep['vram_free_mib']:.0f}"
                  if deep.get("vram_free_mib") is not None else "-")
        print(f"{r['model']:<32} {ctx_s:>9} {kv_s:>8} "
              f"{deep.get('prefill_tps') or 0:>12.1f} {deep.get('gen_tps') or 0:>9.2f} "
              f"{('x%.2f' % r['gen_falloff']) if 'gen_falloff' in r else '-':>12} "
              f"{free_s:>9}  {status}")
    print("\nprefill/gen t/s and free MiB are at the deepest successful rung.")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
