#!/usr/bin/env python3

r"""Tiny read-only HTTP endpoint that greps journalctl, for use on the GPU host.

Exists so an operator (or an agent) working from another machine can pull
llama-server log lines -- memory breakdowns, FA dispatch traces, OOM backtraces
-- without an interactive shell on the box.

Run on the GPU host:

    ./deploy/journal-grep.py --port 8099 --tag llama-b70-server

Then from anywhere on the LAN:

    curl -sG http://192.168.0.23:8099/grep \
        --data-urlencode 'p=memory breakdown|SYCL0 \(Intel' \
        --data-urlencode 'since=20 min ago' \
        --data-urlencode 'after=2'

Query parameters for /grep:

    p          required, Python regex matched against each line
    since      journalctl --since value (default: "30 min ago")
    until      journalctl --until value (optional)
    tag        syslog identifier, overrides --tag (validated charset)
    unit       systemd unit instead of a tag (validated charset)
    after      lines of trailing context per match (default 0, max 20)
    before     lines of leading context per match (default 0, max 20)
    max        cap on returned lines (default 500, max 5000)
    i          set to 1 for a case-insensitive match
    token      required iff the server was started with --token

Endpoints: /grep, /health.

SECURITY: read-only and shell-free -- journalctl is invoked with an argv list,
never a shell string, and the pattern is applied in Python, so it cannot reach a
command interpreter. It is nonetheless an UNAUTHENTICATED window onto your
journal if you leave --token unset. Bind to a LAN address you trust, use --token,
and stop it when you're done.
"""

import argparse
import json
import re
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

# journalctl -t/-u values: keep to a conservative charset so a value can never
# be mistaken for a flag or smuggle anything odd into argv.
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.@:-]{1,128}$")

MAX_CONTEXT = 20
MAX_LINES = 5000
DEFAULT_LINES = 500

ARGS = None


def clamp(value, default, hi, lo=0):
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return default


def read_journal(tag, unit, since, until):
    cmd = ["journalctl", "--no-pager", "--output=short-iso"]
    if unit:
        cmd.append(f"--unit={unit}")
    else:
        cmd.append(f"--identifier={tag}")
    cmd.append(f"--since={since}")
    if until:
        cmd.append(f"--until={until}")
    # shell=False: `cmd` is an argv list, so none of these values are parsed by
    # a shell. The =-joined forms also stop a leading '-' becoming a new flag.
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or f"journalctl exit {p.returncode}")
    return p.stdout.splitlines()


def grep(lines, pattern, ignorecase, before, after, limit):
    flags = re.IGNORECASE if ignorecase else 0
    rx = re.compile(pattern, flags)

    keep = set()
    hits = 0
    for i, line in enumerate(lines):
        if rx.search(line):
            hits += 1
            for j in range(max(0, i - before), min(len(lines), i + after + 1)):
                keep.add(j)

    idx = sorted(keep)
    out, prev = [], None
    for i in idx[:limit]:
        if prev is not None and i != prev + 1:
            out.append("--")
        out.append(lines[i])
        prev = i
    return out, hits, len(idx)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):          # one concise line per request
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % a))

    def _send(self, code, body, ctype="text/plain; charset=utf-8"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}

        if u.path == "/health":
            return self._send(200, json.dumps({
                "ok": True, "tag": ARGS.tag, "auth": bool(ARGS.token)}) + "\n",
                "application/json")

        if u.path != "/grep":
            return self._send(404, "not found: try /grep or /health\n")

        if ARGS.token and q.get("token") != ARGS.token:
            return self._send(403, "bad or missing token\n")

        pattern = q.get("p")
        if not pattern:
            return self._send(400, "missing required parameter 'p'\n")

        tag, unit = q.get("tag", ARGS.tag), q.get("unit")
        for label, value in (("tag", tag), ("unit", unit)):
            if value is not None and not SAFE_NAME.match(value):
                return self._send(400, f"invalid {label}\n")

        before = clamp(q.get("before"), 0, MAX_CONTEXT)
        after = clamp(q.get("after"), 0, MAX_CONTEXT)
        limit = clamp(q.get("max"), DEFAULT_LINES, MAX_LINES, lo=1)

        try:
            rx_lines = read_journal(tag, unit, q.get("since", "30 min ago"),
                                    q.get("until"))
        except subprocess.TimeoutExpired:
            return self._send(504, "journalctl timed out\n")
        except (RuntimeError, OSError) as e:
            return self._send(502, f"journalctl failed: {e}\n")

        try:
            out, hits, total = grep(rx_lines, pattern,
                                    q.get("i") == "1", before, after, limit)
        except re.error as e:
            return self._send(400, f"bad regex: {e}\n")

        header = (f"# {hits} matching lines, {total} lines with context, "
                  f"showing {len(out)} (scanned {len(rx_lines)})\n")
        return self._send(200, header + "\n".join(out) + ("\n" if out else ""))


def main():
    global ARGS
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bind", default="0.0.0.0", help="default: %(default)s")
    ap.add_argument("--port", type=int, default=8099, help="default: %(default)s")
    ap.add_argument("--tag", default="llama-b70-server",
                    help="default syslog identifier (default: %(default)s)")
    ap.add_argument("--token", help="if set, requests must pass ?token=...")
    ARGS = ap.parse_args()

    if not SAFE_NAME.match(ARGS.tag):
        sys.exit("error: --tag has an unexpected charset")

    srv = ThreadingHTTPServer((ARGS.bind, ARGS.port), Handler)
    print(f"journal-grep on http://{ARGS.bind}:{ARGS.port}  tag={ARGS.tag}  "
          f"auth={'token' if ARGS.token else 'NONE'}", flush=True)
    if not ARGS.token:
        print("warning: unauthenticated; anyone on the network can read this "
              "journal. Use --token and stop the server when finished.",
              flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping", flush=True)


if __name__ == "__main__":
    main()
