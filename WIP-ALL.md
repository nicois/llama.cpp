# `wip-all` — working branch guide

This file exists only on `wip-all`. It must be dropped from any upstream-bound branch.

## What this branch is

`wip-all` is a private integration branch: a melange of unrelated workstreams that happen to be
deployed together, not a coherent feature. Nothing here is ever merged upstream as a whole.
Individual commits are carved off onto fresh branches cut from `origin/master`.

The branch is maintained as **one commit per intended PR**. That is the organising rule: it keeps
each candidate self-contained, makes overlap between them visible, and lets each commit message
serve as that unit's description. Consequently:

- **Read `git log` before anything else.** The commit messages are the primary documentation,
  including whether a commit is PR-ready, what was measured, and what is deliberately unfinished.
  This file covers only what a single commit cannot say.
- Amendments are squashed back into the owning commit rather than stacked as fixups.
- The branch is rebased onto `origin/master`, never merged, and force-pushed to `nicois/wip-all`.

Check how far behind the branch is with `git log --oneline $(git merge-base wip-all origin/master)..origin/master`,
and re-align with `git rebase origin/master` — after re-deriving the mirrored commit described below.

## Commit status classes

The commit body is authoritative. This is the summary:

| Commit | Status |
|---|---|
| `sycl: opt-in for detailed memory allocations` | **Mirror** of PR #27631 (draft) — see below |
| `sycl: weight-reorder host staging` | Not PR-ready; only the use-after-free is upstream-relevant, and it needs rewriting against upstream rather than cherry-picking |
| `sycl: reserve mul_mat conversion scratch` | Not PR-ready; f32 and bf16 paths uncovered |
| `sycl: route quantized-KV decode to TILE on Battlemage` | Candidate; the toolchain gate needs agreement upstream |
| `devops: build the SYCL image with oneAPI 2026.1` | Deployment; overlaps the docs commit below |
| `test-backend-ops: cover quantized KV cache at head size 128` | Small and standalone; closest to submittable |
| `sycl: reduce redundant work in Q4_K multi-column MMVQ` | Retained pending triage; +2% is inside run-to-run variance, so not currently intended as a PR |
| `server: cache-aware task scheduling` | Not PR-ready; see `docs/superpowers/2026-08-25-cache-aware-sched-pr-readiness.md` |
| `server: --slot-linger-ms` | Closer to ready, but **depends on** the cache-aware commit |
| `scripts:`, `deploy:`, and all four `docs:` commits | Local only; not upstream material |

Cross-commit couplings to preserve when reordering:

- `--slot-linger-ms` sits on top of cache-aware scheduling and does not stand alone.
- `docs: SYCL oneDNN is packaged separately…` documents two *different* code commits (the 2026.1
  image and the TILE routing). It is deliberately its own commit so the overlap stays visible; it
  needs splitting or assigning before either becomes a PR.
- The cache-aware design and plan documents must be excluded from any upstream-bound branch. They
  are kept in a separate commit from the scheduling code precisely so they can be dropped without
  touching it.

## Mirrored commits

`sycl: opt-in for detailed memory allocations` is a **mirror** of the PR branch
`pr-sycl-memtrace`, not original work on this branch. Do not edit it here — edits belong on the PR
branch, and a hand-maintained second copy silently diverges, which then blocks the rebase when the
PR merges.

Re-derive it instead, as the first step of any rebase:

```sh
git rebase --onto origin/master <mirror-commit> wip-all   # drop the stale mirror
git cherry-pick nicois/pr-sycl-memtrace                   # re-apply the current one
```

When the PR merges upstream, drop the mirror and take no replacement — `origin/master` provides it.
The same treatment applies to any future in-flight PR that this branch needs at runtime.

Upstream history worth knowing: the FA-staging reservation (PR #27629) was merged as `cc83d7b48`
and its commit was dropped from this branch. Issue #27595 remains open for the memory `--fit` still
does not account for; the `mul_mat` scratch commit here addresses the next instance of that same
class of problem.

## Build

```sh
source /opt/intel/oneapi/setvars.sh
cmake -B build-sycl -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL -DGGML_SYCL_DNN=ON \
      -DGGML_SYCL_GRAPH=ON -DGGML_NATIVE=OFF -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake --build build-sycl -j$(nproc)
```

`--target ggml-sycl` alone is enough to check that backend changes compile, and is much faster
than a full build.

oneDNN is **not** bundled in the oneAPI Deep Learning Essentials image from 2026.1 onward. If it is
missing, `find_package(DNNL)` only emits a STATUS message and the build silently loses the oneDNN
GEMM and flash-attention paths — so a build that succeeds is not by itself evidence those paths are
present. `.devops/intel.Dockerfile` on this branch installs it explicitly and fails loudly instead.

## Tests

```sh
cmake --build build --target test-server-sched test-server-sched-sim
./build/bin/test-server-sched        # unit tests over the sched_* ranking functions
./build/bin/test-server-sched-sim    # discrete-event simulator driving the real sched_* functions
```

Both are added by the cache-aware scheduling commit and link `server-context`. A CPU-only `build/`
is sufficient for them.

Server integration tests:

```sh
cd tools/server/tests && ./tests.sh unit/test_cache_aware_sched.py unit/test_slot_linger.py
```

The fixtures in `utils.py` download models from Hugging Face. Where that is blocked, point the
fixture at a local file instead — set `server.model_file` to any small gguf and set
`server.model_hf_repo` and `server.model_hf_file` to `None`. Previously downloaded models are
cached in `tools/server/tests/tmp/`, which is not tracked, so it will be empty on a fresh clone.

`test-backend-ops` on SYCL has pre-existing failures unrelated to this branch: several `MUL_MAT`
bf16 cases, and exactly one `FLASH_ATTN_EXT` case (hsk=256, q8_0 K/V, permute=[0,2,1,3]) which is
the bug written up in `docs/development/sycl-mkl-fa-permuted-quantized-bug.md`. Confirm any
suspected regression against unmodified `master` before attributing it to a change here.

## Hardware, logging and deployment

Two very different SYCL devices are in play, and results do not transfer between them:

- **Arc Pro B70** (Battlemage, BMG G31), 32,656 MiB dedicated — the machine that matters for
  memory-fit and performance work. Runs `llama-server` under podman.
- A local **Arrow Lake iGPU** (Xe-LPG) with unified memory — useful for compile and functional
  checks only. Its "device memory" is system RAM, so VRAM totals, `--fit` behaviour and OOM
  thresholds are meaningless as proxies for the B70. Some diagnostics deliberately behave
  differently there: the memory-tracking commit withholds its `other` figure on non-dedicated
  memory because the number would not mean what it says.

Logging gotchas that cost real time if forgotten:

- ggml backend `GGML_LOG_INFO` output requires common verbosity **4** — `-lv 4`, or
  `LLAMA_ARG_LOG_VERBOSITY=4`. Without it, `GGML_SYCL_*` diagnostics appear to do nothing even
  though they are enabled.
- `--fit` is on by default, so any comparison of VRAM used is meaningless unless `--ctx-size` is
  pinned: the fit logic will otherwise silently resize the context to consume whatever is free.
- Benchmark comparisons must toggle behaviour on a **single binary** via environment variable.
  Comparing two separately configured builds has repeatedly produced spurious results here.

`deploy/journal-grep.py` provides a read-only HTTP endpoint over `journalctl`, so logs can be
pulled from the GPU host without an interactive shell. Its address and token are configured
out-of-band and are deliberately not stored in this repository — ask for them.

## Upstream policy

`AGENTS.md` and `CONTRIBUTING.md` govern anything sent upstream. The parts that bind hardest:

- **AI-written PR descriptions, commit messages and reviewer responses are prohibited** and result
  in immediate PR closure (`AGENTS.md`, "Prohibited AI Usage"). For upstream-bound work, an agent
  may supply evidence, measurements and draft wording as raw material, but the human writes the
  text that is posted.
- `AGENTS.md` states **"Private forks are exempt."** That is why the commit messages on this branch
  may be agent-written: they are working notes for the next session, not upstream submissions. They
  do not survive into a PR unrewritten.
- Do not post comments on upstream issues or PRs from an agent session.

## Files that must never go upstream

`WIP-ALL.md`, the `CLAUDE.md` pointer to it, `docs/superpowers/**`, `deploy/journal-grep.py`, and
`scripts/server-context-bench.py`, `scripts/server-vram-probe.py`, `scripts/sycl-vram-hog.cpp`.
The `docs/development/sycl-*.md` notes are working records; they may become issue material but are
not PR content as they stand.
