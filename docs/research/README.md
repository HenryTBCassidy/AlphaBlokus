# `docs/research/` — Reference material

Reference notes that aren't part of the project's planning or runbook tracks. The two reasons this directory exists:

1. **F2 (bitboard move-generation) needs grounded design decisions.** The technique is decades-old and well-understood; the choices that matter are specific to our game (14×14 Blokus Duo) and our environment (Python on a single GPU). Documenting both gives the F2 implementation a stable foundation rather than reinventing wheels.

2. **Pentobi is our benchmark target.** Understanding how Pentobi plays — its data structures, its move-gen algorithm, its evaluation heuristics, what positions it does and doesn't handle well — is reusable beyond just F2. The longer-term hope is that systematic gaps in Pentobi's play (if any exist) inform self-play training: positions where Pentobi is weakest become positions our agent deliberately practises.

This directory is meant to grow. New research notes drop in alongside the existing ones; cross-link with `[[name]]` syntax or markdown links.

## Contents

| Doc | What's in it |
|---|---|
| [`bitboards.md`](bitboards.md) | Generic bitboard concept — what they are, why they're fast, worked examples on TicTacToe / chess / 14×14 Blokus, the padding-bits trick, mirroring/rotating, where Python is harder. **Read first** if you're new to the technique. |
| [`pentobi/`](pentobi/) | Pentobi-specific research. Their architecture, the move-list precomputation that replaces bitboards in their design, the move-generation algorithm, and what's hard to port to Python. See `pentobi/README.md` for the sub-index. |
| [`policy-head-architecture.md`](policy-head-architecture.md) | The current Blokus Duo network's policy head is a fully-connected layer holding ~95% of total parameters. Notes on why (AGZ inheritance), why this is a poor fit for our action space (Chess-like, not Go-like), what a fully-convolutional alternative would look like (~1200× fewer params), and why "fewer params" likely means *better* training rather than worse here. Written assuming a physics background, not deep ML literacy. |

## Key finding from the first research pass (2026-05-27)

The Pentobi research surfaced a result that reframes the F2 optimisation: **Pentobi does not use bitboards.** Their speed comes from a precomputed move-list table keyed by `(piece, point, adj_status)` that returns ready-made candidate move IDs at run time. The per-move legality check is then a 5-element unrolled byte-OR — no bitwise tricks. See [`pentobi/move-generation.md`](pentobi/move-generation.md) for details and citations.

This means the F2 plan has a real design choice to make:

- **Option 1: port Pentobi's design** (precomputed move-list table + byte arrays). Proven approach, matches the strongest existing Blokus engine, ~5-10 MB of static tables, no bit-twiddling.
- **Option 2: build a true bitboard for Blokus** (~196 bits per state plane, legality via AND of precomputed footprint/contact/corner masks). More speculative — Pentobi presumably evaluated it and chose otherwise — but potentially faster if the bit-ops aren't a Python bottleneck.

The F2 sub-plan (when we draft it) needs to either justify Option 2 vs Pentobi's measured design choice, or default to Option 1.

## What does *not* belong here

- **Project plans / in-flight work** — `docs/plans/`.
- **Operational runbooks** — `docs/guides/` (e.g. remote training, code style).
- **Algorithm / architecture explanations of *our* code** — top-level `docs/02-ALGORITHMS.md`, `docs/03-NEURAL-NETWORKS.md`, etc.
- **External commentary on AlphaZero or other published work that doesn't intersect with our implementation choices** — keep that in your notebook, not here.

The bar for adding a research doc is "non-trivial external reference I'd otherwise have to re-derive from scratch later." Quick "this paper looks interesting" notes belong in your reading list, not in the repo.
