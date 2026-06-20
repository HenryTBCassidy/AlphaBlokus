# Pentobi Harness — Benchmark our net against Pentobi

Build the long-planned **Pentobi adapter + benchmark** so we can finally measure absolute
strength against the strongest open-source Blokus AI — the project's actual goal ("beat
Pentobi level 9 in a majority of 100 games"). This *executes* the design already scoped in
**[06-INTERFACES.md](../06-INTERFACES.md)** (translation layer + GTP adapter, with the build
recipe and gotchas) against the metrics defined in **[05-EVALUATION.md](../05-EVALUATION.md)**
(Pentobi Level / Score / Weighted Score / per-level profile + the heatmap). Read both before
implementing — this plan is the *execution checklist*, not a re-derivation.

**Goal of this branch:** a configurable command that pits a chosen net (MCTS) against
`pentobi-gtp` and reports results, so we can test the `run1` checkpoints tomorrow. It must
support:
- **a custom net** — point it at any checkpoint (`--net <path>`);
- **a single Pentobi difficulty** (`--level N`) *or* a **full sweep** of levels 1–9 (`--sweep`);
- **configurable game count** (`--games N`, split half/half by colour);
- a **report with simple stats** (who won how many, per level/colour, win rate + CI, the
  Pentobi Level/Score/Weighted metrics) **and viewable game replays** rendered the *same way*
  as training-run arena replays.

Plus (final step H6) the "as-we-go" integration: periodically pit the *current* net against
Pentobi during a training run.

**Reuse, don't reinvent (the core design choice).** The training code already has everything
except the Pentobi connection:
- `core/arena.py` — `Arena.play_games(..., record=True)` plays any two `core.players.Player`
  objects and emits `GameRecord`/`MoveRecord` (board cross-validation, colour alternation,
  pass/end handling — all done).
- `reporting/` + `scripts/replay.py` — already render `GameRecord`s (from `ArenaReplays/`
  parquet) into the viewable replays in the training report.

So the **only genuinely new code** is the Pentobi connection: the translation layer (H1), the
GTP subprocess adapter (H3), and a thin **`PentobiPlayer(Player)`** wrapper (H4). We wrap
Pentobi as a `Player`, hand it + a `NetworkPlayer` to the *existing* `Arena`, write the
resulting records to an `ArenaReplays`-style parquet, and the *existing* renderer shows the
games. Consistent rendering falls out for free.

**Out of scope:** the Pygame/Web UI (06-INTERFACES Phase C) — separate, not needed for
benchmarking.

**Correctness discipline (the F2 lesson):** the translation layer is the error-prone part and
a swapped colour/coordinate mapping silently flips every result. Two guards: (a) round-trip
tests on the translation layer, and (b) **board-state cross-validation** every move — our
board and Pentobi's `showboard` must agree, or it's a bug.

---

## Checklist

| # | Item | Effort | Priority | Done |
|---|------|--------|----------|------|
| H1 | **Translation layer** — coord converter + `Action`↔cell-list + board text renderer; round-trip tests over all 91 orientations | ~2.5 hr | High | ✅ |
| H2 | **Build `pentobi-gtp` + verify the GTP interface** — Duo variant, colour mapping, command set — against the *real* binary (pins the 06-INTERFACES ⚠️s) | ~1 hr | High | ✅ |
| H3 | **GTP subprocess adapter** — spawn / send / read-to-double-newline / parse `=`/`?` / close; handle buffering, stderr drain, `pass` | ~1 hr | High | |
| H4 | **`PentobiPlayer(Player)` + reuse the `Arena`** — wrap the GTP adapter as a `Player` so net-vs-Pentobi runs through the *existing* `Arena.play_games(record=True)`; per-move board cross-validation | ~1 hr | High | |
| H5 | **Benchmark CLI + report (stats + replays)** — `--net`, `--level` or `--sweep`, `--games`; reuse Arena → `GameRecord`s → `ArenaReplays`-style parquet; stats + Pentobi metrics + **replays via the existing renderer** | ~2 hr | High | |
| H6 | **(As-we-go) periodic Pentobi eval in the training cycle** — current net vs Pentobi every *N* generations, logged to W&B alongside Elo/arena | ~2 hr | Medium | |

H1–H5 deliver the standalone benchmark (the immediate need). H6 is the training-loop
integration you asked about; it builds on H1–H5 and can follow once they're solid.

---

## H1. Translation layer

Per [06-INTERFACES.md §1](../06-INTERFACES.md). Build, with `ActionCodec` reused (already
exists), in a new module (e.g. `games/blokusduo/pentobi_translation.py`):
- **Coordinate converter** AlphaBlokus `(x,y)` ↔ Pentobi `<letter><number>` (a1–n14). Both
  bottom-left origin; only letters-vs-numbers + 0-vs-1 indexing differ.
- **`Action` → cell list** — apply the piece shape at the position, emit occupied cells as a
  comma-joined Pentobi string (Pentobi moves are *just* the cells, no piece/orientation).
- **cell list → `Action`** — the hard direction: parse cells, match the pattern against the 91
  orientations, recover `(piece_id, orientation, anchor)`. Needs care; test thoroughly.
- **Board text renderer** — `(14,14)` array → the a–n / 1–14 grid (debugging + cross-validation).

**Tests:** round-trip every legal `Action` → cells → `Action` (identity) over all 91
orientations and a spread of positions; and a `pass` round-trips. This is the safety net.

## H2. Build `pentobi-gtp` + verify the interface

Build on the **box** (Linux, has the toolchain) per [06-INTERFACES.md §2 "Building Pentobi"]
(`cmake -DPENTOBI_BUILD_GTP=ON -DPENTOBI_BUILD_GUI=OFF`). Then **empirically pin the things
06-INTERFACES flags as unverified** against the real binary:
- the exact invocation / variant selection for **Blokus Duo** (`--game duo` and/or `set_game`);
- the **colour letters and first-mover mapping** (our White=+1 starts at (4,4) → which GTP
  colour?) — confirm via `showboard`/`genmove`, because a swap silently flips every benchmark;
- the command set we rely on (`clear_board`, `play`, `genmove`, `showboard`, `final_score`, `quit`).

Record the verified facts in the adapter's docstring (supersedes the doc's guesses).

### ✅ Verified interface (2026-06-21, Pentobi v31.0, built on the Mac)

Binary: `~/code/pentobi/build/pentobi_gtp/pentobi-gtp` (built locally — CMake via `uv tool`,
GUI off, Release; *not* in the repo). Launch: `pentobi-gtp --game duo --level N --quiet
[--threads 1] [--seed S]`.

- **Colour mapping (the ⚠️, now pinned by the start squares):** our **White (+1) ↔ `b`**
  (Pentobi "Purple", starts **E10** = our `white_start` array (4,4)); our **Black (−1) ↔ `w`**
  (Pentobi "Orange", starts **J5** = our `black_start`). **The naive White→`w` guess is WRONG** —
  `reg_genmove b` played on e10, `reg_genmove w` on j5. Valid colour tokens are only `b`/`w`
  (`blue`/`white` also accepted); `x`/`o`/`purple`/`g`/`p` are rejected.
- **Board / coords:** 14×14, columns `a`–`n`, rows 1–14, bottom-left origin — matches our
  translation (H1) exactly. Moves are comma-separated **lowercase** cells, e.g. `h7,g8,h8,f9,g9`;
  pass is `pass`.
- **Commands confirmed:** `clear_board`, `play <c> <cells>`, `genmove <c>`, `reg_genmove <c>`
  (generate w/o committing), `showboard`, `final_score`, `all_legal <c>`, `undo`, `quit`.
- **`final_score` format:** `B+N` / `W+N` (margin), where **`B` = our White (+1)**, **`W` = our
  Black (−1)** — same b/w mapping. Parse accordingly.

## H3. GTP subprocess adapter

Per [06-INTERFACES.md §2]. `PentobiAdapter`: spawn `pentobi-gtp` (`bufsize=0`), `send_command`
(write+flush, read until `\n\n`, parse `=`/`?`, raise on error), `close` (`quit` + terminate).
Gotchas to handle: **flush after every write**; **read to the double-newline** (not one line);
**drain stderr or use `--quiet`** (else deadlock); `genmove` may return `pass`.

## H4. `PentobiPlayer(Player)` + reuse the `Arena`

The `core.players.Player` interface is just `__call__(board: IBoard) -> int` (action index).
So `PentobiPlayer` is thin: on `__call__`, send the current position's last move to `pentobi-gtp`
(`play`), ask `genmove`, translate the returned cells → `Action` → action index (via H1), and
keep its own GTP game in sync. Then net-vs-Pentobi is just:

```
Arena(NetworkPlayer(net), PentobiPlayer(level=L), game).play_games(n, record=True)
```

— reusing the *existing* Arena loop, colour alternation, and `GameRecord` capture. **Board
cross-validation:** assert our game board == Pentobi's `showboard` after each move (cheapest
place to catch a translation/colour bug). *(IDEAS I2: at eval we can afford a stronger/exact
search than training — more sims, K=1, no Dirichlet — since we care about strength, not
throughput; set the `NetworkPlayer`/`MCTSConfig` accordingly.)*

## H5. Benchmark CLI + report (stats + viewable replays)

A committed script, e.g. `scripts/pentobi_benchmark.py`, no ad-hoc args:

```bash
uv run python -m scripts.pentobi_benchmark --net <checkpoint> --level 5 --games 100
uv run python -m scripts.pentobi_benchmark --net <checkpoint> --sweep --games 100   # levels 1–9
```

- Plays via the reused `Arena` (one single-threaded `pentobi-gtp` per concurrent game — >8
  threads degrades Pentobi), half/half by colour.
- **Persists `GameRecord`s to an `ArenaReplays`-style parquet** so the existing
  `reporting/` + `scripts/replay.py` render the games **identically to training arena replays**
  (the consistency you want — same board renderer, same replay viewer).
- **Report:** simple stats (wins/losses/draws by level and colour, win rate + 95% CI) plus the
  [05-EVALUATION §2](../05-EVALUATION.md) headline metrics (**Pentobi Level**, **Score**,
  **Weighted Score**, **per-level profile**); on a sweep, the levels×winrate **heatmap**. Built
  with the same `reporting/` infra as the training report so charts/boards look consistent.

First use: point `--net` at the best accepted `run1` checkpoint and sweep levels 1–9.

## H6. (As-we-go) periodic Pentobi eval in the training cycle

The "arena battles vs Pentobi as we go": every *N* generations (05-EVALUATION suggests 10–20),
play the current net vs Pentobi (a subset of levels, fewer games for cheapness) and log the
result to W&B beside the existing Elo-vs-baseline and arena metrics — so we watch absolute
strength climb during the run, not just after. Gated by a config flag (off by default; the
standalone H5 benchmark stays the rigorous 100-game measurement). Builds on H1–H5.
