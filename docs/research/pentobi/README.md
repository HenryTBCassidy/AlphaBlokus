# Pentobi research notes — overview

These four documents capture what is in Pentobi's source tree (the strongest
open-source Blokus engine) that is relevant to AlphaBlokus' F2 optimisation
work. Source studied: `git clone --depth 1 https://github.com/enz/pentobi.git`,
version "31.0-dev" per `CMakeLists.txt:4-7`.

The most important finding up front: **Pentobi does not use a positional
bitboard.** It uses precomputed move lists keyed on
`(piece, point, adjacent-status)` plus byte-per-cell grids. The "bitboard"
framing in our F2 plan is therefore a poor model of what Pentobi actually does.
See [move-generation.md](move-generation.md) for the details.

## Project facts

- **Maintainer:** Markus Enzenberger (`README.md:21`, `LICENSE.md:1`).
- **License:** GPL-3.0-or-later (`README.md:25-39`). All snippets in this
  research are short illustrations; no large block copies.
- **Status:** Maintenance mode, no new features planned (`CONTRIBUTING.md:6-10`).
- **Language and standard:** C++20 (`CMakeLists.txt:23`).
- **Build system:** CMake >= 3.22 (`CMakeLists.txt:1`).
- **Hard dependencies:** Threads (`CMakeLists.txt:39`). GUI build adds Qt6
  Concurrent/LinguistTools/QuickControls2/Xml and gettext + rsvg-convert
  (`CMakeLists.txt:59-72`). **No Boost.**
- **Game variants supported:** Classic (20x20), Duo (14x14), Trigon (9 and 8
  hex sides), Junior, Nexos, Callisto (in 2/3/4-player flavours), GembloQ
  (in 2/3/4-player flavours). Selection lives in `libpentobi_base/Variant.cpp:89-127`.

## High-level architecture

Per `CONTRIBUTING.md:18-58`, the tree splits into reusable "boardgame"
libraries and Pentobi-specific "pentobi" libraries. Approximate line counts
(headers + sources, excluding tests):

| Area                         | LOC      | Role                                            |
| ---------------------------- | -------- | ----------------------------------------------- |
| `libboardgame_base/`         | ~6,300   | Generic geometry, Grid, Marker, transforms, IO  |
| `libboardgame_mcts/`         | ~2,600   | Abstract MCTS (SearchBase, Tree, Node, LGR)     |
| `libpentobi_base/`           | ~10,100  | Blokus rules: Board, BoardConst, pieces, moves  |
| `libpentobi_mcts/`           | ~4,600   | Blokus playing engine (State, PriorKnowledge…)  |
| `pentobi/` (Qt6 GUI)         | ~6,200   | Desktop GUI (skimmed only)                      |
| `libpentobi_gtp/`, `pentobi_gtp/`, `twogtp/`, `learn_tool/` | small | GTP wrapper, tools |

Game logic and the search engine are the relevant parts for AlphaBlokus. The
GUI and GTP layers are not.

## Where to look next

- [architecture.md](architecture.md) — directory and class walkthrough; how a
  game is initialised and played; build-time data layout.
- [move-generation.md](move-generation.md) — the headline
  doc. Storage of board state, piece-orientation data, the legality check,
  move enumeration with the "attach point" anchor, symmetry and scoring.
- [python-port-gotchas.md](python-port-gotchas.md) — the things in Pentobi's
  design that translate badly to Python, plus where there is an obvious
  CPython equivalent.
