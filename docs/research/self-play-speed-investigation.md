# Self-Play Speed Investigation — Profiling Plan + Technique Menu

**Status: for review (decide what to pursue next).** This is a candidates-and-plan
doc, not committed work. Companion to [`../plans/full-cycle-optimisation.md`](../plans/archive/full-cycle-optimisation.md)
(F1–F5) and [`../IDEAS.md`](../IDEAS.md).

## Why this exists

F5 (the cross-worker inference server) was built, proven bit-identical, and
benchmarked at **0.99× — no speedup**. The reason is the headline finding for
*everything* below: **after F4 shrank the net 21.6× (→340K params), GPU
inference is cheap and self-play is CPU-bound.** The ~30% GPU utilisation we
measured is *idle-waiting-on-CPU*, not a starved GPU. So further speed has to
come from the **CPU side**: the Python that runs the MCTS tree search and the
legal-move generation, once per simulation, tens of thousands of times per game.

The mistake we must not repeat: F5 was built on a *stale* profile (the F3-era
measurement that inference was 77.5%, taken on the old big net). **So step one
is always: measure the current profile. Then optimise the biggest real slice.**

---

## First, the thing you asked about: Amdahl's law

Amdahl's law is the ceiling on how much faster the *whole* program gets when you
speed up *one part* of it. If a part takes fraction **p** of total runtime and
you speed that part up by factor **s**, the overall speedup is:

```
total_speedup = 1 / ( (1 - p) + p/s )
```

The key consequence: **even infinitely fast (s → ∞), the most you can ever get
is `1 / (1 - p)`.** You can never save more than the time that part was using.

Worked examples for self-play (illustrative — real `p` comes from profiling):

| If this slice is… | …making it **2× faster** gives | …making it **infinitely fast** gives |
|---|---|---|
| 50% of runtime | 1.33× total | 2.0× total (ceiling) |
| 30% of runtime | 1.18× total | 1.43× total (ceiling) |
| 10% of runtime | 1.05× total | 1.11× total (ceiling) |

This is why I kept saying things were "Amdahl-limited": optimising a 10% slice
to perfection buys at most 1.11×, no matter how clever the trick. It's also why
**the profile must come first** — it tells us which slice is big enough to be
worth attacking, and what the ceiling is before we spend effort. (It's also why
move-gen alone, once F2 cut it to ~9%, stopped being worth a Cython rewrite —
the ceiling was ~1.1×.)

---

## Part A — The profiling investigation (do this first)

**Goal:** a trustworthy, current breakdown of where a self-play game's wall-clock
goes, at two levels: (1) coarse, per MCTS phase; (2) fine, per line in the hot
functions. Plus a memory profile, since the search tree grows unbounded within
an episode.

### What we already know the hot loop looks like

Per MCTS simulation (×300 sims × ~28 moves × N games), the work is:

1. **Descend to a leaf** (`MCTS._descend_to_leaf`): repeatedly
   - `state_key` — `board.tobytes()` (196 B) + dict lookup,
   - `_select_action` — **a Python `for a in np.where(valids)[0]` loop** computing
     a UCB score with `math.sqrt` for *every legal move* (hundreds, early-game),
   - `get_next_state` — board copy + place piece.
2. **Move generation** (`valid_move_masking`, F2 table path) at new leaves.
3. **Batched NN eval** (`predict_batch`) — now cheap (small net).
4. **Backprop** up the path.

`_select_action`'s per-move Python loop is the standout suspect: it's pure
interpreted Python, runs on every descent at every node, and scales with the
(large) Blokus branching factor. But suspicion isn't measurement — that's what
Part A confirms.

### Tools (and why each)

| Tool | Role here | Notes |
|---|---|---|
| **py-spy** | First look — sampling profiler, **handles multiprocessing** (`--subprocesses`), near-zero overhead, no code changes. Gives a flame graph of where real time goes across the 8 workers. | `cProfile` does **not** work well across our `ProcessPoolExecutor` workers — py-spy is the right coarse tool. |
| **Scalene** | Line-level CPU truth, and crucially **splits Python time vs native (numpy/torch) time vs system time**, plus per-line memory. Run on a **single-worker** self-play run for clean attribution. | Tells us if a line's cost is Python-interpreter overhead (→ Cython/Numba helps) or already-native numpy (→ won't). |
| **line_profiler** | Targeted line timings on the specific hot functions py-spy/Scalene flag (`_select_action`, `_descend_to_leaf`, `get_next_state`, `state_key`). Deterministic, exact. | `@profile` decorator on the chosen functions; single-process run. |
| **tracemalloc** | Memory growth of the search tree within an episode (the per-episode dicts that grow 25→50→75…). Confirms whether long games / many sims risk memory pressure at scale. | Already flagged in `scaled-training-run.md`; quantify it. |

### Proposed steps

| # | Step | What it answers |
|---|------|-----------------|
| P1 | Add a `scripts/profile_self_play.py` that plays *one* self-play game single-process (fixed seed, production net + sims) under a `--profiler {pyspy,scalene,lineprofiler,tracemalloc}` switch. | Reproducible, isolated target — no multiprocessing noise. |
| P2 | **py-spy** on an 8-worker self-play run (`--subprocesses`, ~30s sample) → flame graph. | The honest coarse split across real parallel execution: % in select / move-gen / get_next_state / inference / backprop. |
| P3 | **Scalene** on the single-game script. | Per-line CPU, **Python-vs-native split**, and memory — pinpoints interpreter-bound lines (the optimisable ones). |
| P4 | **line_profiler** on the 3–4 functions P2/P3 flag. | Exact line costs to target and to measure against after a change. |
| P5 | **tracemalloc** peak + growth over one long game. | Whether tree memory is a scaling risk (informs caps / pruning). |
| P6 | Write the breakdown into this doc as a table (the "current profile"), with each slice's Amdahl ceiling. | The evidence base for choosing a technique below — and the before-numbers for any change. |

Output of Part A: a table like *"select_action 35%, move-gen 20%, get_next_state
15%, inference 12%, backprop 8%, other 10%"* (numbers TBD) — and from it, the
shortlist of what's actually worth optimising.

---

## Part B — Candidate speed techniques (menu, to choose from after profiling)

Each is annotated with what it targets, the rough leverage (bounded by its
slice via Amdahl), effort, and risk. **None should be started before Part A
says its slice is big enough.**

### B1. Vectorise `_select_action` (numpy)
- **Targets:** the per-move UCB Python loop — replace the `for`-loop with a
  vectorised numpy computation over the legal-move array (compute all UCB scores
  at once, `argmax`).
- **Leverage:** potentially large *if* select is a big slice (suspected). Cheap,
  pure-Python→numpy, no new deps.
- **Risk:** low — must stay bit-identical to the loop (golden test); careful with
  the virtual-loss / first-visit branches.
- **Why first-ish:** lowest effort, no toolchain, attacks the prime suspect.

### B2. Cython on the MCTS hot loop
- **Targets:** `_descend_to_leaf` + `_select_action` + `state_key` compiled to C,
  removing interpreter overhead from the tightest loop.
- **Leverage:** medium-high on the Python-interpreted slice (Scalene's
  "Python time" column sizes this); Amdahl-bounded by that slice.
- **Effort:** medium — `.pyx`, build step, typed memoryviews; the tree dicts are
  awkward to type (may need a restructure).
- **Risk:** medium — build complexity, determinism, harder debugging.

### B3. Numba (`@njit`) on numerical kernels
- **Targets:** self-contained numerical functions (e.g. a UCB-scoring kernel, or
  pieces of move-gen) JIT-compiled.
- **Leverage:** 30–100× on the *kernel*, but only kernels that are pure-numeric
  and isolatable; the MCTS tree (dicts/objects) isn't Numba-friendly.
- **Effort/risk:** low-medium per kernel, but limited surface fits Numba.

### B4. Bitboard board representation
- **Targets:** `state_key`, `get_next_state`, and move-gen — represent the 14×14
  board + piece placements as integer bitmasks, so placement/legality/keying
  become bitwise ops instead of numpy array work.
- **Leverage:** potentially the biggest *structural* win — it speeds **several**
  hot slices at once (keying + state transition + move-gen), so its Amdahl
  ceiling is the sum of them. This is how strong engines (and Pentobi-style move
  code) go fast.
- **Effort:** **high** — a real reimplementation of board internals; needs the
  rigorous equivalence-test discipline (10k+ positions, bit-identical games)
  we used for F2. A multi-week piece, its own plan.
- **Risk:** high (correctness surface is large) but well-understood.

### B5. Free-threaded Python (3.13t / 3.14) — threads instead of processes
- **Targets:** the *parallelism model itself*. Today we use `ProcessPoolExecutor`
  + `spawn` purely to dodge the GIL for CPU-bound move-gen/MCTS — which forces
  per-worker net copies, checkpoint reloads, and the IPC pain that sank F5. On a
  **free-threaded** build the GIL is gone, so CPU-bound *threads* run in
  parallel: one shared net in memory (no IPC, no spawn), threads share the tree
  infrastructure, and a cross-worker batch becomes a trivial in-process call
  (F5's goal, for free).
- **Leverage:** potentially large *and* structural — could simplify the whole
  parallel stack. Reported ~3.5–4× on 4 cores for CPU-bound work.
- **Effort/risk:** **high / speculative** — requires a 3.13t/3.14 interpreter,
  every C-extension (torch, numpy) must support free-threading, and our code must
  be thread-safe. Worth a *spike* (does torch+numpy run free-threaded on the PC?)
  before committing.

### B6. Algorithmic / code-level trims (cheap, always-on)
- Micro-opts the profile will reveal: avoid recomputing `state_key`
  (`board.tobytes()`) when the key is already known; hoist attribute lookups out
  of loops; cache `np.where(valids)[0]` per node instead of recomputing; reuse
  buffers in `get_next_state` instead of copying. Individually small, collectively
  a free ~1.1–1.3× with no new tooling.
- **Adaptive-sim endgame taper** ([`../IDEAS.md` I1](../IDEAS.md)) — not a "make
  the code faster" trick but a "do less work" one: stop spending 300 sims on
  <30-move endgames. Orthogonal, cheap, real wall-clock back. (You said discuss
  later — noted, not included in the sequence.)

---

## Recommended sequence

1. **Part A profiling first** — get the real per-slice breakdown + Amdahl ceilings. ~half a day, no risk. *Nothing below starts until this is in.*
2. **B1 (vectorise `_select_action`)** + **B6 (code-level trims)** — cheap, likely-high-leverage, no toolchain. Measure.
3. Then, guided by the profile, the bigger structural bet — **B4 (bitboards)** if move-gen+state+transition dominate, or **B5 (free-threaded spike)** if the parallelism overhead dominates. These are mutually informative; pick one to plan properly.
4. **B2/B3 (Cython/Numba)** as targeted follow-ups on whatever interpreted kernel remains the top slice.

The discipline throughout (F5's lesson): **measure → confirm the slice is worth it → change behind a flag / with a golden equivalence test → re-measure.**

---

## Sources
- py-spy (sampling, multiprocess): https://github.com/benfred/py-spy
- Scalene (line-level CPU/native/memory): https://github.com/plasma-umass/scalene
- line_profiler: https://github.com/pyutils/line_profiler
- Free-threaded Python benchmarks (3.14): https://www.danilchenko.dev/posts/python-314-free-threading/
- Cython vs Numba overview: https://leapcell.io/blog/supercharging-python-performance-with-cython-and-numba
