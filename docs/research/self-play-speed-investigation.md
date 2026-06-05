# Self-Play Profiling Investigation — Time + Memory

**Status: companion reference for the [profiling investigation](../plans/profiling-investigation.md).** That plan owns the actionable checklist (P1–P8) and the optimise-or-train decision; this doc is the deeper reference it draws on — *why* self-play is now CPU-bound, the **Amdahl analysis** for sizing each lever, the tool rationale, the hot-loop anatomy, and the full **candidate-technique menu** (Part B). The discipline (F5's lesson): measure where time *and* memory actually go **before** prescribing any optimisation.

**Scope note:** the gen-3 training-step OOM is **not** a topic here — that's a known bug with a code-confirmed cause, fixed directly in [`../plans/self-play-memory-fix.md`](../plans/archive/self-play-memory-fix.md). Companion: [`../plans/archive/full-cycle-optimisation.md`](../plans/archive/full-cycle-optimisation.md) (F1–F5) and [`../IDEAS.md`](../IDEAS.md).

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

**Goal:** a trustworthy, current breakdown of both **where the wall-clock goes**
and **where the memory goes**, at two levels each: (1) coarse, per MCTS phase /
per training phase; (2) fine, per line in the hot functions / per allocation site.

### The three memory targets

The OOM that crashed the scaled run was at the **training** step, not in self-play —
so memory must be profiled across the *whole* lifecycle, not just the search tree:

| Target | What grows | Why it matters |
|--------|-----------|----------------|
| **In-episode MCTS tree** | the per-episode node/edge dicts (25→50→75… entries as sims accumulate) | scaling risk for long games / high sim counts; bounded per game |
| **Self-play → replay buffer** | examples accumulating in `train_examples_history` across generations (board + policy + value per move) | the RAM held *between* phases; duplicates what's already on disk |
| **Training step** | `train()` materialising the buffer into dense tensors — the gen-3 OOM site | fixed directly in the [OOM fix](../plans/archive/self-play-memory-fix.md); profiled here only to understand memory at scale once that lands |

The training-step target can be measured **cheaply without a full multi-hour run** —
build a synthetic buffer of realistic size (N examples) and profile a single
`train()` call. (This is optimisation-track context; the OOM bug itself is being
fixed directly, not gated on this.)

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
| **tracemalloc** | Memory growth of the search tree within an episode (the per-episode dicts that grow 25→50→75…). Confirms whether long games / many sims risk memory pressure at scale. | Already flagged in `archive/scaled-training-run.md`; quantify it. |

### The steps

The executable checklist lives in the [profiling plan](../plans/profiling-investigation.md) (P1–P8): build the single-game harness, run py-spy / Scalene / line_profiler for time, tracemalloc for the in-episode tree and the replay buffer, then write up the current profile and decide. The tools above explain *why* each; the plan tracks *doing* them.

Output: a **time** table (e.g. *"select_action 35%, move-gen 20%, get_next_state 15%, inference 12%, backprop 8%, other 10%"* — numbers TBD) plus a **memory** table ranking the tree / buffer components by bytes — and from them, the shortlist of what's worth optimising, each sized by its Amdahl ceiling.

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
