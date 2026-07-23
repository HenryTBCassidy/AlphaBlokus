# Pentobi distillation — expert corpus, SL imitation, RL beyond

The curriculum fix for the L4 plateau. The capacity probe (post-regression-recovery P8) showed the
plateau is **not** a net-size problem — `xl` fit gen-40's self-play data no better than `large`
([`../research/regression-and-next-steps.md`](../research/regression-and-next-steps.md) §3) — so the
binding constraint is the *curriculum*: pure self-play-vs-itself is exhausted for this net. The fix
is to inject external strength by distilling from Pentobi, exactly how AlphaGo (2016) bootstrapped
from human games: supervised move-prediction to imitate a strong player, then RL to surpass it.
Pentobi L9 beats our best net 80–95%+ — it is an information source self-play cannot synthesise
(research R5). Phase 1 (this branch) builds and validates the data half: a diverse, correct,
reusable corpus of Pentobi L9 games. Phases 2–3 (separate branches) build the training half.

**Locked design decisions** (agreed before Phase 1 was built — do not re-litigate):

- **L9 only.** Strongest moves = best policy targets; L9-vs-L9 outcomes = most accurate value
  labels. No low levels (imitating weak play poisons the policy), no difficulty curriculum
  (irrelevant for supervised imitation).
- **Policy target = the single move Pentobi played, stored one-hot** (behavioural cloning).
  Label smoothing is a training-time transform, never stored.
- **Value target = game outcome from the side to move**; the **final score margin** and
  **side-to-move** are stored alongside.
- **Diversity is mandatory and proven, not assumed** — Pentobi at L9 is near-deterministic, so a
  naive corpus is a pile of clones (see D3 for the measured verdict on seed variation vs the
  random-opening-prefix method).
- The corpus is a **one-time reusable asset** (parquet shards, schema in
  [`../07-DATA-STORAGE.md`](../07-DATA-STORAGE.md) § Pentobi Distillation Corpus), consumed by
  every future run.

---

## Checklist

| # | Item | Effort | Priority | Done |
|---|------|--------|----------|------|
| D1 | Corpus generator: game loop + harvesting + parquet shards + diversity metrics (`pentobi/corpus.py`, `scripts/pentobi_corpus.py`, tests) | 1 day | High | ✅ |
| D2 | Engine-path shakeout on the box (L1) + full-row replay validation | 1 h | High | ✅ |
| D3 | **Diversity A/B at L9**: seed-variation-only vs random-opening-prefix, quantified | 2 h box CPU | High | ✅ |
| D4 | L9 pilot (~190 games): correctness validation + throughput measurement | 2 h box CPU | High | ✅ |
| D5 | **Full corpus generation run** (execute the D4 recommendation after human review) | see D5 | High | |
| D6 | Corpus dataloader: shard streaming, symmetry augmentation, label smoothing | 1 day | High | |
| D7 | SL distillation trainer: policy CE + value MSE fine-tune of the best net (+ from-scratch arm) | 2 days | High | |
| D8 | SL evaluation gate: mini-ladder the distilled net | ½ day + box | High | |
| D9 | RL warm-start from the distilled base (continuation hygiene: AdamW, epochs 1, LR 2.5e-4) | ½ day + run | High | |
| D10 | Continuous Pentobi-mix: corpus examples blended into the RL replay buffer | 1–2 days | Medium | |
| D11 | Opponent-pool diversity for RL self-play | 2 days | Medium | |

**Phases + gates:** Phase 1 = D1–D5 (data; D1–D4 are this branch, D5 runs after the
recommendation below is reviewed). Phase 2 = D6–D8 (SL distillation; **gate: D8's ladder
verdict** — R5's criterion, +10 pp at any of L5–L7 after SL alone). Phase 3 = D9–D11 (RL beyond
the teacher; **gate: ladder progression beyond the Phase-2 result**, stop rule per the external
keep-best policy). Nothing in Phase 3 starts unless Phase 2's gate fires.

---

## D1. Corpus generator ✅

**Built** (this branch): `src/alphablokus/games/blokusduo/pentobi/corpus.py` +
`scripts/pentobi_corpus.py`.

- **Game loop** (`play_corpus_game`): one `pentobi-gtp` process plays *both* colours
  (`genmove b`/`genmove w` alternately) — half the processes of engine-vs-engine. The engine is
  built with `--noresign` (new `PentobiGtp` flag) so every game reaches its true final position:
  a resignation would forfeit the score margin the value labels need. The first `k` plies
  (`--opening-random-plies`, default 4) are sampled uniformly among legal placements and relayed
  to the engine; every later ply is Pentobi's. Reuses the existing GTP adapter + move translation
  (`pentobi/gtp.py`, `pentobi/translation.py`) — nothing re-implemented.
- **Harvesting**: one example per *Pentobi* ply (opening plies are never harvested): canonical
  compact board (side-to-move perspective, the self-play buffer's exact form), the played action,
  outcome, margin, side-to-move. Desync guards: every engine move is legality-checked against our
  rules engine, and the engine's `final_score` must equal our computed margin (new public
  `BlokusDuoGame.final_scores`).
- **Storage**: parquet shards, schema a strict superset of `SelfPlayStore`'s (markers asserted
  equal in tests) — documented in [`../07-DATA-STORAGE.md`](../07-DATA-STORAGE.md). Atomic writes
  (`.tmp` → rename); resume = skip existing shards; game `g`'s seeds are a pure function of
  `(--seed, g)` so a rerun regenerates exactly the missing shards.
- **Parallelism**: `--workers` spawn-pool over shards; one single-threaded engine per worker.
  **Each L9 engine wants 2 GB RAM** (Pentobi sizes its search tree as `min(2 GB, RAM/4)` —
  `libpentobi_mcts/Player.cpp get_memory`), so workers ≈ `min(cores, RAM/2.2 GB)`.
- **Tests** (10, engine-free): the same loop/harvest/persist/validate pipeline driven by a real
  uniform-random move source through the real rules engine — ply bookkeeping, label replay
  equivalence, score-mismatch guard, GTP score parsing, schema markers vs `SelfPlayStore`,
  round-trip + one-hot policies + 44-plane re-encode, symmetry-augmentability
  (transposed one-hot = one-hot of `transpose_action`), validator catches corruption,
  diversity-metric counts, multi-shard analyze.

## D2. Engine-path shakeout ✅

8 games at L1 on the box through the real engine (`generate` → `validate` → `analyze`): all 225
positions replay-validated OK (every stored move legal where stored, boards/labels reproduced by
replay, engine margin == rules-engine margin in all games), diversity 8/8 unique games.

## D3. Diversity A/B at L9 ✅

The crux: does `--seed` alone diversify near-deterministic L9 play, or is the random opening
prefix required? Two corpora generated on the box, quantified with `analyze`:

- **(a) seed variation only** — 24 games, `--opening-random-plies 0`, distinct engine seed per
  game (`set_random_seed`).
- **(b) random opening prefix** — the D4 pilot, 192 games, `--opening-random-plies 4`.

**Measured 2026-07-23 (box, i5-13600KF, 12 workers):**

| Metric | (a) seed-only, 24 games | (b) prefix-4, 192 games |
|---|---|---|
| Unique full games | 24/24 (100%) | 192/192 (100%) |
| Distinct openings at ply 1 / 2 / 4 / 6 | **3** / 13 / 21 / 24 | **160** / 192 / 192 / 192 |
| Unique stored positions | 709/773 (91.7%) | **5,811/5,811 (100%)** |
| White / Black wins / draws | **23 / 0 / 1** (96% White) | 144 / 20 / 28 (75% White) |

**Verdict:** seed variation is *not* a clone factory — L9's search randomisation makes every full
game unique and >90% of positions distinct — but it has a hard **opening funnel**: 24 games
explored only **3 first moves** (and 13 two-ply openings), so a large seed-only corpus would pile
policy mass onto a narrow opening tree exactly where coverage matters most for a policy prior.
The random prefix removes the funnel by construction. **Chosen mechanism: both together** (the
generator always gives each game a distinct engine seed; `--opening-random-plies 4` adds the
prefix): prefix for guaranteed opening coverage, seed variation for within-game diversity beyond
the prefix. Note the seed-only colour skew (96% White wins) for Phase 2: value labels from
balanced starts are near-constant per colour; the random prefix also injects unbalanced starts
whose outcomes carry more value signal.

## D4. L9 pilot: correctness + throughput ✅

192 games at L9, 4 random opening plies, seeds 0–191, 12 workers on the box (RAM-capped:
12 × 2 GB engines on 31 GB; the box's i5-13600KF has 14 cores). Measured 2026-07-23:

- **Correctness:** `validate` replayed all **5,811 positions across 48 shards — zero
  mismatches** (legality, board bytes, side-to-move, one-hot policy, value sign, margin,
  terminal scores), on top of the per-game engine-margin cross-check that ran during
  generation. The parquet round-trips through `iter_corpus_examples` into the exact
  `ProcessedExample` shape the training pipeline consumes.
- **Throughput:** **176 games/hour = 5,333 positions/hour at 12 workers** (~14.7 games/hour
  per worker; ~245 s/game). Prefix games are *faster* than seed-only full games (120 games/h)
  — the random prefix replaces Pentobi's most expensive early-search plies. ~**30.3 harvested
  positions/game** (min 24, max 37).
- **Labels:** margins mean +12.4 (White-favoured), median 3, range −43…+88; draws 14.6%.
- **Size:** 580 KB for 5,811 positions (~100 B/row) — a full corpus is ~150 MB.
- Pilot artifacts kept on the box: `~/corpora/pentobi_l9_pilot_20260723` (+ the seed-only A/B
  corpus and the generation log).

## D5. Full corpus generation run

**Target: 50,000 games ≈ 1.5 M stored positions ≈ 3.0 M after 2× symmetry augmentation**
(`generate --num-games 50000 --level 9 --seed 0 --opening-random-plies 4`). At the measured
rate that is ~2,840 worker-hours (13600KF-core basis). Options:

| Route | Wall-clock | Cost | Notes |
|---|---|---|---|
| Box only (12 workers) | ~12 days 24/7 | £0 | 176 games/h → ~4,200 games/day; box otherwise idle-CPU during GPU runs |
| Rented 64 vCPU / 256 GB (e.g. m7a.16xlarge-class, spot) | ~3 days | ~$100–130 spot / ~$250 on-demand | ~60 workers × ~11.7 games/h (cloud cores ~0.8× box); needs ≥2.2 GB RAM/worker (Pentobi allocates min(2 GB, RAM/4) per L9 engine) |
| Rented 96 vCPU / 384 GB (m7a.24xlarge-class, spot) | ~2 days | ~$100–150 spot | Same £/game; finishes faster |

**Recommendation — two-stage, cost-safe:**
1. **Start the box generating now** (free): `--num-games 13000` gives ~13k games / ~800k
   post-symmetry positions in ~3 days — enough to build Phase 2 against and run the first SL
   experiment + D8 ladder gate. The generator is resumable, so these games are not throwaway:
   the same `--seed 0` command line simply continues into the full corpus (shards are a pure
   function of `(seed, game_id)`).
2. **Rent only if the D8 gate fires**: extend to 50k games on a ~64–96 vCPU spot box
   (~$100–150, 2–3 days), running the identical command with `--out` on the rented box (or
   rsync the box shards over and let resume fill the gap). **Sync shards to durable storage
   and verify off-box before any auto-stop/terminate** — the corpus is the product.

Gate: human reviews this recommendation before anything is rented (stage 1 needs no sign-off
beyond starting the box job).

## D6. Corpus dataloader (Phase 2)

Stream `iter_corpus_examples` shards into the training batch pipeline: densify the one-hot policy
per batch (shared `sparse_policy.densify`), apply **label smoothing** at batch time (start
ε≈0.1 over *legal* moves — requires the valid mask from `board_from_compact`, or uniform-over-all
as the cheap first cut), apply the order-2 symmetry (`IGame.get_symmetries`) as 2× augmentation,
rebuild net input via `encode_compact` lazily (exactly `_LazyPolicyDataset`'s trick). Weighted
sampling across shards; held-out split at *game* granularity (reuse the capacity-probe's
game-level split so no position of a held-out game leaks).

## D7. SL distillation trainer (Phase 2)

Fine-tune the current best net (v3 gen-40) on the corpus: policy CE against the smoothed one-hot,
value MSE against the outcome (margin is stored for a later margin-aware experiment — do not
build it into v1). AdamW (weight decay already default), LR ~1e-4, early stop on held-out policy
CE. Also run a **from-scratch arm** (same net size) — v3's operator-ceiling history suggests a
fresh net may imitate better than a converged one; the two arms cost one config each on the box
GPU (supervised training on ~2–4 M positions is hours, not days). Track held-out top-1 accuracy
vs Pentobi moves and value calibration.

## D8. SL evaluation gate (Phase 2)

Mini-ladder (`scripts/mini_ladder.py`, L3–L6 × 50 games × 400 sims) both arms + the v3 gen-40
baseline. **Gate (research R5): +10 pp at any of L5–L7 after SL alone.** If neither arm moves the
ladder, distillation data or recipe is wrong — stop and diagnose before any RL spend.

## D9. RL warm-start from the distilled base (Phase 3)

Resume standard self-play RL from the D8 winner with the continuation hygiene rules (AdamW
default-on, `epochs: 1`, LR 2.5e-4), external keep-best + drift circuit-breaker as the selection
mechanism. Cost: one box run first (free) before any paid scale-up.

## D10. Continuous Pentobi-mix (Phase 3)

Blend a fraction of corpus examples (or freshly generated Pentobi games) into the RL replay
buffer each generation — anchors the policy to expert play while self-play explores beyond it.
Ratio is the experiment (start ~10–25% of buffer positions); the corpus's shard format makes the
mix a dataloader concern, no pipeline surgery.

## D11. Opponent-pool diversity (Phase 3)

Self-play against a pool (past checkpoints + optionally Pentobi itself at eval-time levels)
instead of pure latest-vs-latest — the xl-scaleup research's diversity recommendation. Builds on
the arena/player machinery; scoped in detail when Phase 3 starts.
