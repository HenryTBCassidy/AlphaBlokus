# P0 — Make the measurements informative + reclaim training wall-clock

The first phase of the post-v3 plan (see [`../research/xl-training-scaleup.md`](../../research/xl-training-scaleup.md)
addendum A6, phase P0). v3 showed that our strength instruments are half-blind — 14 of 19 arena
rejections scored *exactly* 50–50 and one tournament pairing was 30/30 draws, because eval play is
deterministic-per-(seed, colour) so near-equal nets just split by colour. We currently **cannot
resolve strength effects below ~50 Elo**, which is precisely the range the next experiment (P1 —
thickening the Gumbel operator) will produce. P0 fixes the instrument first, and reclaims the
training-phase wall-clock that the `dataloader_workers: 0` OOM-workaround is costing (~21% of a
generation), so P1 is both *readable* and *cheaper*.

**Scope:** code + one benchmark run, all on the existing `large` net + 1×5090 (or the local box)
stack. No net-size change, no new architecture. **Prerequisite for P1** (the operator-thickening
`--resume` run). Does **not** touch RunPod — the v3 pod is terminated; this is repo + box work.

Companion docs: [`../guides/PLAN-FORMAT.md`](../../guides/PLAN-FORMAT.md),
[`../guides/STYLE-GUIDE.md`](../../guides/STYLE-GUIDE.md), the v3 addendum (A1, A4, A5, A6).

---

## Checklist

| # | Item | Effort | Priority | Files | Done |
|---|------|--------|----------|-------|------|
| S1 | Opening-diversification config fields + plumb into the arena gate + Elo eval | 1.5 h | High | `config.py`, `training/coach.py` | ✅ |
| S2 | Plumb opening diversification into the pool BayesElo tournament | 1 h | High | `config.py`, `evaluation/tournament_run.py` | ✅ |
| S3 | Validation gate — prove opening diversity measures strength, not opening-luck | 1.5 h | High | (control run + analysis) | ✅ (known-gap control ~64% vs null ~49–51 → diversity measures strength; note: blind to colour-pinning, which `fix-arena-colour-pinning.md` then fixed) |
| S4 | Make `dataloader_workers > 0` reliable (fix the memmap+forkserver deadlock) + validate | 3 h | Medium | `games/base_wrapper.py`, `config.py` | ✅ (validated on box: 8 gens, workers=8+spawn, no deadlock) |
| S5 | Re-baseline the gen-57 donor ladder at 100 games/level (execution) | ~1 h run | Medium | (box run; results doc) | ✅ (gen-57 donor re-laddered: L4 ≈ 48%, weighted ≈ 0.205) |
| S6 | Tests + docs | 1 h | High | `tests/`, `config.py` docstrings, addendum | ✅ |

Execution order: S1→S2 build the instrument, then **S3 validates it — a hard gate: the production opening-diversity defaults are not trusted, and P1 must not rely on the diversified gate/tournament, until S3 passes.** S4 (perf fix) and S5 (baseline run) are independent and parallelisable; S6 last. S4 is the higher-risk row (escape hatch in its section).

> **Status (2026-07-08):** S1, S2, S6 and the *code* of S4 have landed on `feat/p0-instrument-and-dataloader`. Everything is **off by default** — the new opening-diversity fields default to 0 (today's exact behaviour) and `dataloader_context` now defaults to `"spawn"` but is inert while `dataloader_workers == 0` (the committed configs are all pinned to 0). **S3** (the validation control), **S4's** multi-generation GPU validation, and **S5** (the re-baseline ladder) are GPU-box tasks — not yet run. The production opening-diversity defaults (~1.0 / 6 plies) and `dataloader_workers > 0` must not be enabled until S3 / S4-validation pass.
>
> **S6 doc note:** the companion `docs/research/xl-training-scaleup.md` addendum referenced above is not present in this repo (not tracked on `main`), so the S3/S4/S5 verdicts + settings it is meant to record cannot be appended here — capture them there once it exists / lands. The code-level documentation (config docstrings for the new fields + the `dataloader_context` default, this status note, and the unit tests in `tests/evaluation/test_opening_diversity.py`) is complete.

---

## S1. Opening-diversification config fields + plumb into the arena gate + Elo eval

**Current state.** `NetworkPlayer` already implements an opening temperature schedule —
`opening_temp`/`opening_moves` ctor args (`evaluation/players.py:70-71`) with the switch logic at
`players.py:99-101` (sample from the visit distribution for the first `opening_moves` of the
player's own plies, then revert to `temp`). But the arena/Elo call sites construct players
**without** them, so they default to 0 (fully deterministic): `coach.py:495` (`prev_player`),
`coach.py:501` (`new_player`), `coach.py:595` (Elo/oracle player). `pentobi_benchmark.py:290-296`
already passes them — that's the working precedent to copy.

**Target.**
- Add two fields to `RunConfig` (`config.py:339`), defaulting to 0 so existing configs are
  unchanged: `arena_opening_temp: float = 0.0`, `arena_opening_moves: int = 0`. Docstring: "Opening
  temperature for arena/Elo eval games — >0 injects opening diversity so near-equal nets don't split
  exactly 50/50 (the v3 gate-resolution problem). Production: ~1.0 for ~6 plies."
- Pass them into all three `NetworkPlayer` constructions in `coach.py` (both arena players + the Elo
  player), reading from `self.config`.
- Set them in the production configs (`blokus_cloud_v2.json` and the eventual `_v4`): start
  `arena_opening_temp: 1.0`, `arena_opening_moves: 6` (one plausible default — validate in S5).

**Design decision to resolve in review (call it out, don't silently pick):** diversifying the
*gate* changes training dynamics, not just measurement — it should reduce false rejections of
marginally-better candidates (the plateau symptom: 14/19 rejections were exact 50/50), which is a
genuine improvement, but it also adds variance to accept/reject. Two options:
  1. **Diversify the gate too** (recommended — attacks the false-rejection stall directly). Apply
     the opening schedule symmetrically to both arena players so it stays fair.
  2. **Keep the gate deterministic; diversify only the tournament (S2) + a separate strength eval.**
     Safer for training-dynamics continuity, but leaves the false-rejection problem in place.
The plan assumes option 1; flag for Henry's decision before implementing.

---

## S2. Plumb opening diversification into the pool BayesElo tournament

**Current state.** The tournament now lives in `evaluation/tournament_run.py` (`run_tournament`,
called by the thin `scripts/tournament_elo.py` wrapper). Its per-pairing `NetworkPlayer`s are built
with `temp=0` and no opening schedule → the 30/30-draw pairings (e.g. gen32 vs gen40) that made the
pool-Elo curve unresolvable below ~50 Elo (addendum A3).

**Target.**
- Add `opening_temp`/`opening_moves` to `TournamentConfig` (`config.py:181`), defaulting to 0.
  Recommend production defaults matching S1 (1.0 / 6) so the tournament and gate measure play from
  the same distribution.
- Plumb them into the `NetworkPlayer` construction inside `run_tournament`'s pairing loop.
- The pool BayesElo fit itself is unchanged — it just receives non-degenerate W/L/D counts, so the
  ratings gain resolution. `games_per_pairing` (default 30, `config.py:194`) can stay; note in the
  docstring that with opening diversity each pairing now carries ~`games_per_pairing` independent
  games instead of ~half (mirrored deterministic pairs no longer collapse).

---

## S3. Validation gate — prove opening diversity measures strength, not opening-luck

**Why.** `opening_temp` samples from the MCTS *visit distribution*, so it picks among moves search
already rated well (weighted to the best), not random blunders — and Blokus's wide, near-symmetric
opening has many genuinely comparable first placements, so the diversity is among *good* openings.
**But** at aggressive settings the sampler reaches into the worse tail and can seed genuinely
disadvantaged positions, which would make games measure "who recovers from a bad opening" instead
of "who's stronger." This row proves the chosen settings don't do that. **Until it passes, the
S1/S2 production defaults are not trusted and P1 must not rely on the diversified gate/tournament.**

**Settings guidance — start modest.** `opening_temp = 1.0`, `opening_moves = 6` (sample the top-few
reasonable openings for the first ~6 plies, then greedy). Do *not* go aggressive (high temp / long
horizon) — that is exactly what risks tail moves. If the control below fails, dial these down before
anything else.

**The control (execution + analysis; no new code beyond S1/S2):**
- **Known-gap test.** Take a clearly-stronger vs clearly-weaker net (e.g. v3 gen-40 vs gen-5, or
  vs the gen-0 anchor). Play the pairing at ≥100 games, once with diversity OFF (temp=0) and once ON
  (temp=1 / 6 plies). **Pass condition: the real gap survives with diversity ON** — the stronger net
  still wins decisively (roughly the same side of ~65%, not collapsed toward 50/50). If diversity
  washes out a known strength gap, it's measuring opening-luck → fail; reduce temp/moves and re-test.
- **Null test.** Play a net against an identical copy of itself with diversity ON. **Pass condition:
  ~50/50 with proper binomial variance across games** — not a degenerate exact-50/50, and no
  systematic bias to one colour. Confirms diversity de-correlated the mirrored pairs without adding
  bias.
- Both must hold before the production defaults are set (S1/S2) and before P1 relies on the gate.

**Fallback if the control shows bias (opening-luck leaking in):**
- **Paired openings** — sample an opening once and play it from *both* colours with both nets, so a
  symmetric slightly-off opening cancels (the standardized-opening-suite trick from engine testing).
  A small change to how the arena/tournament seeds a pairing.
- Further out only if paired openings still show bias: a curated opening book of vetted diverse
  openings (zero bad-move risk, more work).

---

## S4. Make `dataloader_workers > 0` reliable + validate

**Current state.** Training runs with `dataloader_workers: 0` (`config.py:156`) — the workaround
after the memmap + forkserver DataLoader path deadlocked at v3's gen-4 training step (hung between
"Starting Training" and "Epoch 1/1"; gens 1–3 were fine — an intermittent worker-startup race). With
`workers=0` the per-item densify (sparse policy → 17,837-dim + board encode) runs single-threaded on
the main process and starves the GPU: v3's training phase was **2.7× the cost model** (760 s vs
~280 s/gen), ~21% of total wall-clock (addendum A5). The context knob already exists —
`dataloader_context: "forkserver"|"spawn"|"fork"` (`config.py:171`), resolved by
`resolve_dataloader_context` (`base_wrapper.py:44`, falls back to spawn) — and the memmap-vs-in-RAM
dataset split is at `base_wrapper.py:423`.

**Target.** Make `workers > 0` deadlock-free so we can turn it back on.
- **Leading hypothesis: `spawn` context.** The deadlock hit under `forkserver`; `spawn` starts each
  worker from a fully fresh interpreter (no inherited fork/forkserver state), which most cleanly
  avoids torch-DataLoader × JAX-loaded-process hazards. First change: try `dataloader_context:
  "spawn"` and reproduce.
- **Diagnose before committing.** Attempt a local repro: small buffer, `workers=8`, JAX imported,
  memmap path, iterate several epochs; if it doesn't repro locally (it's intermittent), rely on the
  box validation below. Check the usual suspects surfaced in the v3 logs: memmap file build/open
  under workers, `persistent_workers` interaction, and `torch.compile` first-batch compilation
  colliding with worker spawn.
- **Validation (the real gate):** a ≥6-generation run at `workers=8` on the box that clears the
  buffer-fill generation (where v3 hung) **twice** without stalling, at the large net. Confirm
  training-phase s/gen drops toward the ~280 s model.
- **Escape hatch (keep the plan unblocked):** if `spawn` can't be made reliable within the effort
  budget, keep `workers=0`, record the ~21%/×1.33–1.6 wall-clock cost in the config docstring +
  addendum A5, and defer this to a dedicated investigation. P1 can run at `workers=0`; it's just
  costlier. **This row must not gate S1/S2/S3/S5.**

---

## S5. Re-baseline the gen-57 donor ladder at 100 games/level

**Why.** Every "did the net improve?" comparison is against the gen-57 starting net, but our only
gen-57 ladder is at 40 games/level (wide CIs). To read P1's result we need the baseline pinned at
the same fidelity as v3's final ladder (100 games/level, levels 1–9, 400 sims).

**Target (execution, no code).** On the box (`gpu-anywhere`), run
`scripts.pentobi_benchmark --config <large cfg> --net <gen-57 donor> --sweep --sims 400 --games 100
--workers 8` against the gen-57 net (already local / on the box from `blokus_cloud_60`). Record the
per-level win rates + CIs alongside v3's final ladder in the addendum's A1 table (replacing the
40-g/level donor row), and fetch the HTML report to `temp/benchmarks/`. This is a ~1 h run; I can
execute it independently of the code work.

---

## S6. Tests + docs

- **Unit test** the opening schedule plumbing: a `NetworkPlayer` with `opening_temp>0`,
  `opening_moves=k` samples (non-argmax) for its first `k` plies then reverts to greedy — assert the
  temperature switch at the boundary (real objects, no mocks). Assert `coach` + `tournament_run`
  pass the config values through (construction-time check).
- **Docs:** config docstrings (S1/S2 fields, S4 outcome), a short results note in the
  `xl-training-scaleup.md` addendum recording S3's validation verdict + settings, S5's pinned
  baseline, and S4's dataloader verdict; update `CLAUDE.md`'s gotcha on `dataloader_workers` if S4
  lands the fix.
- Note: the *empirical* validation of the opening-diversity settings is **S3** (the hard gate), not
  here — S6 covers the code-level unit test and documentation only.

---

## Not in P0 (deliberately)

- **P1** (Gumbel n→512, considered→64, `--resume` from gen-40) is a config + run, gated on this
  plan landing — authored separately, not here.
- The self-play buffer from v3 must be **preserved** (it's the `--resume` source for P1) — tracked
  outside this plan.
- No net-size change (`xl` is phase P4), no distributed self-play (IDEAS I6), no async actor-learner.
