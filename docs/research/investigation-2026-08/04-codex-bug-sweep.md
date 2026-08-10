# AlphaBlokus bug sweep — Codex (2026-08-04)

Scope: the six requested items, in order, against `main` at `92bf8db`. No application code was changed and no training or remote job was started. Scratch probes lived in `/tmp`.

## Verdict summary

| Item | Verdict |
|---|---|
| 1. JAX PRNG uniqueness | **Clean** for the current single-device backend |
| 2. Legal masking vs top-k | **Clean** — masking precedes top-k at root and children |
| 3. Completed-Q depth perspective | **Clean** — Q is local player-to-move/parent perspective with one sign flip per live ply |
| 4. Replay sampling | **One verified defect; one strength-relevant suspicion** |
| 5. Optimizer continuation | **Clean** — moments and LR clock have explicit, tested continuation semantics |
| 6. Missing exact tests | **Suspicion/test gap** — forced/terminal behavior passes probes, but no exhaustive Blokus or independent Gumbel oracle exists |

## VERIFIED DEFECTS

### CODEX-01 — Unseeded replay pre-shuffle defeats reproducible training and eval selection

- **Severity:** `degrades-silently`
- **File:** `src/alphablokus/training/replay_buffer.py:80-89` (trigger); `src/alphablokus/training/coach.py:156-166` (missing Python RNG seed); `src/alphablokus/games/base_wrapper.py:799-816` (later seeded shuffle)
- **What is wrong:** `flat_shuffled_examples()` calls the module-level `random.shuffle`, but `Coach` seeds NumPy and Torch only. The later DataLoader has a carefully generation-seeded Torch generator, but its indices address a differently pre-shuffled list in every process. The fixed-seed eval set also selects deterministic indices from that nondeterministically ordered list. Thus the code and comments promise seeded A/B ordering while the effective examples and frozen eval positions vary.
- **Concrete failure scenario:** Two arms use the same seed, same self-play examples and same architecture except for one intended treatment. The OS-seeded Python shuffle permutes each arm differently before the shared Torch permutation is applied. Mini-batch composition, Adam trajectory and the 200 selected eval examples differ, adding uncontrolled variation at a measured policy-CE noise floor of about 0.05 nats.
- **Executed verification:** Three fresh Python subprocesses flattened the same synthetic 20-position buffer and returned three different orders. Repository search found no `random.seed(...)`; the only Python `random` consumer in `src/alphablokus` is this shuffle.
- **How to verify cheaply after fixing:** Run the same synthetic flatten in three fresh subprocesses and compare hashes, then build the eval set twice with the same seed and assert byte-identical `compact_boards.npy`. Also run two tiny same-seed TTT training jobs and compare per-batch example IDs or final checkpoint hashes.
- **Proposed fix (diff sketch):** remove the redundant pre-shuffle and let the already seeded DataLoader do the only training shuffle. Rename the method to `flat_examples()` and preserve canonical game/ply order for deterministic indexing. If retaining the pre-shuffle is required, pass a local `random.Random(_shuffle_seed(seed, generation))` into the replay buffer rather than using global state. Add a regression test spanning `ReplayBuffer -> build_or_load_eval_set -> DataLoader`.

```diff
-from random import shuffle
 ...
-examples = [example for game in self.games for example in game]
-shuffle(examples)
-return examples
+return [example for game in self.games for example in game]
```

## SUSPICIONS

### CODEX-02 — Position-uniform value training preserves the White-outcome shortcut

- **Severity:** `degrades-silently`
- **File:** `src/alphablokus/training/replay_buffer.py:80-89`; `src/alphablokus/games/base_wrapper.py:809-826`
- **What is wrong:** The replay buffer deliberately preserves games for eviction, then discards those boundaries for training. Every position is used once per epoch and policy and value share the same uniform-by-position sampler. There is no game-, outcome-, role- or phase-aware weighting for the value loss. Labels are correct, but a White-heavy game distribution makes absolute mover role a cheap predictor of the outcome and gives the value head no sampling pressure to learn beyond it.
- **Concrete failure scenario:** With roughly three quarters of games won by White, most White-to-move rows have `v=+1` and most Black-to-move rows have `v=-1`. The value head learns role/parity while training value MSE falls. Gumbel completed-Q then injects that biased value into unvisited actions at every node, producing plausible-looking but weak policy targets.
- **Executed verification:** The resolved `blokus_paired_gate_rerun` config says 60,000 games, 10,000 fresh games/generation and two epochs. Its log confirms the actual buffer reached exactly 60,000 games at generation 6, held 3,777,048 augmented examples, and logged lifetime reuse 12 (`2 epochs x 6 generations`). The generation-1 position-uniform eval sample contains 151 White-win, 33 Black-win and 14 draw rows among 198 role-inferable positions (76.3% / 16.7% / 7.1%); it has 103 White-to-move, 95 Black-to-move and 2 pass-ambiguous rows. White-to-move mean target was +0.515 while Black-to-move mean target was -0.684. Phase counts were 94 opening, 92 midgame and 14 endgame using placed-piece thirds. This 200-position sample lacks game lineage, so it is descriptive, not a valid confidence interval. On 400 older raw games where boundaries could be reconstructed, position-uniform sampling only modestly length-biased the source game (mean length 29.53 versus 29.27 under uniform-game sampling; games of length at least 32 were 24.0% of games but 26.9% of positions). The material skew is outcome/role, not game length in that sample.
- **How to verify cheaply:** Preserve `game_id`, absolute mover role, ply and outcome in a diagnostic-only replay export for one current generation. Report every minibatch's outcome/role/phase mix and compare the value head with colour-only and colour-by-phase baselines. Bootstrap all uncertainty by game. Then train from one frozen buffer with identical policy batches and compare unweighted versus outcome-balanced **value-only** loss across at least three weight seeds.
- **Proposed fix (diff sketch):** keep policy loss uniform by position, but carry an aligned value weight derived at game level. Start with inverse-frequency weights for White-win/Black-win/draw games, normalized to mean one; optionally correct the small length bias with a `1 / positions_in_game` factor. Return unreduced per-row value MSE, multiply by the aligned weights, and then reduce. Log role/outcome/phase composition and effective weights. Do not rebalance the policy targets in the same change.

```diff
-l_v = self.loss_v(target_vs, outputs.value)
+per_row_v = self.loss_v_unreduced(target_vs, outputs.value)
+l_v = (per_row_v * value_weights).sum() / value_weights.sum()
```

### CODEX-03 — Production Gumbel search lacks small independent exact oracles

- **Severity:** `degrades-silently`
- **File:** `tests/games/blokusduo/jax/test_gumbel.py:1-7`; `tests/games/blokusduo/jax/test_search.py:102-115`
- **What is wrong:** Gumbel tests assert only structural invariants and end-to-end completion; the file explicitly says there is no Python oracle. The only terminal-root search test uses PUCT. There is no pinned real non-terminal one-legal-action Gumbel case, no short Blokus endgame solved by exhaustive negamax, and no independent implementation of Sequential Halving plus completed-Q to catch a library-integration or perspective regression. Production bfloat16 is also absent from these exact-path tests.
- **Concrete failure scenario:** A future change flips one interior discount, applies a mask after compact selection, or passes completed-Q in the wrong perspective. Large stochastic tests still produce legal normalized policies and complete games, so CI remains green while every self-play target is biased.
- **Executed verification:** The existing focused search suites passed (`9 passed`). Scanning all 5,000 cached positions found 40 real non-terminal states with exactly one legal placement and 333 pass-only states, so fixtures already exist. A direct Gumbel probe on cached forced action `8923` and terminal pass `17836` chose the only legal action and assigned it exactly 1.0 target mass with finite normalized targets. Current behavior is sound; the missing regression oracle is the suspicion.
- **How to verify cheaply:** Add four deterministic tests: (1) cached forced non-pass under PUCT and Gumbel; (2) terminal/pass under both policies; (3) two- to four-ply endgames solved by an independent exhaustive negamax over the Python rules, asserting Q sign/action at every depth; (4) a tiny synthetic tree with fixed priors, leaf values and fixed Gumbels, comparing mctx output with a small NumPy completed-Q/Sequential-Halving reference. Run the same invariants in float32 and bfloat16.
- **Proposed fix (diff sketch):** add `tests/games/blokusduo/jax/test_exact_gumbel.py` with hand-selected cache indices and a local reference implementation that imports no production search helpers. Parameterize `dtype` and search policy where meaningful. Assert exact sole-action mass, terminal reward, alternating Q signs, considered-action schedule, chosen action and improved-policy probabilities.

## GENUINELY CLEAN

### Item 1 — PRNG uniqueness in lockstep JAX self-play

- `backend.py:167-180` folds generation into the base seed and advances a persistent stream once per wave.
- `actors.py:98-100,137-138` gives every scan ply a distinct key and separates search, sampling and tie-breaking streams.
- mctx generates root Gumbels with shape `(batch, actions)`, not `(actions,)`, and splits each simulation key into `batch_size` keys before its vmapped traversal. The deterministic recurrent function does not consume per-game randomness.
- Executed probe: 3,840 derived keys spanning four generations, three waves, five plies, four simulations and 16 games were all unique. A `(64,64)` Gumbel draw had zero duplicate rows and mean cross-row correlation `2.3e-5`. An actual Gumbel search on 64 identical roots produced 28 chosen actions, 64 distinct target rows and 64 distinct visit rows.
- Device scope: the backend uses one ordinary `jax.jit` on the default device; there is no `pmap`, sharding or multi-process device fan-out. Therefore no device namespace is needed today. If multi-device replicas are later added, fold `jax.process_index()` and device/replica index into the base key before generation.

### Item 2 — Top-k occurs after legal masking

- Root Gumbel path: `search.py:194-198` masks illegal logits to `-inf` before `topk_legal`.
- Root PUCT path: `masked_root_logits` masks (and applies any legal-only noise) before `topk_legal`.
- Every child: `search.py:174-177` computes the legal mask and applies it before `topk_legal`.
- Executed adversarial probe: all illegal root logits were set to `1000` and every legal logit to at most `1`; with 414 legal moves and `top_k=64`, both batch rows still returned 64 distinct legal IDs and zero illegal IDs. Padded slots only occur when fewer than K legal actions exist.

### Item 3 — Completed-Q perspective at every depth

- `search.py:166-175` evaluates a transition reward from the parent mover's perspective. A live edge has discount `-1`, a terminal edge has its exact mover-perspective reward and discount `0`, and the child value is predicted for the child player-to-move.
- mctx's `tree.qvalues(node) = reward + discount * child_value` is therefore in that node/parent player's perspective. Its backup applies the stored discount at every edge. `qtransform_completed_by_mix_value` combines those local-perspective Qs with the same node's raw value, so completed values never mix perspectives.
- Executed forced three-ply probe: raw Q signs at depths 0/1/2 were `+0.919 / -0.961 / +1.000`, exactly matching alternating players. The terminal edge resolved to +1 for its mover. The completed-Q transformation preserved the local ordering; a single-action node can rescale to zero without changing a decision.

### Item 4 — Replay capacity and full-pass arithmetic (apart from CODEX-01/02)

- Whole games are evicted atomically by `deque(maxlen=replay_buffer_games)` and resume reconstructs the newest games from persisted `game_sizes`.
- The production run's actual buffer matched its resolved 60,000-game configuration at generation 6 and thereafter. Training throughput row counts exactly match the logged flattened position counts.
- DataLoader uses `drop_last=False` and each position appears once per epoch. At `epochs=2`, `B=60,000`, `F=10,000`, a position resident for six generations is presented 12 times over its full lifetime. No hidden sub-sampling exists.

### Item 5 — Optimizer continuation, LR schedule and weight decay

- `base_wrapper.py:362-371` builds one AdamW parameter group containing all network parameters with configured decay.
- Candidate training mutates the same `nnet` optimizer across generations. Accepted candidates keep moments. Before training, `temp.pth.tar` saves weights plus optimizer and scheduler; rejection restores the pre-candidate Adam moments while intentionally leaving the global scheduler clock advanced. Resume restores moments and scheduler. Cross-run warm-start intentionally calls `load_weights`, resetting optimizer and schedule at the new config LR.
- The LR scheduler is created once for the wrapper and stepped once per epoch; it does not restart each generation. The audited historical run explicitly used a constant `1e-3` LR.
- Focused tests passed (`7 passed`) for global schedulers, warm-start reset, AdamW and decay reassertion. Historical accepted checkpoints empirically contain one group with decay `0`, as expected because that run predates AdamW; optimizer step increased from 1,236 at accepted generation 1 to 129,906 at generation 20, proving moment/step continuation. Current `load_checkpoint` reasserts the resolved config's decay after loading legacy optimizer state, and the test pins every parameter group to `1e-4` under the default config.

### Existing coverage also checked

- JAX/Python legal-mask parity over 5,000 positions and step/result parity tests.
- PUCT terminal root, legal-only support, normalized distributions and root-noise legality.
- Gumbel legal support, chosen-action legality and backend completion.
- Replay game-boundary persistence/resume, full-pass batch counts and memmap equivalence.
- Optimizer checkpoint, rejection and resume semantics.
- Previously declared-clean value-label sign, harvest equivalence, symmetry, checkpoint gate-revert and duplicate-game questions were not re-audited beyond the intersections required above.
