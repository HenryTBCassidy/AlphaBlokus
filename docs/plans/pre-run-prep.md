# Pre-run prep

Quick plan: the few things worth doing **before** committing to a serious multi-hour Blokus training run, so we don't waste it on an OOM, an under-explored search, or an unvalidated loop. Companion to [`scaled-training-run.md`](scaled-training-run.md) (the big run this leads into) and the now-complete [`full-cycle-optimisation.md`](full-cycle-optimisation.md).

Two must-dos (memory, Dirichlet), one optional speed nicety (fp16), then a short end-to-end validation run that exercises the whole loop with everything on.

---

## Checklist

| # | Item | Effort | Priority | Done |
|---|------|--------|----------|------|
| R1 | **Memory fix** — store the training-example policy as float32 (not float64); confirm board dtype too. Then a 1-gen dry run measuring peak RSS at scale, and set `max_queue_length` / `max_generations_lookback` / `num_eps` from what it shows | 1.5-2 hr | High | |
| R2 | **Dirichlet root noise** — add `P(s,a) = (1-ε)·p + ε·η`, `η ~ Dir(α)` at the root, self-play only, off at arena/Elo. Config-gated (`dirichlet_epsilon`, `dirichlet_alpha`), default ε=0 → bit-identical when off | 2-3 hr | High | See [R2 detail](#r2-dirichlet-root-noise) |
| R3 | **fp16 inference** (optional, speed only — *not* required for a valid run). Wrap `predict`/`predict_batch` forward in `torch.autocast` on CUDA, behind a `fp16_inference` flag | 1 hr | Low / optional | |
| R4 | **End-to-end validation run** — short run (2 gens, ~20 eps, 8 workers, conv head, F2+F3 on, Dirichlet on) of the *full Coach loop* on the PC. Confirm: no crash/OOM/NaN, Elo computed, checkpoints save/load across gens, W&B logs. The first time the whole stack runs together | 1 hr + ~40 min run | High | |

Total: ~6 focused hours + the validation run. R3 is skippable.

---

## R1. Memory fix

The blocker from [`scaled-training-run.md`](scaled-training-run.md): each training example stores the dense 17,837-long policy, and it's currently **float64 (~143 KB/position)** because `get_action_prob` returns Python floats that `np.asarray` widens. At 1000 games × 2 symmetries ≈ 60k positions/gen, the replay buffer + the contiguous copy `base_wrapper.train` builds overflow the 24 GB WSL cap.

**Fix:** cast the policy to **float32** when the example is assembled (`core/self_play.py`, where the `(board, pi, value)` tuple is built) — training casts to float32 anyway, so nothing is lost; this halves the dominant cost. Also confirm `as_multi_channel`'s dtype and cast to float32 if it comes back float64.

**Then measure, don't guess:** a 1-generation dry run at the intended scale with a peak-RSS probe (`/usr/bin/time -v` or a `psutil` sampler). Set `max_queue_length` (hold one full gen), `max_generations_lookback` (2-3), and `num_eps` from the actual peak. Record the numbers in `scaled-training-run.md`.

---

## R2. Dirichlet root noise

Standard AlphaZero exploration we currently lack. At the **root only**, during **self-play only**, mix Dirichlet noise into the legal-move priors:

```
P(s,a) = (1 - ε)·p_a + ε·η_a,   η ~ Dirichlet(α)   over the legal actions
```

**Parameters (config-gated on `MCTSConfig`):**
- `dirichlet_epsilon: float = 0.0` — mixing weight; **0 disables** (so existing behaviour is bit-identical until turned on). AlphaZero used 0.25.
- `dirichlet_alpha: float = 0.03` — concentration. Rule of thumb α ≈ 10 / (typical legal moves); Blokus's ~400 early legal moves is Go-like (AlphaGo Zero used 0.03), so 0.03 is a sane start.

**Wiring:** add an explicit `add_root_noise: bool = False` param to `MCTS.get_action_prob`. `core/self_play.py` passes `add_root_noise=True`; arena/Elo (`NetworkPlayer`) leave it `False`. Noise is applied once per move, after the root is expanded (its priors set), before the sims — to `policy_priors[root_key]`, renormalised over legal moves. Uses the episode's seeded RNG, so it stays reproducible.

**Tests:** ε=0 → root priors unchanged (bit-identical); ε>0 → priors still sum to 1 and stay zero on illegal moves; same seed → same noisy priors (determinism). Self-play applies it, arena doesn't.

**Why before the run:** without root noise, early generations (weak prior + 300 sims over 400+ legal moves) under-explore, risking a run that learns poorly and a misleading "it doesn't work." Cheap, standard, high-leverage.

---

## R3. fp16 inference (optional — skip for the first run)

Half-precision forward pass: wrap the forward in `torch.autocast(device_type="cuda", dtype=torch.float16)` inside `predict` / `predict_batch`, **only on CUDA**, behind a `net_config.fp16_inference: bool = False` flag. ~1.3–1.8× on the inference slice on the 3060 Ti, compounding with F3. Inference-only, so no GradScaler / training-stability concerns; slight numerical differences are fine for self-play.

**Not needed for a valid run** — pure speed. Recommended to **defer**: do the first serious run without it, and revisit in a later speed pass (alongside Option B) if self-play throughput is the thing to attack. Listed here only so it's captured.

---

## R4. End-to-end validation run

The benchmarks ran self-play/arena/Elo as isolated phases; the **full Coach training loop** (self-play → train → arena → Elo → checkpoint → next gen) with **F1 + F2 + F3 + F4 + Dirichlet all on at once** has never actually run. Before a 12-hour run, do a short one:

- Config: 2 generations, ~20 eps, 300 sims, 8 workers, conv head (default), `use_optimised_movegen: true`, `mcts_batch_size: 16`, Dirichlet on, `load_model: false` (fresh — old FC checkpoints won't load into the conv net).
- Run on the PC via a **held-open SSH session** (not `systemd-run` — see `docs/guides/REMOTE-TRAINING.md`).
- **Confirm:** runs to completion; no crash / OOM / NaN; Elo-vs-gen0 is computed; checkpoints save and load across the generation boundary; W&B logs the run.

If it's clean, the scaled run (`scaled-training-run.md`) is go.

---

## Out of scope

- The big scaled run itself — that's `scaled-training-run.md`, gated on R4 passing.
- Further speed work (Option B cross-worker server, adaptive sim budget) — reopen a fresh optimisation plan later if runs show we need M3.
- Pentobi benchmarking — matters eventually, not for a first "does it learn" run.
