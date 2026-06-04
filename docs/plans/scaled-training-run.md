# First scaled Blokus training run

The F1→F3 optimisation stack (~14× cumulative vs serial) makes ~1,000 self-play games per generation cost the same wall-clock as the old 80. This plan captures the config + reasoning for the first training run that actually exploits that — and the memory constraints that bite when you scale games this hard.

**Status:** Planned, not launched. Intended to kick off at end of day so it doesn't block daytime F4 benchmarking on the PC. Prepared 2026-06-02.

---

## Motivation

The old baseline (`blokus_pc_second`) ran 80 games/gen — far too few for AlphaZero-style learning. Same self-play wall-clock now buys ~1,000 games/gen. The goal of this run: confirm the net actually learns better with an order of magnitude more data per generation, and get a real Elo-vs-gen-0 curve at a sensible scale.

---

## Proposed config

Base: `blokus_pc_second.json`. Changes and rationale:

| Param | Old | New | Why |
|-------|-----|-----|-----|
| `num_eps` | 80 | **1000** | The headline change — exploit the F1-F3 throughput. (May scale back; see memory section.) |
| `num_generations` | 5 | **10** | ~70 min/gen → ~12 h, an overnight run. |
| `num_mcts_sims` | 300 | **300** | Hold, to attribute any gain to the data scaling, not a sims confound. More sims = sharper policy targets but linear cost; revisit (→400-600) in a follow-up. |
| `mcts_batch_size` | (1) | **16** | **Critical — the F3 speedup only happens if this is set.** Defaults to 1. |
| `max_queue_length` | 5000 | **70000** | Per-generation `deque(maxlen=…)`. At ~60k positions/gen, 5000 would silently drop ~92% of each gen's data. Must hold a full generation. |
| `max_generations_lookback` | 10 | **1** | **Measured 2026-06-02** (after the float32 policy fix): ~223 KB/position incl. the train-time copies, so 1000 games × 1 gen (~60k positions) peaks at **~13.4 GB — fits 24 GB**; ×2 gens (~120k) hits ~26.8 GB — over. So lookback=1 at 1000 games. (Higher lookback would need the lazy-Dataset change — deferred — to drop the contiguous copies.) Also: fresher data is better than gens-old weak-net data. |
| `epochs` | 5 | **2** | One epoch = one full pass over the training data updating weights each batch. With ~60k fresh positions/gen, 5 passes risks overfitting that gen + wastes time. 1-2 is the AlphaZero norm when data is plentiful. |
| `temp_threshold` | 6 | **12** | Games are ~24-30 plies; at 6 you explore only the opening then play greedy. 12 keeps midgame diverse — the point of generating many games. |
| `batch_size` (train) | 64 | **256** | Big dataset trains faster with bigger batches; fine on the GPU at 64f×4b. |
| `num_arena_matches` | 50 | **50** | Reliable 55%-threshold gating; cheap now. |
| `elo_games_per_gen` | 20 | **50** | The "is it getting stronger" curve; cheap now. |
| `net` (filters×blocks) | 64×4 | **64×4** | Hold — a bigger net eats the F3 inference gains and adds a confound. Scale data first. |
| `policy_head` | fc | **conv** (default since F4) | The conv head (21.6× fewer params) is now the default — this run is also the de-facto strength validation for it. Doesn't change the memory analysis below (the stored policy *target* is still the dense 17,837-vector). |
| `lr_scheduler` | none | **cosine** | Sensible decay over a 10-gen run. |
| `cpuct` | 2.5 | 2.5 | Unchanged. |
| `seed` | 42 | 42 | Reproducible. |

---

## Memory — resolved (pre-run-prep R1, 2026-06-02)

Scaling games collides with the WSL 24 GB cap. What we did and measured:

- **The fix that landed:** the policy target is now stored **float32** (`core/self_play.py`), not float64 — halves the dominant per-position cost (the 17,837-long policy was ~143 KB as float64). Board planes were already float32.
- **Measured peak:** **~223 KB/position** including `base_wrapper.train`'s transient contiguous `np.array` + tensor copies (the copies, not the buffer, dominate the peak). Linear in positions.
- **Verdict — use `lookback=1` at 1000 games:**
  - 1000 games × 1 gen (~60k positions) → **~13.4 GB peak — fits** (≈10 GB headroom).
  - 1000 games × 2 gens (~120k) → ~26.8 GB — **over** 24 GB.
- **If we later want more replay** (lookback ≥ 2 at 1000 games): the proper fix is a **lazy training Dataset** (`__getitem__` converts each position to a tensor on demand, avoiding the contiguous copy → drops the cost back to ~106 KB/position). Deferred — not needed for the first run.

So the config above uses `num_eps=1000`, `max_generations_lookback=1`, `max_queue_length=70000`. 60k fresh positions/gen is plenty of training signal; replay across gens can come later via the lazy Dataset if training looks unstable.

---

## Arena & Elo in parallel?

They're independent (arena = new vs previous net; Elo = new vs frozen gen-0), so logically concurrent. But both are GPU-bound and the GPU is already saturated by the 8 self-play workers' worth of inference — running the two phases simultaneously would just split the same GPU, not finish faster. They're each already parallelised across workers internally. At 50+50 games × ~4 s ≈ a few minutes total, it's not a bottleneck worth re-architecting. **Leave sequential.**

---

## Pre-flight checklist

- [ ] `mcts_batch_size: 16` set in the config (defaults to 1 — easy to miss).
- [ ] Decide on the float32-policy storage change (recommended) — or accept reduced scale.
- [ ] 1-gen dry run on the PC; record peak RAM; set `lookback`/`num_eps` accordingly.
- [ ] Confirm disk space for 10 gens × ~60k positions of parquet self-play history.
- [ ] Confirm checkpoint/resume behaviour if we want to stop/restart (CLAUDE.md notes a `main.py` checkpoint-loading issue when `load_model: true` — verify before relying on resume).
- [ ] Launch via a **held-open foreground SSH session** (Claude Code background task), NOT `systemd-run`/`nohup` — see `docs/guides/REMOTE-TRAINING.md` (WSL tears the distro down otherwise).

## What to watch during the run

- **Elo vs gen-0** climbing — the headline "is it learning" signal.
- **Policy/value loss** trending down without diverging.
- **Arena accept rate** — if the new net keeps failing the 55% gate, learning has stalled.
- **RAM** during the first real generation (catch an OOM before it wastes hours).

---

## Open / deferred

- **Dirichlet root noise** (`IDEAS.md` I2) is still absent — exploration rests on temperature alone. The highest-value pre-run addition if we want this run to be more than a first look; deferred for now.
- **K=1 vs K=16 strength comparison** (deferred from F3) could be folded in by running one arena with mismatched K on a checkpoint from this run.
- **More MCTS sims** (→400-600) — a follow-up experiment once we've seen the 1000-games-at-300-sims baseline.
