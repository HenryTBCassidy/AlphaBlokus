# DeepMind self-play run configurations (AlphaGo → MuZero)

A grounded reference for the run-configuration numbers DeepMind actually used across their self-play papers — hardware, data volume, generation structure, MCTS depth, network size, and wall-clock — plus what each choice means for **AlphaBlokus**. Compiled 2026-06-20 from the primary papers and their Methods/supplementary sections (citations at the bottom).

Why this lives in `research/` rather than a notebook: every one of these knobs maps onto a decision we make in `RunConfig` (generations, games/gen, `num_mcts_sims`, net size, replay window, gating). We used this table directly to design `blokus_run2_bignet`. Keep it close.

> **Caveat on precision.** Where a figure is stated in a paper it's cited. Two recurring gaps: **parameter counts are not given in any of these papers** (the "~23M for 20×256" figure is a widely-repeated community estimate, not from the text), and several AlphaZero/MuZero training details ("identical to AlphaGo Zero") are inherited rather than restated — flagged inline.

---

## At a glance

| | AlphaGo (2016) | AlphaGo Zero (2017) | AlphaZero (2018) | MuZero (2020) |
|---|---|---|---|---|
| **Seeded from humans?** | **Yes** (KGS games) | No — tabula rasa | No — tabula rasa | No — tabula rasa |
| **Hardware** | Single: 48 CPU / 8 GPU. Match: 1,202 CPU / 176 GPU. SL trained on 50 GPUs. *No TPUs.* | Train: 64 GPU + 19 param servers. Inference: 4 TPU. | **5,000 first-gen TPUs** (self-play) + **64 second-gen TPUs** (train). | Board: **1,000 TPU** self-play + **16 TPU** train. Atari: 32 + 8. |
| **Total self-play games** | ~1.28M RL (derived) + 30M positions for value net | **4.9M** (20-block) / 29M (40-block) | **chess 44M · shogi 24M · Go 21M** | 1M-game rolling buffer (board) |
| **Generation structure** | 3 stages: SL → RL self-play → value net | **Gated**: best-player generates 25k games/iter; gate = 400 games, >55% | **Continuous**, no gate, single net updated continually | Continuous actor/learner |
| **Training steps** | SL 340M (batch 16); value 50M (batch 32) | **700k** × 2,048 (20b); 3.1M × 2,048 (40b) | **700k** × 4,096 | **1M** × 2,048 (board) |
| **MCTS sims/move** | ~100k (value-net config); 5 s/move in matches | **1,600** | **800** | 800 (board) / 50 (Atari) |
| **Network** | 13-layer **conv**, 192 filters, 19×19×48 in | 20×256 (& 40×256) **resnet**, dual head, 19×19×17 in | 20×256 resnet (per AGZ) | 3 nets (repr/dynamics/predict), 16×256 |
| **Replay window** | value net: 30M *distinct* games, 1 position each | most recent **500,000 games** | (500k, inherited; not restated) | most recent **1M games** |
| **Wall-clock** | SL ~3 wk + value ~1 wk | 20b: **3 days** (beat Lee at 36h). 40b: 40 days | chess **9h** · shogi **12h** · Go **34h** | not published (board) |

---

## AlphaGo (2016) — the human-seeded one

The only one bootstrapped from human play, and the relevant template if we ever seed from Pentobi.

- **Hardware.** Single-machine config: 40 search threads, **48 CPUs, 8 GPUs**. Distributed (match) config: **1,202 CPUs, 176 GPUs** (~250 Elo stronger). The SL policy net was trained on **50 GPUs** for ~3 weeks. No TPUs in this generation.
- **Supervised seeding.** SL policy net trained on **29.4M positions from 160,000 KGS games** (6–9 dan players), reaching **57.0%** move-prediction accuracy (prior SOTA 44.4%). 340M training steps, mini-batch 16.
- **RL self-play.** Policy improved by self-play against a **pool of previous iterations** (anti-overfit), mini-batch = 128 games, pool refreshed every 500 iterations, run for 10,000 mini-batches → **~1.28M games** (derived from the cadence, not stated). The value net was trained on **30M positions, each from a *distinct* self-play game (one position per game)** — a deliberate decorrelation trick (training on full games overfit badly: MSE 0.37 → 0.23).
- **MCTS.** ~100,000 simulations in the headline value-net configuration; tournament play at 5 s/move. Mixing parameter λ=0.5 (value net + rollouts).
- **Network.** 13-layer **convolutional** policy net, 192 filters, 19×19×48 input planes. Value net = same conv body + an extra plane, 256-unit FC, tanh scalar output. Not residual.

**For us:** the decorrelation lesson is the one to carry — *one value label per game is highly correlated within a game*, which is exactly why our value head is data-starved (see [[deepmind-run-configs]] § "what this means"). And it's the proof that **seeding from a strong teacher works** before any self-play.

---

## AlphaGo Zero (2017) — pure self-play, gated

Architecturally the closest to our framework (alpha-zero-general lineage): discrete generations with a best-vs-candidate arena gate.

- **Hardware.** Optimisation on **64 GPU workers + 19 CPU parameter servers**; the three components (self-play, optimisation, evaluator) run asynchronously. The final trained agent runs inference on **4 TPUs** (AlphaGo Lee used 48).
- **Data & cadence.** **4.9M** self-play games for the 20-block run (**29M** for the 40-block). The current best player generates **25,000 games per iteration**; a checkpoint is taken **every 1,000 training steps**; the evaluator plays the candidate **400 games** against the current best and promotes it only on a **>55% win margin** (the "gate"). Training samples uniformly from the **most recent 500,000 games**.
- **Training.** 20-block: **700,000 mini-batches of 2,048**; 40-block: 3.1M × 2,048. SGD, momentum 0.9, L2 = 1e-4, learning-rate annealing (~1e-2 → 1e-4; the exact table is an image in the paper).
- **MCTS.** **1,600 simulations/move** (~0.4 s). Dirichlet noise η ~ Dir(**0.03**), ε = **0.25**. Temperature τ=1 for the first 30 moves, then τ→0. (`c_puct` value is not stated in the paper.)
- **Network.** 20×256 and 40×256 residual towers, dual policy+value heads on a shared trunk. Input **19×19×17** = 8 history planes per player + 1 colour-to-play plane. Policy head = 362 logits.
- **Wall-clock.** 20-block: **3 days** (surpassed AlphaGo Lee at 36h). 40-block: 40 days, final Elo ~5,185.

**For us:** this is our structural cousin — the **gate (400 games, >55%)** and the **rolling replay window** are the two ideas we inherit. Our window is meant to be a smaller version of their 500k-game window; note the bug that pins ours at a rolling 5 generations regardless of config ([[project_window_shrink_bug]]).

---

## AlphaZero (2018) — general, continuous, no gate

The generalisation to chess/shogi/Go, and the one that **deleted the gating machinery**.

- **Hardware.** "**5,000 first-generation TPUs** to generate self-play games and **64 second-generation TPUs** to train." (The "16 TPUs" figure floating around is a misquote — it's 64.) A farm; one training instance per game (3 separate runs).
- **Data & structure.** Training games: **chess 44M, shogi 24M, Go 21M**. The key change from AGZ: *"AlphaZero simply maintains a single neural network that is updated continually … omitting the evaluation step and the selection of best player."* **No gate, no best-vs-candidate** — self-play always uses the latest weights, fed continuously into the replay buffer.
- **Training.** **700,000 steps, mini-batch 4,096.** LR 0.2 → 0.02 → 0.002 → 0.0002 (stepped). SGD momentum 0.9 (per AGZ, not restated).
- **MCTS.** **800 simulations/move.** Dirichlet α = **{0.3 chess, 0.15 shogi, 0.03 Go}** — scaled inversely to the typical legal-move count of each game.
- **Network.** 20×256 residual tower (per AGZ). Input planes: Go 17, chess 119, shogi 362. Policy outputs: **chess 4,672** (8×8×73), **shogi 11,259**, **Go 362**.
- **Wall-clock (700k steps each).** chess **9h**, shogi **12h**, Go **34h**. Surpassed prior SOTA at: chess 4h/300k steps, shogi <2h/110k, Go 8h/165k.

**For us:** two takeaways. (1) The **Dirichlet α scales with branching** — Blokus's ~400–500 opening legal moves are Go-like, so our α=0.03 is the right ballpark. (2) Continuous training is the destination but **needs separate self-play/train hardware**; on one GPU we keep the synchronous gated loop and only *simulate* continuity via a rolling buffer (IDEAS I4 / `replay-buffer-refactor.md`).

---

## MuZero (2020) — learned model

Included for completeness; the conclusion is "not for us."

- **Hardware.** Board games: **1,000 TPU self-play + 16 TPU train**. Atari: 32 + 8 (far fewer actors — only 50 sims/move vs 800).
- **Data.** Board: rolling buffer of the most recent **1M games**. Atari: most recent 125k sequences of length 200. Continuous actors feed the buffer.
- **Training.** **1M mini-batches.** Batch **2,048** (board) / 1,024 (Atari). **Unroll K=5**; loss = value + reward + policy + L2.
- **MCTS.** 800 sims (board) / 50 (Atari). Crucially, search runs **entirely on a learned dynamics model** — the tree never touches the real game rules.
- **Network.** Three nets — **representation** h (obs → hidden state), **dynamics** g (state, action → reward, next state), **prediction** f (state → policy, value). Board games: 16 residual blocks, 256 planes.

**For us: skip it.** MuZero exists to plan *without* a simulator (its whole point is Atari, where the rules are unknown). We have a perfect, cheap Blokus simulator, so the representation+dynamics machinery is pure overhead — it adds approximation error and triples the network count for zero benefit. **AlphaZero is the correct template when you own the rules.**

---

## What this means for AlphaBlokus

The numbers above span a ~1,000× hardware gulf from our single RTX 3060 Ti, so they're not targets to match — they're a map of *which knobs matter and roughly where the sweet spots sit*. The distilled principles:

1. **MCTS sims/move is "depth."** Superhuman runs used 800–1,600; small reimplementations use 25–200. We run 800 (with a branching taper). Relative to Blokus's ~400–500 opening branching, even 800 is thin in the opening — see [I1 in IDEAS](../IDEAS.md).
2. **Sample reuse (~1.5×) is a symptom of being data-rich, not a target.** AlphaZero's low reuse fell out of having 5,000 TPUs out-generating 64 training TPUs. We are the opposite (data-poor, self-play-bound), so we *cannot* copy 1.5× — the real lever for us is **more games per generation**, not tuning reuse. Value targets especially: one noisy outcome shared across ~30 positions, so value learning is bottlenecked by **game count**.
3. **Replay window = data efficiency.** AGZ's 500k-game window (~20 iterations) is what made 4.9M games go far. Our analogue is `max_generations_lookback` — currently buggy (pinned at a rolling 5 generations; [[project_window_shrink_bug]]), being fixed in `replay-buffer-refactor.md`.
4. **Gating is optional insurance.** AGZ used it (400 games, >55%); AlphaZero dropped it at scale. In our small/noisy regime we **keep it** — it's cheap protection against a regressed net poisoning self-play.
5. **Network size: 20×256 (~23M params, est.) was the workhorse.** Ours is tiny by comparison (64×4 ≈ 1M; run2 bumps to 128×8). Bigger nets learn slower per generation but raise the ceiling — judge them late, not early.
6. **Seeding works.** AlphaGo bootstrapped from 160k human games before any RL. Our equivalent is seeding from **Pentobi** — pragmatic for beating it faster, at the cost of tabula-rasa purity. Self-play provides the ceiling; seeding just skips the painful random-play start.
7. **Blokus is far smaller than Go.** Shorter games (~28 plies vs ~250), lower late-game branching, scoring rather than checkmate, and Pentobi is "strong amateur," not "world champion." The bet (see [`docs/08-TRAINING-ESTIMATES.md`](../08-TRAINING-ESTIMATES.md)) is that we need **nowhere near** AGZ-scale data to beat Pentobi.

---

## Sources

- **AlphaGo** — Silver et al., *Mastering the game of Go with deep neural networks and tree search*, Nature 529:484–489 (2016). [Official PDF](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf)
- **AlphaGo Zero** — Silver et al., *Mastering the game of Go without human knowledge*, Nature 550:354–359 (2017). [Nature](https://www.nature.com/articles/nature24270)
- **AlphaZero** — Silver et al., *A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play*, Science 362:1140–1144 (2018); preprint [arXiv:1712.01815](https://arxiv.org/abs/1712.01815).
- **MuZero** — Schrittwieser et al., *Mastering Atari, Go, chess and shogi by planning with a learned model*, Nature 588:604–609 (2020); preprint [arXiv:1911.08265](https://arxiv.org/abs/1911.08265).
