# AlphaBlokus — AI Assistant Context

> Extended context for Claude and other AI assistants working on this project. Read this alongside the README and `AGENTS.md` before starting any task. AGENTS.md carries the operational rules (commands, gotchas, doc map); this file carries the *why* — architecture rationale and background that doesn't fit a rule list.

---

## Who Is Henry?

- **Background:** Data scientist at Transak (trading and pricing). Previously quantitative researcher at G-Research (London quant fund)
- **Technical skills:** Strong Python, software engineering, statistics. Has built an AlphaZero tic-tac-toe implementation (this project) and a mini Shakespeare GPT (character-level transformer)
- **ML experience:** Conceptual understanding is solid. Hands-on production ML training experience is limited — this is his first major ML project beyond tutorials
- **Tools:** Unlimited Claude and Codex access. Uses AI assistance heavily for implementation
- **Communication style:** Results-oriented. Dislikes overcomplication and tutorial-mode explanations. Will push back if you're adding unnecessary complexity. Prefers "do the thing that works best" over "let's try everything for learning"
- **Game knowledge:** Understands Blokus from a player's perspective but isn't a competitive player

---

## Key Architecture Decisions (and Why)

### Why ResNet over plain CNN?

Residual connections prevent gradient degradation in deeper networks. Blokus needs more capacity than Tic-Tac-Toe's 4-layer CNN. The AlphaZero paper uses ResNet — we follow the proven recipe. Configurable depth and width (`num_residual_blocks`, `num_filters`) let us experiment without code changes. The policy head is fully convolutional by default (F4): the action space is `(cell, orientation)` pairs, so a 1×1 conv emitting one logit plane per orientation costs ~47K params where the legacy FC head cost ~7M.

### Why 44x14x14 board encoding?

The network needs both spatial state (where pieces are) and piece inventory (which pieces remain). The 44-channel encoding follows the AlphaZero convention of one binary plane per piece type per player: 21 planes per player showing exactly where each piece sits on the board, plus 2 aggregate planes (union of all current/opponent pieces). Piece inventory is implicit — an all-zero plane means that piece hasn't been played. Conv filters see which specific piece occupies each position, enabling piece-shape-aware spatial reasoning. See `../plans/archive/board-encoding-options.md` for the full design rationale and alternatives considered.

### Why 91 piece-orientations (not 168)?

Each piece has up to 8 orientations (4 rotations x 2 flips), but many are duplicates due to symmetry. The 1-square piece has only 1 unique orientation. A symmetric 2x2 square has only 1. Asymmetric pentominoes have all 8. After symmetry reduction: 91 unique orientations. This keeps the action space as small as possible while being complete. `OrientationCodec` maps `(piece_id, orientation)` to **contiguous** integer IDs 0–90 — no gaps.

### Why game-agnostic framework?

The `IBoard` / `IGame[TBoard]` / `INeuralNetWrapper` protocols let us add new games without touching MCTS, Coach, or Arena. Tic-Tac-Toe validated the framework before we invested time in Blokus game logic, and it remains the fast regression game. The coupling to concrete games is concentrated in exactly one module — `alphablokus/registry.py`, the composition root — so "game-agnostic" is enforced, not aspirational.

### Why two self-play backends (python and jax), and why both are first-class

Self-play generation dominates wall-clock. The **python engine** (`search/mcts.py` + `selfplay/episode.py` + `parallel/pool.py`) is the framework itself: it drives arena, Elo, and Pentobi evaluation on every run, it is the only engine for TicTacToe, and it is the correctness oracle. The **jax backend** (`games/blokusduo/jax/`) reimplements Blokus self-play GPU-natively — rules as int8 matmuls, mctx search over a top-K compact action space, an inference-only jnp net bridged from the torch checkpoint each generation — and at production net size generates games ~12× faster (`docs/research/jax-pipeline-ab.md`). Production Blokus configs use jax + Gumbel search; everything else stays python. Neither is legacy; don't quarantine either.

**The python↔jax parity contract** (what keeps the fast path honest):

- The jax **rules kernels** are bit-identical to the python rules engine — parity tests sweep thousands of stratified positions comparing legal-move masks, terminal detection, and encodings.
- The jax **PUCT search** is tuned to match the python search's arithmetic (raw-Q transform, cpuct mapping — see `games/blokusduo/jax/search.py`'s module docstring), and its move agreement is measured against the exact search; at K=64 it tracks it *better* than the python engine's own K=16 virtual-loss batching.
- The **harvester** (`jax/harvest.py`) emits training examples in the *exact* representation `selfplay/episode.py` produces (compact canonical boards, sparse policies, same draw-sign convention), so storage, resume, and training are backend-agnostic.
- **Gumbel mode is a deliberate behavioural change** (different policy target: completed-Q improved policy, no Dirichlet/temperature), opt-in via `search_policy: "gumbel"` and validated by its own A/B training run, not by parity.

### Why canonical form = channel reordering?

The canonical form is handled by `board.as_multi_channel(current_player)`, which places the current player's 21 piece planes in channels 0-20 and the opponent's in channels 21-41. The network always sees "my pieces" in the first channel group and "opponent's pieces" in the second, regardless of which colour it's playing. This replaces the old approach of multiplying a single-channel board by the player value (+/-1).

### Why a rolling game-sized replay buffer?

Training uses a `deque` of the last `replay_buffer_games` games (compact boards + sparse policies), trained with full epoch passes — no sampling knob. Staleness and reuse fall out of `B`, `F` (`num_eps`) and `E` (`epochs`): see `docs/02-ALGORITHMS.md`. This replaced a generation-window design whose window could never shrink below 5 generations, and it is what lifted the training-step RAM ceiling (compact boards are ~175× smaller than dense planes).

---

## Gotchas (context-level; the operational list lives in AGENTS.md)

1. **`state_key` uses `_piece_placement_board.tobytes()`.** MCTS hashes board states using the `state_key` property on `IBoard`. For BlokusDuo this uses the piece placement board's raw numpy byte representation (196 bytes, int8). Anything that changes the placement board's dtype/layout silently invalidates the tree keys.

2. **Draws are `1e-4`, not `0`.** `get_game_ended` returns `0` for "game still running", so the draw sentinel is `1e-4`; value targets for drawn games are `±1e-4 ≈ 0`. The jax kernels reproduce the same convention (`DRAW_VALUE`).

3. **Coordinate systems are confusing.** Board coordinates use bottom-left origin (matching standard Blokus notation). Array indices use top-left origin (matching numpy). `CoordinateIndexDecoder` (in `games/blokusduo/codec.py`) handles conversion. Always be explicit about which system you're in.

4. **Internal Elo curves are not cross-comparable.** Each run's Elo is anchored to its own random gen-0 net. A run can trail another on internal Elo and still beat it head-to-head (observed in the jax A/B). Head-to-head arenas are the arbiter between runs.

5. **Checkpoints carry optimizer + scheduler state.** `save_checkpoint` stores `state_dict` *and* `optimizer_state_dict` (+ scheduler when configured), so Adam momentum survives resume. Only `state_dict` keys are bridged to the jax net.

6. **Performance claims go stale fast here.** Any "X dominates the profile" claim is only true at a specific net size and backend — the move-gen/inference split flipped once already (see `docs/08-TRAINING-ESTIMATES.md`'s banner), and the jax backend changed the workload shape again. Cite `docs/research/` measurements, not folklore.

---

## Things NOT to Do

- **Don't rewrite the framework.** MCTS, Coach, Arena, move generation, and the game rules are validated — relocate or extend, never re-derive
- **Don't add 4-player Blokus support until Duo beats Pentobi level 9.** It's a stretch goal, not a current priority. 4-player would require fundamental changes (coalition dynamics, non-zero-sum) — see `../01-BACKGROUND.md` for details
- **Don't switch to Transformer/Mamba/ViT.** ResNet is proven for AlphaZero. Novel architectures are research risk we don't need
- **Don't build a web UI.** The HTML reporting that already exists is sufficient for analysis
- **Don't pad options for educational value.** If one approach clearly works, recommend it. Henry will push back on unnecessary alternatives
- **Don't trust stale docs over code.** When a doc and the source disagree, the source wins — and fix the doc

---

## Where to Start Reading

Follow the pipeline top-down; each hop is one seam:

1. **`src/alphablokus/cli.py`** — entry point: config load, resume/report-only dispatch, Coach construction via `registry.instantiate_game_and_network`.
2. **`src/alphablokus/training/coach.py`** — one generation end-to-end: self-play → train → arena → strength eval → flush metrics.
3. **`src/alphablokus/selfplay/generate.py`** — the backend dispatch (serial / worker pool / jax) that Coach's phase 1 collapses into.
4. From there, follow your task: `search/mcts.py` (the search), `games/blokusduo/game.py` + `movegen/` (rules + move generation), `games/blokusduo/jax/` (the GPU backend), `evaluation/` (arena/Elo/acceptance), `storage/` + `reporting/` (parquet + report).

Supporting reading: `README.md` (status), the relevant `docs/` reference file for the domain you're touching, and `docs/plans/` for what's in flight.
