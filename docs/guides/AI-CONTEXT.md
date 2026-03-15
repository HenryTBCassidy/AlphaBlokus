# AlphaBlokus — AI Assistant Context

> Context for Claude and other AI assistants working on this project. Read this alongside the README before starting any task.

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

Residual connections prevent gradient degradation in deeper networks. Blokus needs more capacity than Tic-Tac-Toe's 4-layer CNN. The AlphaZero paper uses ResNet — we follow the proven recipe. Configurable depth (1-8 blocks) lets us experiment without code changes.

### Why 44x14x14 board encoding?

The network needs both spatial state (where pieces are) and piece inventory (which pieces remain). The 44-channel encoding follows the AlphaZero convention of one binary plane per piece type per player: 21 planes per player showing exactly where each piece sits on the board, plus 2 aggregate planes (union of all current/opponent pieces). Piece inventory is implicit -- an all-zero plane means that piece hasn't been played. Conv filters see which specific piece occupies each position, enabling piece-shape-aware spatial reasoning. See `../plans/archive/board-encoding-options.md` for the full design rationale and alternatives considered.

### Why 91 piece-orientations (not 168)?

Each piece has up to 8 orientations (4 rotations x 2 flips), but many are duplicates due to symmetry. The 1-square piece has only 1 unique orientation. A symmetric 2x2 square has only 1. Asymmetric pentominoes have all 8. After symmetry reduction: 91 unique orientations. This keeps the action space as small as possible while being complete.

### Why game-agnostic framework?

The `IGame` and `INeuralNetWrapper` protocols let us add new games without touching MCTS, Coach, or Arena. Tic-Tac-Toe validated the framework before we invested time in Blokus game logic. Could extend to Connect 4, Othello, or other games later.

### Why no parallel MCTS yet?

Premature optimisation. Get the pipeline working correctly first, then optimise. The current bottleneck is implementing move generation, not MCTS speed. Parallelism is documented in `../02-ALGORITHMS.md` for when we're ready.

### Why canonical form = channel reordering?

The canonical form is handled by `board.as_multi_channel(current_player)`, which places the current player's 21 piece planes in channels 0-20 and the opponent's in channels 21-41. The network always sees "my pieces" in the first channel group and "opponent's pieces" in the second, regardless of which colour it's playing. This replaces the old approach of multiplying a single-channel board by the player value (+/-1).

---

## Gotchas

1. **Move generation is the hard part.** The board, pieces, placement validation, and neural network are all done. The blocker is efficiently generating legal moves from the placement point cache. This was the predicted bottleneck in the project plan and it proved correct.

2. **The action space is huge.** 17,837 actions means the policy head's final FC layer alone is ~9M parameters. MCTS iterating over all actions (even illegal ones) is O(17,837) per simulation. Must optimise to only iterate valid moves.

3. **`state_key` uses `_piece_placement_board.tobytes()`.** MCTS hashes board states using the `state_key` property on `IBoard`. For BlokusDuo this uses the piece placement board's raw numpy byte representation (196 bytes, int8).

4. **Piece-orientation IDs have gaps.** The `BidirectionalDict` assigns IDs sequentially but skips IDs when moving to the next piece (the populate_lookup loop increments `i` one extra time between pieces). Be careful when mapping between action indices and piece-orientation IDs.

5. **No optimizer state in checkpoints.** The wrapper saves only `state_dict`, not the Adam optimizer's momentum buffers. This means training "restarts" each generation. Acceptable for now, but saving optimizer state would give smoother training.

6. **`calc_conv2d_output` exists but is only used once.** The Blokus ResNet uses it to compute the flattened size for the FC layers. The Tic-Tac-Toe net hardcodes the computation. If you change kernel sizes or padding, use this helper.

7. **Coordinate systems are confusing.** Board coordinates use bottom-left origin (matching standard Blokus notation). Array indices use top-left origin (matching numpy). `CoordinateIndexDecoder` handles conversion. Always be explicit about which system you're in.

8. **`eval.ipynb` has move generation design notes.** Check the notebook for Henry's thinking about the move generation algorithm before implementing.

---

## Things NOT to Do

- **Don't rewrite the framework.** MCTS, Coach, Arena work correctly — they're validated on Tic-Tac-Toe. Focus on completing the Blokus game logic
- **Don't over-engineer move generation.** Get it working correctly first, optimise second. A correct but slow `valid_moves()` is infinitely better than a fast but buggy one
- **Don't add 4-player Blokus support until Duo beats Pentobi level 9.** It's a stretch goal, not a current priority. 4-player would require fundamental changes (coalition dynamics, non-zero-sum) — see `docs/01-BACKGROUND.md` for details
- **Don't switch to Transformer/Mamba/ViT.** ResNet is proven for AlphaZero. Novel architectures are research risk we don't need
- **Don't implement learning rate scheduling before the first successful training run.** Get the basics working, then refine
- **Don't build a web UI.** The HTML reporting that already exists is sufficient for analysis
- **Don't pad options for educational value.** If one approach clearly works, recommend it. Henry will push back on unnecessary alternatives
- **Read the README first** — it has the current project status

---

## Where to Start

- **`README.md`** — Project overview and current status
- **This file** (`docs/guides/AI-CONTEXT.md`) — AI context, architecture decisions, and gotchas
- **The relevant `docs/` file** for whatever work is being done (e.g., `docs/02-ALGORITHMS.md` for MCTS work, `docs/03-NEURAL-NETWORKS.md` for net work)
- **`docs/plans/bug-fixes.md`** — The active bug list
