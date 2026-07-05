/**
 * Shared engine types. The UI talks to an `Engine` and never to a concrete
 * backend, so the in-browser engine (rules + ONNX + MCTS) and the local
 * Python server client are interchangeable.
 */

/** +1 = White (first mover, start (4,4) array coords), -1 = Black. */
export type Player = 1 | -1;

/**
 * Blokus Duo game state, mirroring the JAX kernel's `GameState`
 * (games/blokusduo/jax/kernels.py).
 */
export interface GameState {
  /** Signed placement board, 196 cells row-major (top-left origin): +pieceId White, -pieceId Black, 0 empty. */
  ppb: Int8Array;
  /** Piece inventory, 2 player slots x 22 piece ids (index 0 unused, always 0). 1 = still in hand. */
  remaining: Uint8Array;
  /** Last piece id placed per player slot, 0 = none yet. */
  lastPiece: Int8Array;
  currentPlayer: Player;
}

/** Result of `gameResult`: 0 ongoing, +1/-1 win/loss for the queried player, DRAW_VALUE draw. */
export const DRAW_VALUE = 1e-4;

export interface GameStatus {
  isOver: boolean;
  /** Final score per player slot (0 = White, 1 = Black); present once over. */
  scores?: [number, number];
  /** +1 White wins, -1 Black wins, 0 draw; present once over. */
  winner?: number;
}

export interface DifficultyLevel {
  id: string;
  label: string;
  searchPolicy: 'policy' | 'puct' | 'gumbel';
  /** Simulation budget; 0 = raw policy (no search). */
  sims: number;
  description: string;
}

export interface SearchResult {
  action: number;
  /** Net/search value estimate for the side to move, in [-1, 1]. */
  value: number;
  /** Simulations actually run (0 for raw policy). */
  sims: number;
  elapsedMs: number;
}

export interface EngineInfo {
  name: string;
  /** True when moves come from the full-strength local Python stack. */
  isFullStrength: boolean;
  difficulties: DifficultyLevel[];
}

/**
 * The one interface the frontend calls. Implementations: `BrowserEngine`
 * (rules + ONNX net + MCTS in the page) and `ServerEngine` (HTTP client to
 * the local Python server, which answers with the real torch/MCTS stack).
 */
export interface Engine {
  init(): Promise<EngineInfo>;
  /** Legal action ids for the state's current player (pass id included when forced). */
  legalMoves(state: GameState): Promise<number[]>;
  /** Pure state transition; throws on illegal actions. */
  applyMove(state: GameState, action: number): GameState;
  bestMove(state: GameState, difficulty: DifficultyLevel, history: number[]): Promise<SearchResult>;
  gameStatus(state: GameState): GameStatus;
}
