/**
 * Game controller: owns the game state, drives turns against the engine, and
 * notifies the view after every change. Pure state machine — no DOM here.
 */

import { initialState } from '../engine/rules';
import type {
  DifficultyLevel,
  Engine,
  EngineInfo,
  GameState,
  GameStatus,
  Player,
  SearchResult,
} from '../engine/types';

export type Phase = 'humanTurn' | 'engineTurn' | 'over';

export interface ThinkingProgress {
  done: number;
  total: number;
}

export class GameController {
  state: GameState = initialState();
  history: number[] = [];
  humanPlayer: Player = 1;
  difficulty: DifficultyLevel;
  phase: Phase = 'humanTurn';
  legal: number[] = [];
  status: GameStatus = { isOver: false };
  thinking: ThinkingProgress | null = null;
  lastEngineMove: SearchResult | null = null;
  resigned = false;
  /** Guards against a stale engine reply landing after New Game. */
  private generation = 0;

  constructor(
    readonly engine: Engine,
    readonly info: EngineInfo,
    private readonly onUpdate: () => void,
  ) {
    this.difficulty = info.difficulties[Math.floor(info.difficulties.length / 2)]!;
  }

  async newGame(humanPlayer: Player, difficulty: DifficultyLevel): Promise<void> {
    this.generation++;
    this.state = initialState();
    this.history = [];
    this.humanPlayer = humanPlayer;
    this.difficulty = difficulty;
    this.phase = this.state.currentPlayer === humanPlayer ? 'humanTurn' : 'engineTurn';
    this.status = { isOver: false };
    this.thinking = null;
    this.lastEngineMove = null;
    this.resigned = false;
    this.legal = await this.engine.legalMoves(this.state);
    this.onUpdate();
    if (this.phase === 'engineTurn') await this.runEngineTurn();
  }

  /** Apply the human's chosen action (already validated against `legal`). */
  async humanMove(action: number): Promise<void> {
    if (this.phase !== 'humanTurn' || !this.legal.includes(action)) return;
    await this.applyAction(action);
    if (!this.status.isOver) await this.runEngineTurn();
  }

  resign(): void {
    if (this.status.isOver) return;
    this.generation++; // cancel any in-flight engine turn
    this.resigned = true;
    this.phase = 'over';
    this.status = { isOver: true, winner: -this.humanPlayer };
    this.thinking = null;
    this.onUpdate();
  }

  private async applyAction(action: number): Promise<void> {
    this.state = this.engine.applyMove(this.state, action);
    this.history.push(action);
    this.status = this.engine.gameStatus(this.state);
    if (this.status.isOver) {
      this.phase = 'over';
      this.legal = [];
    } else {
      this.legal = await this.engine.legalMoves(this.state);
      this.phase = this.state.currentPlayer === this.humanPlayer ? 'humanTurn' : 'engineTurn';
    }
    this.onUpdate();
  }

  private async runEngineTurn(): Promise<void> {
    const generation = this.generation;
    while (this.phase === 'engineTurn' && !this.status.isOver) {
      // Forced single option (a pass): play immediately, no search theatre.
      if (this.legal.length === 1) {
        await this.applyAction(this.legal[0]!);
        continue;
      }
      this.thinking = { done: 0, total: this.difficulty.sims };
      this.onUpdate();
      const result = await this.engine.bestMove(this.state, this.difficulty, this.history);
      if (generation !== this.generation) return; // New Game/resign superseded us
      this.thinking = null;
      this.lastEngineMove = result;
      await this.applyAction(result.action);
    }
  }

  reportProgress(done: number, total: number): void {
    if (this.thinking) {
      this.thinking = { done, total };
      this.onUpdate();
    }
  }
}
