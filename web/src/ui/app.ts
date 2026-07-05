/**
 * The view: board SVG, piece trays, controls. Renders from the controller's
 * state and forwards interactions to it. No engine logic here.
 */

import type { LoadedAssets } from '../engine/tables';
import type { DifficultyLevel, GameState, Player } from '../engine/types';
import type { GameController } from './controller';
import type { Grid, OrientationMaps } from './orientation';
import { BOARD_SIZE, buildOrientationMaps, encodeAction } from './orientation';

const SVG_NS = 'http://www.w3.org/2000/svg';
const CELL = 36;
const MARGIN = 26;
const REPO_URL = 'https://github.com/HenryTBCassidy/AlphaBlokus';
const FULL_STRENGTH_GUIDE_URL = `${REPO_URL}#play-against-the-net`;

interface Selection {
  pieceId: number;
  orientationId: number;
}

export class AppView {
  private readonly maps: OrientationMaps;
  private selection: Selection | null = null;
  private hoverCell: number | null = null;
  private controller!: GameController;

  // Static skeleton elements, created once in mount().
  private boardCellsLayer!: SVGGElement;
  private overlayLayer!: SVGGElement;
  private statusEl!: HTMLElement;
  private scoresEl!: HTMLElement;
  private humanTrayEl!: HTMLElement;
  private engineTrayEl!: HTMLElement;
  private previewEl!: HTMLElement;
  private passButton!: HTMLButtonElement;
  private resignButton!: HTMLButtonElement;
  private difficultySelect!: HTMLSelectElement;
  private colorSelect!: HTMLSelectElement;
  private engineBadgeEl!: HTMLElement;

  constructor(
    private readonly root: HTMLElement,
    private readonly assets: LoadedAssets,
  ) {
    this.maps = buildOrientationMaps(assets.pieces);
  }

  attach(controller: GameController): void {
    this.controller = controller;
    this.mount();
    this.render();
  }

  /** Full re-render of the dynamic layers. Called by the controller on every update. */
  render = (): void => {
    if (!this.controller) return;
    this.renderBoard();
    this.renderOverlay();
    this.renderTrays();
    this.renderPreview();
    this.renderStatus();
  };

  // -- Skeleton ---------------------------------------------------------------

  private mount(): void {
    this.root.innerHTML = '';
    const layout = el('div', 'layout');

    const header = el('header', 'header');
    header.append(el('h1', '', 'AlphaBlokus'));
    this.engineBadgeEl = el('span', 'engine-badge', this.controller.info.name);
    if (this.controller.info.isFullStrength) this.engineBadgeEl.classList.add('full-strength');
    header.append(this.engineBadgeEl);
    layout.append(header);

    const main = el('div', 'main');
    main.append(this.buildBoardSvg());

    const side = el('aside', 'side');
    side.append(this.buildControls());
    this.statusEl = el('div', 'status');
    side.append(this.statusEl);
    this.scoresEl = el('div', 'scores');
    side.append(this.scoresEl);
    this.previewEl = el('div', 'preview');
    side.append(this.previewEl);
    side.append(el('h3', '', 'Your pieces'));
    this.humanTrayEl = el('div', 'tray');
    side.append(this.humanTrayEl);
    side.append(el('h3', '', 'Engine pieces'));
    this.engineTrayEl = el('div', 'tray tray-small');
    side.append(this.engineTrayEl);
    main.append(side);
    layout.append(main);

    const footer = el('footer', 'footer');
    const link = document.createElement('a');
    link.href = FULL_STRENGTH_GUIDE_URL;
    link.target = '_blank';
    link.rel = 'noreferrer';
    link.textContent = 'How to play the full-strength engine locally';
    footer.append(link);
    layout.append(footer);

    this.root.append(layout);
    this.bindKeys();
  }

  private buildControls(): HTMLElement {
    const controls = el('div', 'controls');

    this.difficultySelect = document.createElement('select');
    for (const level of this.controller.info.difficulties) {
      const option = document.createElement('option');
      option.value = level.id;
      option.textContent = level.label;
      option.title = level.description;
      if (level.id === this.controller.difficulty.id) option.selected = true;
      this.difficultySelect.append(option);
    }

    this.colorSelect = document.createElement('select');
    for (const [value, label] of [
      ['1', 'Play White (first)'],
      ['-1', 'Play Black (second)'],
    ] as const) {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = label;
      this.colorSelect.append(option);
    }

    const startNewGame = () => {
      const level = this.findDifficulty(this.difficultySelect.value);
      const humanPlayer = Number(this.colorSelect.value) as Player;
      this.selection = null;
      void this.controller.newGame(humanPlayer, level);
    };
    // Switching side requires a fresh game, so apply it immediately — picking
    // "Play Black" now actually makes you Black and lets the engine open,
    // rather than silently waiting for a New game click.
    this.colorSelect.addEventListener('change', startNewGame);
    // Difficulty applies to the engine's subsequent moves without discarding
    // the game in progress (New game reads it too).
    this.difficultySelect.addEventListener('change', () => {
      this.controller.difficulty = this.findDifficulty(this.difficultySelect.value);
    });

    const newGame = button('New game', 'primary', startNewGame);

    this.passButton = button('Pass', '', () => {
      void this.controller.humanMove(this.assets.tables.passIndex);
    });
    this.resignButton = button('Resign', 'danger', () => {
      if (confirm('Resign this game?')) this.controller.resign();
    });

    controls.append(
      this.difficultySelect,
      this.colorSelect,
      newGame,
      this.passButton,
      this.resignButton,
    );
    return controls;
  }

  private findDifficulty(id: string): DifficultyLevel {
    return this.controller.info.difficulties.find((level) => level.id === id)!;
  }

  private buildBoardSvg(): SVGSVGElement {
    const size = BOARD_SIZE * CELL + 2 * MARGIN;
    const svg = document.createElementNS(SVG_NS, 'svg');
    svg.setAttribute('viewBox', `0 0 ${size} ${size}`);
    svg.classList.add('board');

    const grid = document.createElementNS(SVG_NS, 'g');
    for (let row = 0; row < BOARD_SIZE; row++) {
      for (let col = 0; col < BOARD_SIZE; col++) {
        const rect = document.createElementNS(SVG_NS, 'rect');
        rect.setAttribute('x', String(MARGIN + col * CELL));
        rect.setAttribute('y', String(MARGIN + row * CELL));
        rect.setAttribute('width', String(CELL));
        rect.setAttribute('height', String(CELL));
        rect.classList.add('cell-bg');
        rect.dataset['cell'] = String(row * BOARD_SIZE + col);
        grid.append(rect);
      }
    }
    svg.append(grid);

    // Blokus notation labels: columns a..n, rows 14 (top) .. 1 (bottom).
    for (let col = 0; col < BOARD_SIZE; col++) {
      svg.append(
        svgText(MARGIN + col * CELL + CELL / 2, size - 7, String.fromCharCode(97 + col)),
        svgText(MARGIN + col * CELL + CELL / 2, 16, String.fromCharCode(97 + col)),
      );
    }
    for (let row = 0; row < BOARD_SIZE; row++) {
      svg.append(
        svgText(12, MARGIN + row * CELL + CELL / 2 + 4, String(BOARD_SIZE - row)),
        svgText(size - 12, MARGIN + row * CELL + CELL / 2 + 4, String(BOARD_SIZE - row)),
      );
    }

    // Start squares (array coords (4,4) White / (9,9) Black).
    for (const [startRow, startCol] of [
      [4, 4],
      [9, 9],
    ]) {
      const dot = document.createElementNS(SVG_NS, 'circle');
      dot.setAttribute('cx', String(MARGIN + startCol! * CELL + CELL / 2));
      dot.setAttribute('cy', String(MARGIN + startRow! * CELL + CELL / 2));
      dot.setAttribute('r', '5');
      dot.classList.add('start-dot');
      svg.append(dot);
    }

    this.boardCellsLayer = document.createElementNS(SVG_NS, 'g');
    svg.append(this.boardCellsLayer);
    this.overlayLayer = document.createElementNS(SVG_NS, 'g');
    this.overlayLayer.style.pointerEvents = 'none';
    svg.append(this.overlayLayer);

    svg.addEventListener('mousemove', (event) => {
      const cell = this.cellFromEvent(svg, event);
      if (cell !== this.hoverCell) {
        this.hoverCell = cell;
        this.renderOverlay();
      }
    });
    svg.addEventListener('mouseleave', () => {
      this.hoverCell = null;
      this.renderOverlay();
    });
    svg.addEventListener('click', (event) => {
      const cell = this.cellFromEvent(svg, event);
      if (cell !== null) this.onBoardClick(cell);
    });
    svg.addEventListener('wheel', (event) => {
      if (!this.selection) return;
      event.preventDefault();
      this.rotateSelection();
    });
    return svg;
  }

  private cellFromEvent(svg: SVGSVGElement, event: MouseEvent): number | null {
    const rect = svg.getBoundingClientRect();
    const scale = (BOARD_SIZE * CELL + 2 * MARGIN) / rect.width;
    const x = (event.clientX - rect.left) * scale - MARGIN;
    const y = (event.clientY - rect.top) * scale - MARGIN;
    const col = Math.floor(x / CELL);
    const row = Math.floor(y / CELL);
    if (row < 0 || row >= BOARD_SIZE || col < 0 || col >= BOARD_SIZE) return null;
    return row * BOARD_SIZE + col;
  }

  private bindKeys(): void {
    document.addEventListener('keydown', (event) => {
      if (!this.selection) return;
      if (event.key === 'r' || event.key === 'R') this.rotateSelection();
      else if (event.key === 'f' || event.key === 'F') this.flipSelection();
      else if (event.key === 'Escape') {
        this.selection = null;
        this.render();
      }
    });
  }

  // -- Interactions -------------------------------------------------------------

  private rotateSelection(): void {
    if (!this.selection) return;
    this.selection.orientationId = this.maps.rotate[this.selection.orientationId]!;
    this.render();
  }

  private flipSelection(): void {
    if (!this.selection) return;
    this.selection.orientationId = this.maps.flip[this.selection.orientationId]!;
    this.render();
  }

  private onBoardClick(cell: number): void {
    if (!this.selection || this.controller.phase !== 'humanTurn') return;
    const row = Math.floor(cell / BOARD_SIZE);
    const col = cell % BOARD_SIZE;
    const action = encodeAction(this.selection.orientationId, row, col);
    if (this.controller.legal.includes(action)) {
      this.selection = null;
      this.hoverCell = null;
      void this.controller.humanMove(action);
    }
  }

  // -- Rendering ------------------------------------------------------------------

  private renderBoard(): void {
    const state = this.controller.state;
    this.boardCellsLayer.innerHTML = '';
    for (let cell = 0; cell < state.ppb.length; cell++) {
      const value = state.ppb[cell]!;
      if (value === 0) continue;
      const row = Math.floor(cell / BOARD_SIZE);
      const col = cell % BOARD_SIZE;
      const rect = document.createElementNS(SVG_NS, 'rect');
      rect.setAttribute('x', String(MARGIN + col * CELL + 1.5));
      rect.setAttribute('y', String(MARGIN + row * CELL + 1.5));
      rect.setAttribute('width', String(CELL - 3));
      rect.setAttribute('height', String(CELL - 3));
      rect.setAttribute('rx', '4');
      rect.classList.add(value > 0 ? 'piece-white' : 'piece-black');
      this.boardCellsLayer.append(rect);
    }

    // Outline the engine's last placement so it's easy to spot.
    const lastAction = this.lastEnginePlacement();
    if (lastAction !== null) {
      for (const cell of this.actionCells(lastAction)) {
        const row = Math.floor(cell / BOARD_SIZE);
        const col = cell % BOARD_SIZE;
        const outline = document.createElementNS(SVG_NS, 'rect');
        outline.setAttribute('x', String(MARGIN + col * CELL + 1.5));
        outline.setAttribute('y', String(MARGIN + row * CELL + 1.5));
        outline.setAttribute('width', String(CELL - 3));
        outline.setAttribute('height', String(CELL - 3));
        outline.setAttribute('rx', '4');
        outline.classList.add('last-move');
        this.boardCellsLayer.append(outline);
      }
    }
  }

  private lastEnginePlacement(): number | null {
    const history = this.controller.history;
    if (history.length === 0) return null;
    // Find the most recent non-pass action made by the engine's colour.
    const engineParity = this.controller.humanPlayer === 1 ? 1 : 0; // ply index parity of engine moves
    for (let i = history.length - 1; i >= 0; i--) {
      if (i % 2 === engineParity && history[i] !== this.assets.tables.passIndex) return history[i]!;
    }
    return null;
  }

  private actionCells(action: number): number[] {
    const move = this.assets.tables.actionToMove[action]!;
    if (move < 0) return [];
    const cells: number[] = [];
    const base = move * this.assets.tables.cellsPerMove;
    for (let k = 0; k < this.assets.tables.cellsPerMove; k++) {
      const cell = this.assets.tables.cells[base + k]!;
      if (cell === this.assets.tables.nullCell) break;
      cells.push(cell);
    }
    return cells;
  }

  private renderOverlay(): void {
    this.overlayLayer.innerHTML = '';
    if (!this.selection || this.controller.phase !== 'humanTurn') return;

    // Ghost preview of the selected piece, following the cursor: green when the
    // placement is legal, red otherwise. (Legal placements are shown by the
    // ghost snapping legal — no separate anchor-dot markers, which confused by
    // marking bounding-box corners rather than where the piece connects.)
    const orientationId = this.selection.orientationId;
    if (this.hoverCell === null) return;
    const row = Math.floor(this.hoverCell / BOARD_SIZE);
    const col = this.hoverCell % BOARD_SIZE;
    const action = encodeAction(orientationId, row, col);
    const isLegal = this.controller.legal.includes(action);
    const grid = this.maps.grids[orientationId]!;
    for (let i = 0; i < grid.length; i++) {
      for (let j = 0; j < grid[i]!.length; j++) {
        if (grid[i]![j] === 0) continue;
        const cellRow = row + i;
        const cellCol = col + j;
        if (cellRow >= BOARD_SIZE || cellCol >= BOARD_SIZE) continue;
        const rect = document.createElementNS(SVG_NS, 'rect');
        rect.setAttribute('x', String(MARGIN + cellCol * CELL + 3));
        rect.setAttribute('y', String(MARGIN + cellRow * CELL + 3));
        rect.setAttribute('width', String(CELL - 6));
        rect.setAttribute('height', String(CELL - 6));
        rect.setAttribute('rx', '4');
        rect.classList.add(isLegal ? 'ghost-legal' : 'ghost-illegal');
        this.overlayLayer.append(rect);
      }
    }
  }

  private renderTrays(): void {
    const humanSlot = this.controller.humanPlayer === 1 ? 0 : 1;
    this.renderTray(this.humanTrayEl, humanSlot, true);
    this.renderTray(this.engineTrayEl, humanSlot === 0 ? 1 : 0, false);
  }

  private renderTray(container: HTMLElement, slot: number, interactive: boolean): void {
    container.innerHTML = '';
    for (const piece of this.assets.pieces.pieces) {
      const available = this.controller.state.remaining[slot * 22 + piece.id] === 1;
      const orientationId = this.maps.firstOrientation.get(piece.id)!;
      const tile = el('div', 'tray-piece');
      if (!available) tile.classList.add('used');
      if (interactive && this.selection?.pieceId === piece.id) tile.classList.add('selected');
      tile.title = piece.name;
      tile.append(this.pieceSvg(this.maps.grids[orientationId]!, interactive ? 10 : 7, slot));
      if (interactive && available) {
        tile.addEventListener('click', () => {
          this.selection =
            this.selection?.pieceId === piece.id ? null : { pieceId: piece.id, orientationId };
          this.render();
        });
      }
      container.append(tile);
    }
  }

  private pieceSvg(grid: Grid, cellSize: number, slot: number): SVGSVGElement {
    const svg = document.createElementNS(SVG_NS, 'svg');
    svg.setAttribute('viewBox', `0 0 ${5 * cellSize} ${5 * cellSize}`);
    svg.setAttribute('width', String(5 * cellSize));
    svg.setAttribute('height', String(5 * cellSize));
    for (let i = 0; i < grid.length; i++) {
      for (let j = 0; j < grid[i]!.length; j++) {
        if (grid[i]![j] === 0) continue;
        const rect = document.createElementNS(SVG_NS, 'rect');
        rect.setAttribute('x', String(j * cellSize + 0.5));
        rect.setAttribute('y', String(i * cellSize + 0.5));
        rect.setAttribute('width', String(cellSize - 1));
        rect.setAttribute('height', String(cellSize - 1));
        rect.setAttribute('rx', '1.5');
        rect.classList.add(slot === 0 ? 'piece-white' : 'piece-black');
        svg.append(rect);
      }
    }
    return svg;
  }

  private renderPreview(): void {
    this.previewEl.innerHTML = '';
    if (!this.selection) {
      this.previewEl.append(el('p', 'hint', 'Select a piece, then click an anchor on the board.'));
      return;
    }
    const humanSlot = this.controller.humanPlayer === 1 ? 0 : 1;
    this.previewEl.append(
      this.pieceSvg(this.maps.grids[this.selection.orientationId]!, 18, humanSlot),
    );
    const actions = el('div', 'preview-actions');
    actions.append(
      button('Rotate (R)', '', () => this.rotateSelection()),
      button('Flip (F)', '', () => this.flipSelection()),
    );
    this.previewEl.append(actions);
  }

  private renderStatus(): void {
    const controller = this.controller;
    this.passButton.disabled =
      controller.phase !== 'humanTurn' || !controller.legal.includes(this.assets.tables.passIndex);
    this.resignButton.disabled = controller.status.isOver;

    const remaining = (slot: number): number => {
      let squares = 0;
      for (let pieceId = 1; pieceId <= 21; pieceId++) {
        if (controller.state.remaining[slot * 22 + pieceId] === 1) {
          squares += this.assets.tables.pieceSizes[pieceId]!;
        }
      }
      return squares;
    };
    const humanSlot = controller.humanPlayer === 1 ? 0 : 1;
    this.scoresEl.innerHTML = '';
    this.scoresEl.append(
      el(
        'div',
        '',
        `You: ${89 - remaining(humanSlot)} squares placed, ${remaining(humanSlot)} left`,
      ),
      el(
        'div',
        '',
        `Engine: ${89 - remaining(1 - humanSlot)} squares placed, ${remaining(1 - humanSlot)} left`,
      ),
    );

    let text: string;
    if (controller.status.isOver) {
      if (controller.resigned) {
        text = 'You resigned — engine wins.';
      } else {
        const scores = controller.status.scores!;
        const humanScore = scores[humanSlot]!;
        const engineScore = scores[1 - humanSlot]!;
        text =
          humanScore === engineScore
            ? `Draw (${humanScore} — ${engineScore}).`
            : humanScore > engineScore
              ? `You win ${humanScore} — ${engineScore}!`
              : `Engine wins ${engineScore} — ${humanScore}.`;
      }
    } else if (controller.phase === 'engineTurn') {
      const progress = controller.thinking;
      text = progress
        ? `Engine thinking… ${progress.done}/${progress.total} simulations`
        : 'Engine thinking…';
    } else if (
      controller.legal.length === 1 &&
      controller.legal[0] === this.assets.tables.passIndex
    ) {
      text = 'No legal placements — you must pass.';
    } else {
      text = 'Your turn.';
    }
    const last = controller.lastEngineMove;
    if (last && !controller.status.isOver) {
      const percent = Math.round(((last.value + 1) / 2) * 100);
      text += ` (engine eval: ${percent}% for engine, ${Math.round(last.elapsedMs)} ms)`;
    }
    this.statusEl.textContent = text;
  }
}

// -- tiny DOM helpers ------------------------------------------------------------

function el(tag: string, className = '', text = ''): HTMLElement {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

function button(label: string, className: string, onClick: () => void): HTMLButtonElement {
  const node = document.createElement('button');
  node.textContent = label;
  if (className) node.className = className;
  node.addEventListener('click', onClick);
  return node;
}

/** Expose a scripting hook for the headless e2e game (plan step W12). */
export function exposeTestHook(controller: GameController, state: () => GameState): void {
  (window as unknown as Record<string, unknown>)['__alphablokus'] = { controller, state };
}

function svgText(x: number, y: number, text: string): SVGTextElement {
  const node = document.createElementNS(SVG_NS, 'text');
  node.setAttribute('x', String(x));
  node.setAttribute('y', String(y));
  node.setAttribute('text-anchor', 'middle');
  node.classList.add('board-label');
  node.textContent = text;
  return node;
}
