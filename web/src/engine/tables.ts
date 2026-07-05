/**
 * Loading + parsing of the exported rules assets (scripts/export_web_assets.py).
 *
 * `rules.bin` is a little-endian concatenation of the static geometry tables
 * the Python JAX kernels are built from: per-move footprint / edge-halo /
 * corner-halo cell lists, piece sizes, and start cells. The manifest records
 * each array's dtype/shape/offset, so parsing is table-driven.
 */

export interface ManifestArray {
  name: string;
  dtype: string;
  shape: number[];
  offset: number;
  num_bytes: number;
}

export interface NetVariantFile {
  path: string;
  sha256: string;
  bytes: number;
}

export interface Manifest {
  encodingVersion: string;
  boardSize: number;
  numCells: number;
  actionSize: number;
  passIndex: number;
  numOrientations: number;
  numChannels: number;
  nullCell: number;
  rules: { path: string; sha256: string; bytes: number; arrays: ManifestArray[] };
  pieces: { path: string; sha256: string };
  net: {
    checkpoint: string;
    numFilters: number;
    numResidualBlocks: number;
    policyHead: string;
    numParameters: number;
    files: Record<string, NetVariantFile>;
  } | null;
}

export interface PieceOrientation {
  orientationId: number;
  pieceId: number;
  orientation: string;
  grid: number[][];
}

export interface PiecesData {
  pieces: { id: number; name: string; size: number }[];
  orientations: PieceOrientation[];
}

/** Parsed rules tables; every per-move array is indexed by move id 0..numMoves-1. */
export interface RulesTables {
  boardSize: number;
  numCells: number;
  actionSize: number;
  passIndex: number;
  nullCell: number;
  numMoves: number;
  /** Piece id (1..21) per move. */
  piece: Uint8Array;
  /** Flat action id per move. */
  actionId: Uint32Array;
  /** Footprint cells, MAX 5 per move, NULL_CELL padded (packed left). */
  cells: Uint8Array;
  /** Edge-adjacent halo cells, MAX 16 per move, NULL_CELL padded. */
  adjCells: Uint8Array;
  /** Diagonal (corner) attach cells, MAX 16 per move, NULL_CELL padded. */
  attachCells: Uint8Array;
  /** Squares per piece id (index 0 unused). */
  pieceSizes: Uint8Array;
  /** Flat array-index start cell per player slot (0 = White, 1 = Black). */
  startCells: Int32Array;
  /** action id -> move id, -1 for non-placement actions (incl. pass). */
  actionToMove: Int32Array;
  cellsPerMove: number;
  adjPerMove: number;
  attachPerMove: number;
}

function sliceArray(buffer: ArrayBuffer, entry: ManifestArray): ArrayBuffer {
  // slice() copies into a fresh zero-offset buffer, so multi-byte views are
  // always aligned regardless of the entry's offset in the blob.
  return buffer.slice(entry.offset, entry.offset + entry.num_bytes);
}

/** Parse the rules blob against its manifest description. */
export function parseRulesTables(manifest: Manifest, blob: ArrayBuffer): RulesTables {
  const byName = new Map(manifest.rules.arrays.map((entry) => [entry.name, entry]));
  const need = (name: string): ManifestArray => {
    const entry = byName.get(name);
    if (!entry) throw new Error(`rules.bin manifest missing array '${name}'`);
    return entry;
  };

  const pieceEntry = need('piece');
  const cellsEntry = need('cells');
  const adjEntry = need('adj_cells');
  const attachEntry = need('attach_cells');

  const numMoves = pieceEntry.shape[0]!;
  const piece = new Uint8Array(sliceArray(blob, pieceEntry));
  const actionId = new Uint32Array(sliceArray(blob, need('action_id')));
  const cells = new Uint8Array(sliceArray(blob, cellsEntry));
  const adjCells = new Uint8Array(sliceArray(blob, adjEntry));
  const attachCells = new Uint8Array(sliceArray(blob, attachEntry));
  const pieceSizes = new Uint8Array(sliceArray(blob, need('piece_sizes')));
  const startCells = new Int32Array(sliceArray(blob, need('start_cells')));

  const actionToMove = new Int32Array(manifest.actionSize).fill(-1);
  for (let move = 0; move < numMoves; move++) {
    actionToMove[actionId[move]!] = move;
  }

  return {
    boardSize: manifest.boardSize,
    numCells: manifest.numCells,
    actionSize: manifest.actionSize,
    passIndex: manifest.passIndex,
    nullCell: manifest.nullCell,
    numMoves,
    piece,
    actionId,
    cells,
    adjCells,
    attachCells,
    pieceSizes,
    startCells,
    actionToMove,
    cellsPerMove: cellsEntry.shape[1]!,
    adjPerMove: adjEntry.shape[1]!,
    attachPerMove: attachEntry.shape[1]!,
  };
}

export interface LoadedAssets {
  manifest: Manifest;
  tables: RulesTables;
  pieces: PiecesData;
}

/** Fetch + parse manifest, rules blob and pieces JSON from a static asset base URL. */
export async function loadAssets(baseUrl: string): Promise<LoadedAssets> {
  const manifest = (await fetchJson(`${baseUrl}/manifest.json`)) as Manifest;
  const [blob, pieces] = await Promise.all([
    fetchBinary(`${baseUrl}/${manifest.rules.path}`),
    fetchJson(`${baseUrl}/${manifest.pieces.path}`) as Promise<PiecesData>,
  ]);
  return { manifest, tables: parseRulesTables(manifest, blob), pieces };
}

async function fetchJson(url: string): Promise<unknown> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to fetch ${url}: ${response.status}`);
  return response.json();
}

async function fetchBinary(url: string): Promise<ArrayBuffer> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to fetch ${url}: ${response.status}`);
  return response.arrayBuffer();
}
