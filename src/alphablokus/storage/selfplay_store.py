"""Per-generation self-play history persistence (flat parquet files).

Board and policy arrays are serialised as raw bytes with shape/dtype metadata
in the parquet schema. Boards are stored compact (``IBoard.to_compact``);
policies are currently stored dense on disk — the sparse-on-disk format is
tracked by ``docs/plans/oom-hardening.md`` (O1/O2).
"""
from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

if TYPE_CHECKING:
    from pathlib import Path

    from alphablokus.selfplay.episode import ProcessedExample



class SelfPlayStore:
    """Read and write per-generation self-play training data.

    Each generation is stored as a single flat parquet file.  Board and policy
    arrays are serialised as raw bytes, with shape and dtype metadata stored
    in the parquet file-level schema so they can be reconstructed on load.

    Boards are stored in their **compact** form (``IBoard.to_compact`` — e.g.
    the int8 14×14 placement board for Blokus, the 3×3 grid for TicTacToe), and
    the schema carries a ``board_kind`` marker (``BOARD_KIND``). Files written
    before this scheme have no marker and held the dense ``(C, N, N)`` encoding;
    they cannot be loaded into the compact buffer (the loader refuses them
    rather than silently misreading the bytes). Such runs must be resumed from
    their dense checkpoints before this refactor, not after.

    Args:
        directory: Root directory for self-play history files.
    """

    # Schema marker for the compact-board storage format. Its absence means a
    # legacy dense-encoding file, which ``load`` refuses (see class docstring).
    BOARD_KIND: str = "compact_v1"

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def save(
        self,
        examples: deque[ProcessedExample],
        generation: int,
        game_sizes: list[int] | None = None,
    ) -> None:
        """Save one generation's self-play examples to a parquet file.

        Args:
            examples: The generation's training examples (flat, in game order).
            generation: Generation number (used in the filename).
            game_sizes: Per-game position counts, in the same order the examples
                are laid out. When provided, ``load_games`` uses them to split the
                flat positions back into per-game lists so the rolling
                games-sized replay buffer can be reconstructed on resume. When
                ``None``, game boundaries are not recorded.
        """
        if not examples:
            return

        self._directory.mkdir(parents=True, exist_ok=True)

        boards, policies, values = zip(*examples, strict=False)

        df = pd.DataFrame({
            "board": [b.tobytes() for b in boards],
            "policy": [p.tobytes() for p in policies],
            "value": list(values),
        })

        sample_board = boards[0]
        sample_policy = policies[0]
        metadata = {
            "board_kind": self.BOARD_KIND,
            "board_shape": ",".join(str(d) for d in sample_board.shape),
            "board_dtype": str(sample_board.dtype),
            "policy_size": str(sample_policy.shape[0]),
            "policy_dtype": str(sample_policy.dtype),
        }
        if game_sizes is not None:
            metadata["game_sizes"] = ",".join(str(s) for s in game_sizes)

        table = pa.Table.from_pandas(df)
        merged_metadata = {
            **(table.schema.metadata or {}),
            **{k.encode(): v.encode() for k, v in metadata.items()},
        }
        table = table.replace_schema_metadata(merged_metadata)

        filepath = self._directory / self._filename(generation)
        pq.write_table(table, filepath)
        logger.info(f"Saved {len(df)} self-play examples to {filepath.name}")

    def load(self, generation: int) -> deque[ProcessedExample] | None:
        """Load a single generation's self-play examples from a parquet file.

        Returns ``None`` if the file does not exist (caller decides how to
        handle missing data).

        Args:
            generation: Generation number to load.

        Returns:
            A deque of ``ProcessedExample`` tuples, or ``None`` if the file
            is missing.
        """
        filepath = self._directory / self._filename(generation)
        if not filepath.exists():
            return None

        table = pq.read_table(filepath)
        metadata = {k.decode(): v.decode() for k, v in table.schema.metadata.items()}

        # Refuse legacy dense files explicitly rather than misreading their bytes
        # as a compact array (a (44,14,14) float32 blob is not a (14,14) int8 one).
        board_kind = metadata.get("board_kind")
        if board_kind != self.BOARD_KIND:
            raise ValueError(
                f"{filepath.name} has board_kind={board_kind!r}, expected "
                f"{self.BOARD_KIND!r}. Legacy dense self-play files cannot be "
                "loaded into the compact replay buffer — resume such runs from "
                "their checkpoints before the replay-buffer refactor.",
            )

        board_shape = tuple(int(d) for d in metadata["board_shape"].split(","))
        board_dtype = np.dtype(metadata["board_dtype"])
        policy_size = int(metadata["policy_size"])
        policy_dtype = np.dtype(metadata["policy_dtype"])

        df = table.to_pandas()
        examples: deque[ProcessedExample] = deque()
        for _, row in df.iterrows():
            board = np.frombuffer(row["board"], dtype=board_dtype).reshape(board_shape).copy()
            policy = np.frombuffer(row["policy"], dtype=policy_dtype).reshape(policy_size).copy()
            examples.append((board, policy, float(row["value"])))

        logger.info(f"Loaded {len(examples)} examples from {filepath.name}")
        return examples

    def load_window(
        self,
        up_to_generation: int,
        window_size: int,
    ) -> list[deque[ProcessedExample]]:
        """Load self-play examples for a sliding window of generations.

        Loads generations from ``max(0, up_to_generation - window_size)``
        through ``up_to_generation`` (inclusive), skipping any files that do
        not exist.

        Args:
            up_to_generation: The most recent generation to include.
            window_size: How many past generations to look back.

        Returns:
            A list of deques, one per loaded generation, in generation order.
            Empty list if the directory does not exist or no files are found.
        """
        if not self._directory.exists():
            logger.warning(f"Self-play history directory not found: {self._directory}")
            return []

        start_gen = max(0, up_to_generation - window_size)
        history: list[deque[ProcessedExample]] = []

        for gen in range(start_gen, up_to_generation + 1):
            loaded = self.load(gen)
            if loaded is not None:
                history.append(loaded)

        logger.info(
            f"Loaded {sum(len(e) for e in history)} total examples "
            f"from {len(history)} generations"
        )
        return history

    def load_games(self, generation: int) -> list[list[ProcessedExample]] | None:
        """Load a generation's examples split back into per-game lists.

        Uses the ``game_sizes`` schema metadata written by :meth:`save` to
        restore the game boundaries the flat parquet rows would otherwise lose.
        Returns ``None`` if the file is missing. A file saved without
        ``game_sizes`` is returned as a single game (logged), so the buffer can
        still be refilled, just at coarser eviction granularity.
        """
        flat = self.load(generation)
        if flat is None:
            return None

        game_sizes = self._read_game_sizes(generation)
        if game_sizes is None:
            logger.warning(
                "No game_sizes metadata in self-play file {} — treating its "
                "{} positions as one game.", generation, len(flat),
            )
            return [list(flat)]

        games: list[list[ProcessedExample]] = []
        iterator = iter(flat)
        for size in game_sizes:
            games.append([next(iterator) for _ in range(size)])
        return games

    def load_recent_games(
        self, last_file_index: int, num_games: int,
    ) -> deque[list[ProcessedExample]]:
        """Reconstruct the rolling replay buffer from recent generation files.

        Loads per-generation files newest-first starting at ``last_file_index``,
        accumulating games until at least ``num_games`` are gathered (or files
        run out), then returns the newest ``num_games`` games as a
        ``deque(maxlen=num_games)`` — exactly what a fresh run would hold at that
        point. This is the resume path for the games-sized buffer.

        Args:
            last_file_index: Newest generation file index to load from.
            num_games: Buffer capacity in games (``replay_buffer_games``).
        """
        buffer: deque[list[ProcessedExample]] = deque(maxlen=num_games)
        if not self._directory.exists():
            logger.warning(f"Self-play history directory not found: {self._directory}")
            return buffer

        # Collect newest-first, then replay oldest→newest into the maxlen deque
        # so it keeps the newest ``num_games`` games (older overshoot evicts).
        collected_newest_first: list[list[list[ProcessedExample]]] = []
        total_games = 0
        file_index = last_file_index
        while file_index >= 0 and total_games < num_games:
            games = self.load_games(file_index)
            if games is not None:
                collected_newest_first.append(games)
                total_games += len(games)
            file_index -= 1

        for gen_games in reversed(collected_newest_first):
            buffer.extend(gen_games)

        logger.info(
            "Reconstructed replay buffer: {} games ({} positions) from files ≤ {}",
            len(buffer), sum(len(g) for g in buffer), last_file_index,
        )
        return buffer

    def _read_game_sizes(self, generation: int) -> list[int] | None:
        """Read the per-game position counts from a file's schema metadata.

        Reads only the parquet footer (not the row data), so it's cheap.
        """
        filepath = self._directory / self._filename(generation)
        if not filepath.exists():
            return None
        metadata = pq.read_schema(filepath).metadata or {}
        raw = metadata.get(b"game_sizes")
        if raw is None:
            return None
        return [int(s) for s in raw.decode().split(",")]

    @staticmethod
    def _filename(generation: int) -> str:
        """Generate filename for a single generation's self-play data."""
        return f"self_play_{generation}.parquet"
