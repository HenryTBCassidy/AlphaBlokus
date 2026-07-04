"""Per-generation self-play history persistence (flat parquet files).

Board and policy arrays are serialised as raw bytes with shape/dtype metadata
in the parquet schema. Boards are stored compact (``IBoard.to_compact``) and
policies are stored **sparse** as ``(indices, values)`` byte pairs — the same
``ProcessedExample`` form the live replay buffer holds — so neither save nor
load ever materialises a dense policy vector (the dense-on-disk format
OOM-killed 10k-game generations; see ``docs/plans/oom-hardening.md`` O1/O2).
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
    from collections.abc import Sequence
    from pathlib import Path

    from alphablokus.selfplay.episode import ProcessedExample


class SelfPlayStore:
    """Read and write per-generation self-play training data.

    Each generation is stored as a single flat parquet file.  Board and policy
    arrays are serialised as raw bytes, with shape and dtype metadata stored
    in the parquet file-level schema so they can be reconstructed on load.

    Rows hold ``ProcessedExample`` tuples exactly as the live buffer does:
    boards in their **compact** form (``IBoard.to_compact`` — e.g. the int8
    14×14 placement board for Blokus, the 3×3 grid for TicTacToe) and policies
    as sparse ``(indices, values)`` pairs (int32 action ids + float32
    probabilities, only the nonzero entries — see ``storage/sparse_policy.py``).

    The schema carries two format markers, ``board_kind`` (``BOARD_KIND``) and
    ``policy_kind`` (``POLICY_KIND``). Files written before either scheme lack
    the corresponding marker — dense ``(C, N, N)`` boards and/or dense
    full-action-space policies — and cannot be loaded into the sparse compact
    buffer: ``load`` refuses them rather than silently misreading the bytes.
    Such runs must be resumed from their checkpoints instead.

    Args:
        directory: Root directory for self-play history files.
    """

    # Schema marker for the compact-board storage format. Its absence means a
    # legacy dense-encoding file, which ``load`` refuses (see class docstring).
    BOARD_KIND: str = "compact_v1"

    # Schema marker for the sparse-policy storage format (``policy_indices`` +
    # ``policy_values`` columns). Its absence means a legacy file holding one
    # dense ``policy`` blob per row, which ``load`` refuses (see class docstring).
    POLICY_KIND: str = "sparse_v1"

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def save(
        self,
        examples: Sequence[ProcessedExample],
        generation: int,
        policy_size: int,
        game_sizes: list[int] | None = None,
    ) -> None:
        """Save one generation's self-play examples to a parquet file.

        Args:
            examples: The generation's training examples (flat, in game order),
                with sparse ``(indices, values)`` policies — persisted as-is,
                never densified.
            generation: Generation number (used in the filename).
            policy_size: Length of the dense action space the sparse policies
                index into. Stored in the schema metadata so consumers can
                densify on demand without asking the game.
            game_sizes: Per-game position counts, in the same order the examples
                are laid out. When provided, ``load_games`` uses them to split the
                flat positions back into per-game lists so the rolling
                games-sized replay buffer can be reconstructed on resume. When
                ``None``, game boundaries are not recorded.
        """
        if not examples:
            return

        self._directory.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(
            {
                "board": [board.tobytes() for board, _pi, _value in examples],
                "policy_indices": [
                    np.ascontiguousarray(indices, dtype=np.int32).tobytes() for _b, (indices, _v), _value in examples
                ],
                "policy_values": [
                    np.ascontiguousarray(values, dtype=np.float32).tobytes() for _b, (_i, values), _value in examples
                ],
                "value": [float(value) for _b, _pi, value in examples],
            }
        )

        sample_board = examples[0][0]
        metadata = {
            "board_kind": self.BOARD_KIND,
            "board_shape": ",".join(str(d) for d in sample_board.shape),
            "board_dtype": str(sample_board.dtype),
            "policy_kind": self.POLICY_KIND,
            "policy_size": str(policy_size),
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
            A deque of ``ProcessedExample`` tuples (sparse policies, exactly
            the live-buffer form), or ``None`` if the file is missing.
        """
        filepath = self._directory / self._filename(generation)
        if not filepath.exists():
            return None

        parquet_file = pq.ParquetFile(filepath)
        metadata = {k.decode(): v.decode() for k, v in (parquet_file.schema_arrow.metadata or {}).items()}
        self._refuse_legacy_formats(filepath.name, metadata)

        board_shape = tuple(int(d) for d in metadata["board_shape"].split(","))
        board_dtype = np.dtype(metadata["board_dtype"])

        # Stream row-group batches straight off the Arrow reader — no
        # ``to_pandas``/``iterrows`` (which held Arrow + pandas + deque copies of
        # the whole file at once and walked 800k rows pathologically slowly).
        # Only one batch of raw bytes is resident beyond the growing deque.
        examples: deque[ProcessedExample] = deque()
        for batch in parquet_file.iter_batches():
            for board_bytes, indices_bytes, values_bytes, value in zip(
                batch.column("board").to_pylist(),
                batch.column("policy_indices").to_pylist(),
                batch.column("policy_values").to_pylist(),
                batch.column("value").to_pylist(),
                strict=True,
            ):
                board = np.frombuffer(board_bytes, dtype=board_dtype).reshape(board_shape).copy()
                indices = np.frombuffer(indices_bytes, dtype=np.int32).copy()
                values = np.frombuffer(values_bytes, dtype=np.float32).copy()
                examples.append((board, (indices, values), float(value)))

        logger.info(f"Loaded {len(examples)} examples from {filepath.name}")
        return examples

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
                "No game_sizes metadata in self-play file {} — treating its {} positions as one game.",
                generation,
                len(flat),
            )
            return [list(flat)]

        games: list[list[ProcessedExample]] = []
        iterator = iter(flat)
        for size in game_sizes:
            games.append([next(iterator) for _ in range(size)])
        return games

    def load_recent_games(
        self,
        last_file_index: int,
        num_games: int,
    ) -> deque[list[ProcessedExample]]:
        """Reconstruct the rolling replay buffer from recent generation files.

        Loads per-generation files newest-first starting at ``last_file_index``,
        accumulating games until at least ``num_games`` are gathered (or files
        run out), then returns the newest ``num_games`` games as a
        ``deque(maxlen=num_games)`` — exactly what a fresh run would hold at that
        point (sparse policies included, so a resumed buffer's RAM equals the
        live buffer's). This is the resume path for the games-sized buffer.

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
            len(buffer),
            sum(len(g) for g in buffer),
            last_file_index,
        )
        return buffer

    def _refuse_legacy_formats(self, filename: str, metadata: dict[str, str]) -> None:
        """Refuse legacy on-disk formats explicitly rather than misreading bytes.

        A file without ``board_kind`` holds dense ``(C, N, N)`` board encodings;
        one without ``policy_kind`` holds dense full-action-space policy blobs.
        Either way the bytes cannot be reinterpreted as the current sparse
        compact format, so we fail loudly with the reason.
        """
        board_kind = metadata.get("board_kind")
        if board_kind != self.BOARD_KIND:
            raise ValueError(
                f"{filename} has board_kind={board_kind!r}, expected "
                f"{self.BOARD_KIND!r}. Legacy dense self-play files cannot be "
                "loaded into the compact replay buffer — resume such runs from "
                "their checkpoints before the replay-buffer refactor.",
            )
        policy_kind = metadata.get("policy_kind")
        if policy_kind != self.POLICY_KIND:
            raise ValueError(
                f"{filename} has policy_kind={policy_kind!r}, expected "
                f"{self.POLICY_KIND!r}. Legacy dense-policy self-play files "
                "cannot be loaded into the sparse replay buffer — resume such "
                "runs from their checkpoints instead (see "
                "docs/plans/oom-hardening.md O1).",
            )

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
