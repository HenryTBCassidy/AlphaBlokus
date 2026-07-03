"""Rolling games-sized replay buffer with parquet persistence.

Extracted from ``Coach`` so the buffer's mechanics (eviction, flattening,
save/load/resume round-trips through :class:`SelfPlayStore`) live in one
place. This is also where the continuous-generations work (IDEAS I4 lineage)
and the sparse-on-disk format (``docs/plans/oom-hardening.md`` O1–O3) land.
"""
from __future__ import annotations

from collections import deque
from random import shuffle
from typing import TYPE_CHECKING

from alphablokus.storage.selfplay_store import SelfPlayStore
from alphablokus.storage.sparse_policy import as_dense

if TYPE_CHECKING:
    from alphablokus.config import RunConfig
    from alphablokus.interfaces import IGame
    from alphablokus.selfplay.episode import GameExamples, ProcessedExample


class ReplayBuffer:
    """The last ``replay_buffer_games`` self-play games, plus their persistence.

    Holds one inner list of positions per game — game boundaries are preserved
    so eviction drops whole games (oldest auto-evict via ``deque(maxlen=...)``).
    The current generation's fresh games are tracked separately: they are what
    gets persisted each generation, while training always flattens the whole
    buffer.
    """

    def __init__(self, config: RunConfig, game: IGame) -> None:
        self._config = config
        self._game = game
        self._store = SelfPlayStore(config.self_play_history_directory)
        self.games: deque[GameExamples] = deque(maxlen=config.replay_buffer_games)
        self._fresh_games: list[GameExamples] = []

    def __len__(self) -> int:
        return len(self.games)

    def __getitem__(self, index: int) -> GameExamples:
        return self.games[index]

    @property
    def capacity_games(self) -> int:
        """Maximum number of games the buffer holds before evicting."""
        return self._config.replay_buffer_games

    def add_generation(self, fresh_games: list[GameExamples]) -> None:
        """Push one generation's games into the buffer (oldest auto-evict)."""
        self._fresh_games = fresh_games
        self.games.extend(fresh_games)

    def flat_shuffled_examples(self) -> list[ProcessedExample]:
        """Flatten the whole rolling buffer to a shuffled list of positions.

        Every position across all games currently in the buffer is used for
        training (``epochs`` full passes); the per-game structure only governs
        eviction, not training.
        """
        examples = [example for game in self.games for example in game]
        shuffle(examples)
        return examples

    def save_fresh(self, file_index: int) -> None:
        """Save this generation's fresh self-play games to a parquet file.

        Persists the fresh games only (not the whole buffer) with their
        per-game sizes so the games-sized buffer can be reconstructed on
        resume. Delegates to :meth:`SelfPlayStore.save`.
        """
        if not self._fresh_games:
            return
        game_sizes = [len(game) for game in self._fresh_games]
        flat = [example for game in self._fresh_games for example in game]
        if not flat:
            return
        # In-RAM examples hold sparse policies (indices, values); the on-disk
        # store keeps dense, so densify a transient copy here. By this point the
        # self-play worker pool is torn down, so the memory is free.
        action_size = self._game.get_action_size()
        dense = deque(
            (board, as_dense(pi, action_size), value)
            for board, pi, value in flat
        )
        self._store.save(dense, file_index, game_sizes=game_sizes)

    def load_recent(self, up_to_generation: int) -> None:
        """Refill the rolling buffer from parquet files on disk.

        Loads recent generation files (newest at file index
        ``up_to_generation``) until ``replay_buffer_games`` games are gathered.
        Used by the ``--load_model`` warm-start path. Delegates to
        :meth:`SelfPlayStore.load_recent_games`.
        """
        # Loaded games hold DENSE policies (the on-disk format) while live
        # self-play appends sparse ones — tracked by oom-hardening O2.
        self.games = self._store.load_recent_games(  # type: ignore[assignment]
            up_to_generation, self._config.replay_buffer_games,
        )

    def load_for_resume(self, last_completed_generation: int) -> None:
        """Refill the rolling buffer to resume training at ``last + 1``.

        Self-play parquet files are 0-indexed (file ``k`` holds generation
        ``k+1``'s data — see :meth:`save_fresh`), so generation ``G``'s data
        lives in file index ``G-1``. We reconstruct the games-sized buffer the
        next generation would hold by loading recent files newest-first until
        ``replay_buffer_games`` games are gathered.
        """
        self.load_recent(last_completed_generation - 1)
