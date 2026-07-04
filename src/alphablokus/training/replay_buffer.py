"""Rolling games-sized replay buffer with parquet persistence.

Extracted from ``Coach`` so the buffer's mechanics (eviction, flattening,
save/load/resume round-trips through :class:`SelfPlayStore`) live in one
place. Examples hold sparse policies end-to-end — live buffer, save, and
resume all speak ``ProcessedExample`` (``docs/plans/oom-hardening.md`` O1–O2);
this is also where the continuous-generations work (IDEAS I4 lineage) lands.
"""

from __future__ import annotations

from collections import deque
from random import shuffle
from typing import TYPE_CHECKING

from alphablokus.storage.selfplay_store import SelfPlayStore

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

    def begin_generation(self) -> None:
        """Reset fresh-game tracking before a generation's self-play starts.

        Games streamed in via :meth:`add_game` after this call are what
        :meth:`save_fresh` persists for the generation.
        """
        self._fresh_games = []

    def add_game(self, game: GameExamples) -> None:
        """Append one completed self-play game (oldest auto-evict via maxlen).

        The streaming entry point: self-play backends hand each game over as it
        finishes, so a whole generation is never accumulated outside the buffer.
        Only a reference is tracked for :meth:`save_fresh` — no copy.
        """
        self._fresh_games.append(game)
        self.games.append(game)

    def add_generation(self, fresh_games: list[GameExamples]) -> None:
        """Push one generation's games into the buffer at once.

        Convenience for already-materialised generations (tests, tools); the
        Coach streams per game via :meth:`begin_generation` + :meth:`add_game`.
        """
        self.begin_generation()
        for game in fresh_games:
            self.add_game(game)

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
        # Examples are persisted exactly as the live buffer holds them — sparse
        # policies included. No densify: the whole-generation dense copy this
        # step used to build (~25 GB at 10k games) is what OOM-killed the box.
        self._store.save(
            flat,
            file_index,
            policy_size=self._game.get_action_size(),
            game_sizes=game_sizes,
        )

    def load_recent(self, up_to_generation: int) -> None:
        """Refill the rolling buffer from parquet files on disk.

        Loads recent generation files (newest at file index
        ``up_to_generation``) until ``replay_buffer_games`` games are gathered.
        Used by the ``--load_model`` warm-start path. Delegates to
        :meth:`SelfPlayStore.load_recent_games`.
        """
        # Loaded games hold the same sparse policies live self-play appends —
        # a resumed buffer's RAM equals the live buffer's.
        self.games = self._store.load_recent_games(
            up_to_generation,
            self._config.replay_buffer_games,
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
