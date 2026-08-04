"""Rolling games-sized replay buffer with parquet persistence.

Extracted from ``Coach`` so the buffer's mechanics (eviction, flattening,
save/load/resume round-trips through :class:`SelfPlayStore`) live in one
place. Examples hold sparse policies end-to-end — live buffer, save, and
resume all speak ``ProcessedExample`` (``docs/plans/archive/oom-hardening.md`` O1–O2);
this is also where the continuous-generations work (IDEAS I4 lineage) lands.
"""

from __future__ import annotations

import hashlib
from collections import deque
from typing import TYPE_CHECKING

from loguru import logger

from alphablokus.storage.selfplay_store import SelfPlayStore

if TYPE_CHECKING:
    from alphablokus.config import RunConfig
    from alphablokus.interfaces import IGame
    from alphablokus.selfplay.episode import GameExamples, ProcessedExample


def game_fingerprint(game: GameExamples) -> str:
    """A stable content hash identifying one self-play game.

    Needed because the eval set's source games must be kept **out of training**
    for as long as the eval set is in use, and a game's position in the buffer is
    not a usable identity: the deque shifts as games evict, and a resume rebuilds
    the buffer from parquet in a fresh order. Content is the only stable handle.

    Hashes the game's length plus its final position's compact board — a
    fully-played Blokus board is effectively unique (a survey of 770k episodes
    found zero exact duplicate games), and hashing one board per game instead of
    all of them keeps this cheap enough to run over a 60k-game buffer every
    generation.
    """
    digest = hashlib.sha256()
    digest.update(str(len(game)).encode())
    if game:
        digest.update(game[-1][0].tobytes())
    return digest.hexdigest()


class ReplayBuffer:
    """The last ``replay_buffer_games`` self-play games, plus their persistence.

    Holds one inner list of positions per game — game boundaries are preserved
    so eviction drops whole games (oldest auto-evict via ``deque(maxlen=...)``).
    The current generation's fresh games are tracked separately: they are what
    gets persisted each generation, while training flattens the whole buffer
    *minus the eval set's source games* (see :meth:`exclude_games`).
    """

    def __init__(self, config: RunConfig, game: IGame) -> None:
        self._config = config
        self._game = game
        self._store = SelfPlayStore(config.self_play_history_directory)
        self.games: deque[GameExamples] = deque(maxlen=config.replay_buffer_games)
        self._fresh_games: list[GameExamples] = []
        # Fingerprints of the games the eval set was sampled from. Those games
        # are withheld from :meth:`flat_examples` so the "held-out" diagnostics
        # are genuinely held out.
        self._excluded_fingerprints: set[str] = set()

    def __len__(self) -> int:
        return len(self.games)

    def __getitem__(self, index: int) -> GameExamples:
        return self.games[index]

    @property
    def capacity_games(self) -> int:
        """Maximum number of games the buffer holds before evicting."""
        return self._config.replay_buffer_games

    @property
    def fresh_game_count(self) -> int:
        """Games added since :meth:`begin_generation` — this generation's yield."""
        return len(self._fresh_games)

    @property
    def fresh_position_count(self) -> int:
        """Positions added since :meth:`begin_generation`."""
        return sum(len(game) for game in self._fresh_games)

    @property
    def excluded_fingerprints(self) -> frozenset[str]:
        """Fingerprints currently withheld from training."""
        return frozenset(self._excluded_fingerprints)

    def exclude_games(self, fingerprints: set[str] | frozenset[str]) -> None:
        """Withhold these games from :meth:`flat_examples`.

        Called with the eval set's source-game fingerprints. Until this existed the
        eval set was sampled *from* the training buffer and then trained on: its
        200 positions, their symmetry twins and their ~60 same-game siblings each
        stayed in training for ``replay_buffer_games / num_eps`` generations at
        ``epochs`` passes apiece. Every "held-out" per-epoch diagnostic was
        therefore in-sample early in a run and then silently changed meaning when
        those positions aged out — which is how a run could report eval top-1 ~0.99
        while its real strength fell.

        Replaces any previous exclusion set, so a rebuilt eval set releases the
        old games back into training.
        """
        self._excluded_fingerprints = set(fingerprints)

    def held_out_game_count(self) -> int:
        """How many games currently in the buffer are withheld from training."""
        if not self._excluded_fingerprints:
            return 0
        return sum(1 for game in self.games if game_fingerprint(game) in self._excluded_fingerprints)

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

    def flat_examples(self) -> list[ProcessedExample]:
        """Flatten the buffer to a list of training positions, in buffer order.

        Every position across all games currently in the buffer is used for
        training (``epochs`` full passes); the per-game structure governs eviction
        and the eval-set holdout, not batching.

        **Excludes the eval set's source games** (:meth:`exclude_games`), whole
        games at a time. Dropping only the sampled positions would not be enough:
        their symmetry twins and same-game siblings carry the same outcome label
        and would still leak the answer.

        **Deliberately not shuffled.** This used to call ``random.shuffle``, which
        was both redundant and actively harmful: the training ``DataLoader``
        already reshuffles every epoch through an explicitly seeded generator
        (``base_wrapper._shuffle_seed``), so the extra pass bought nothing — while
        ``random.shuffle`` draws from Python's *global* ``random`` module, which
        the Coach never seeds (it seeds ``numpy`` and ``torch`` only). The result
        was that a run at a fixed seed was not reproducible: the eval set was
        sampled with a seeded numpy generator from an unseeded list, so the
        indices reproduced and the positions they pointed at did not. Two A/B arms
        at the same seed therefore differed before the treatment did anything.

        Keeping the order canonical makes the whole path reproducible, and drops a
        full copy-and-shuffle of a 60k-game buffer per generation.
        """
        if not self._excluded_fingerprints:
            return [example for game in self.games for example in game]

        kept: list[ProcessedExample] = []
        withheld = 0
        for game in self.games:
            if game_fingerprint(game) in self._excluded_fingerprints:
                withheld += 1
                continue
            kept.extend(game)
        if withheld:
            logger.debug(
                "Withheld {} eval-set source game(s) from training ({} positions kept)",
                withheld,
                len(kept),
            )
        return kept

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
