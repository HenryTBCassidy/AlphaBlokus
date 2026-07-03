"""The composition root: the one module that names concrete game code.

Everything else in the framework depends only on the protocols in
``interfaces.py``; this module owns every dispatch from ``config.game``
strings to concrete classes — game construction, network construction, and
the jax self-play backend. New games register here and nowhere else. Do not
import ``alphablokus.games.*`` from framework code outside this module.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from alphablokus.games.blokusduo.game import BlokusDuoGame
from alphablokus.games.blokusduo.neuralnets.wrapper import NNetWrapper as BlokusDuoNNetWrapper
from alphablokus.games.blokusduo.pieces import default_pieces_path
from alphablokus.games.tictactoe.game import TicTacToeGame
from alphablokus.games.tictactoe.neuralnets.wrapper import NNetWrapper as TicTacToeNNetWrapper

if TYPE_CHECKING:
    from alphablokus.core.config import RunConfig
    from alphablokus.core.interfaces import IGame, INeuralNetWrapper, IOracle
    from alphablokus.selfplay.generate import SelfPlayBackendFn


def instantiate_game(config: RunConfig) -> IGame:
    """Instantiate just the game (no network) from the run config.

    Used by inference-server workers, which need the game rules but **not**
    their own network — the server process owns the single GPU net, so building
    a net per worker would recreate the multi-net GPU contention the server
    is designed to remove.
    """
    match config.game:
        case "tictactoe":
            return TicTacToeGame()
        case "blokusduo":
            return BlokusDuoGame(pieces_config_path=default_pieces_path())
        case unknown:
            raise ValueError(
                f"Unknown game: {unknown!r}. Expected 'tictactoe' or 'blokusduo'.",
            )


def instantiate_game_and_network(config: RunConfig) -> tuple[IGame, INeuralNetWrapper]:
    """Instantiate the game and neural net wrapper from the run config.

    Dispatches on ``config.game`` and constructs the per-game wrapper with
    a fresh random-init network. Callers that want trained weights should
    call ``nnet.load_checkpoint(...)`` afterwards.
    """
    game = instantiate_game(config)
    match config.game:
        case "tictactoe":
            nnet: INeuralNetWrapper = TicTacToeNNetWrapper(game, config)
        case "blokusduo":
            nnet = BlokusDuoNNetWrapper(game, config)
        case unknown:  # pragma: no cover - already validated in instantiate_game
            raise ValueError(
                f"Unknown game: {unknown!r}. Expected 'tictactoe' or 'blokusduo'.",
            )
    return game, nnet


def resolve_jax_selfplay_backend(config: RunConfig) -> SelfPlayBackendFn:
    """Resolve the GPU-native self-play backend for the configured game.

    The jax backend is game-specific (rules as array kernels), so unlike the
    protocol-typed factories above it only exists where someone has built it —
    currently Blokus Duo only. The import is deferred so python-backend runs
    never require the ``jax`` extra.
    """
    match config.game:
        case "blokusduo":
            from alphablokus.games.blokusduo.jax.backend import generate_self_play_games
            return generate_self_play_games
        case unknown:
            raise ValueError(
                f"selfplay_backend 'jax' supports only 'blokusduo' (got {unknown!r}); "
                "use selfplay_backend 'python'.",
            )


def resolve_oracle(config: RunConfig, game: IGame) -> IOracle | None:
    """Resolve the perfect-play oracle for the configured game, if one exists.

    Only games small enough to solve exactly have one (currently TicTacToe's
    minimax). ``None`` means the framework skips oracle-based evaluation.
    """
    match config.game:
        case "tictactoe":
            from alphablokus.games.tictactoe.oracle import TicTacToeOracle

            assert isinstance(game, TicTacToeGame), "config.game/game instance mismatch"
            return TicTacToeOracle(game)
        case _:
            return None
