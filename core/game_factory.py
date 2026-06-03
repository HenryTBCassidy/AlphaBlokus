"""Single source for instantiating ``(game, nnet)`` from a ``RunConfig``.

Previously this dispatch lived in ``main.py`` and was copy-pasted into
``scripts/benchmark_phases.py``. ``core/parallel_self_play.py`` needs the
same factory inside each worker process, so the logic moves here and the
two existing call sites import it.

Keeping it in ``core/`` (alongside the protocols it returns) rather than
in a scripts/ utility means new entry points pick it up automatically.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from games.blokusduo.game import BlokusDuoGame
from games.blokusduo.neuralnets.wrapper import NNetWrapper as BlokusDuoNNetWrapper
from games.tictactoe.game import TicTacToeGame
from games.tictactoe.neuralnets.wrapper import NNetWrapper as TicTacToeNNetWrapper

if TYPE_CHECKING:
    from core.config import RunConfig
    from core.interfaces import IGame, INeuralNetWrapper

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BLOKUS_PIECES = _REPO_ROOT / "games" / "blokusduo" / "pieces.json"


def instantiate_game(config: RunConfig) -> IGame:
    """Instantiate just the game (no network) from the run config.

    Used by F5 inference-server workers, which need the game rules but **not**
    their own network — the server process owns the single GPU net, so building
    a net per worker would recreate the multi-net GPU contention F5 removes.
    """
    match config.game:
        case "tictactoe":
            return TicTacToeGame()
        case "blokusduo":
            return BlokusDuoGame(pieces_config_path=_BLOKUS_PIECES)
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
