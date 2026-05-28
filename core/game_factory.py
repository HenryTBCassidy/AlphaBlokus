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


def instantiate_game_and_network(config: RunConfig) -> tuple[IGame, INeuralNetWrapper]:
    """Instantiate the game and neural net wrapper from the run config.

    Dispatches on ``config.game`` and constructs the per-game wrapper with
    a fresh random-init network. Callers that want trained weights should
    call ``nnet.load_checkpoint(...)`` afterwards.
    """
    match config.game:
        case "tictactoe":
            game: IGame = TicTacToeGame()
            nnet: INeuralNetWrapper = TicTacToeNNetWrapper(game, config)
        case "blokusduo":
            game = BlokusDuoGame(pieces_config_path=_BLOKUS_PIECES)
            nnet = BlokusDuoNNetWrapper(game, config)
        case unknown:
            raise ValueError(
                f"Unknown game: {unknown!r}. Expected 'tictactoe' or 'blokusduo'.",
            )
    return game, nnet
