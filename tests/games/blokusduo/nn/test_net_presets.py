"""Net-size presets construct, forward, and scale sensibly on the Blokus net.

Net size is a first-class budget-vs-strength knob for cloud runs
(docs/plans/cloud-scale-training.md C5): every ``NET_PRESETS`` entry must
build a working ``AlphaBlokusDuo`` and parameter counts must grow with the
preset ladder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from alphablokus.config import NET_PRESETS, NetConfig
from alphablokus.games.base_wrapper import count_parameters
from alphablokus.games.blokusduo.nn.net import AlphaBlokusDuo

if TYPE_CHECKING:
    from alphablokus.games.blokusduo.game import BlokusDuoGame

PRESET_LADDER = ["small", "medium", "large", "xl"]


def _net_for(preset: str, blokus_game: BlokusDuoGame) -> AlphaBlokusDuo:
    sizes = NET_PRESETS[preset]
    config = NetConfig(
        learning_rate=1e-3,
        dropout=0.3,
        epochs=1,
        batch_size=4,
        cuda=False,
        num_filters=sizes["num_filters"],
        num_residual_blocks=sizes["num_residual_blocks"],
    )
    board = blokus_game.initialise_board()
    rows, cols = blokus_game.get_board_size()
    return AlphaBlokusDuo(
        board_rows=rows,
        board_cols=cols,
        action_size=blokus_game.get_action_size(),
        num_input_channels=board.num_channels,
        config=config,
    )


def test_preset_ladder_is_complete() -> None:
    assert sorted(NET_PRESETS) == sorted(PRESET_LADDER)
    assert NET_PRESETS["small"] == {"num_filters": 64, "num_residual_blocks": 4}


def test_every_preset_builds_and_forwards(blokus_game: BlokusDuoGame) -> None:
    board = blokus_game.initialise_board()
    batch = torch.from_numpy(board.as_multi_channel(1)).unsqueeze(0).repeat(2, 1, 1, 1).float()
    for preset in PRESET_LADDER:
        net = _net_for(preset, blokus_game)
        net.eval()
        with torch.no_grad():
            log_pi, value = net(batch)
        assert log_pi.shape == (2, blokus_game.get_action_size())
        assert value.shape == (2, 1)
        assert torch.isfinite(log_pi).all() and torch.isfinite(value).all()


def test_parameter_counts_grow_up_the_ladder(blokus_game: BlokusDuoGame) -> None:
    counts = [count_parameters(_net_for(preset, blokus_game)) for preset in PRESET_LADDER]
    assert counts == sorted(counts)
    assert counts[0] < counts[-1] // 5  # xl is a genuinely different size class, not a nudge
