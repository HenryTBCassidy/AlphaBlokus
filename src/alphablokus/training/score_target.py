"""The score head's target transform — one definition, used by trainer and diagnostics.

The auxiliary score head (plan ``docs/plans/score-auxiliary-target.md`` S2) predicts the
final score margin *from the side to move*. Raw margins are unusable as a regression
target: measured over the real corpora they run roughly −88…+88 with a **median of 3**, so
a raw-margin MSE is dominated by a handful of blowouts and fights the value head for the
shared body's capacity.

The target is therefore ``tanh(margin / score_scale)`` and the head itself ends in
``tanh``. That puts both heads on the same bounded ±1 scale — which is what makes
``NetConfig.score_loss_weight`` mean what it says — and spends the resolution where the
mass is. At the default ``score_scale = 25``:

===========  ======
``margin``   target
===========  ======
1            0.040
3            0.120
10           0.380
25           0.762
60           0.984
88           0.999
===========  ======

Positions with **no** margin (v2 opening rows: a DAG node has many games through it, and
``link`` aggregates their *outcomes*, not their margins) carry ``None``, which becomes
``NaN`` here and is masked out of the loss rather than being invented as a zero.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


def scale_margin(margin: float, score_scale: float) -> float:
    """One margin → its bounded score-head target ``tanh(margin / score_scale)``.

    Args:
        margin: Final score margin from the side to move (positive = they won by that
            many points).
        score_scale: ``NetConfig.score_scale`` — the margin that maps to ``tanh(1) ≈
            0.76``. Must be positive.

    Returns:
        The target in ``(-1, 1)``.
    """
    if score_scale <= 0.0:
        raise ValueError(f"score_scale must be positive, got {score_scale}")
    return float(np.tanh(margin / score_scale))


def scale_margins(margins: Sequence[float | None], score_scale: float) -> NDArray[np.float32]:
    """Vectorised :func:`scale_margin`; ``None`` becomes ``NaN`` (masked in the loss).

    Args:
        margins: Per-position margins, index-aligned with the training examples.
            ``None`` marks a position with no single margin (v2 opening rows).
        score_scale: ``NetConfig.score_scale``.

    Returns:
        ``(len(margins),)`` float32 targets, ``NaN`` wherever the margin was ``None``.
    """
    if score_scale <= 0.0:
        raise ValueError(f"score_scale must be positive, got {score_scale}")
    raw = np.array([np.nan if margin is None else float(margin) for margin in margins], dtype=np.float64)
    return np.tanh(raw / score_scale).astype(np.float32)
