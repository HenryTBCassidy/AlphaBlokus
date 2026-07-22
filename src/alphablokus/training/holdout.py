"""Game-level held-out splits + out-of-sample fit metrics for the capacity probe.

The `xl` demotion in ``docs/research/xl-training-scaleup.md`` A4 rested on
diagnostics computed against the net's *own* self-play targets — circular, and
structurally unable to detect a capacity limit (a capacity-bound net generates
weaker targets and then fits them comfortably). The decisive experiment is
out-of-sample: can a bigger net fit the *same frozen data* better on positions
it never trained on? (docs/research/regression-and-next-steps.md §3.1/§3.4,
docs/plans/post-regression-recovery.md P6.)

Two pieces live here so ``scripts/capacity_probe.py`` stays a thin CLI:

- :func:`split_games_holdout` — split **by game**, never by position: positions
  within a game are strongly correlated (shared trajectory, shared outcome
  label, symmetry-augmented pairs), so a position-level split leaks the
  held-out answers into training.
- :func:`evaluate_holdout` — mean policy cross-entropy / KL and value MSE of a
  predictor over held-out examples. Typed against a structural protocol
  (anything with ``predict_encoded``) so this framework module never imports
  ``games.*``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from alphablokus.storage.sparse_policy import as_dense

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

    from alphablokus.selfplay.episode import GameExamples, ProcessedExample

# Probabilities are clipped here before the log so a zero predicted mass on a
# target action contributes a large-but-finite CE instead of inf.
_LOG_EPS = 1e-12


class SupportsEncodedPrediction(Protocol):
    """The one inference surface holdout evaluation needs (structural).

    Matched by ``BaseNNetWrapper.predict_encoded`` without this module ever
    importing ``games.*`` (the registry rule).
    """

    def predict_encoded(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(N, C, H, W) encoded boards → ((N, A) policy probs, (N,) values)."""
        ...


@dataclass(frozen=True)
class HoldoutMetrics:
    """Out-of-sample fit of a predictor on held-out self-play examples.

    Attributes:
        policy_ce: Mean cross-entropy of the predicted policy against the
            stored search targets, in nats. The arm-comparison quantity.
        policy_kl: ``policy_ce − target_entropy`` — the reducible part. The
            stochastic-target noise floor (two searches of one position give
            different targets) lives in ``target_entropy`` and is common to
            every arm evaluated on the same split, so it cancels in deltas.
        target_entropy: Mean entropy of the stored targets, in nats.
        value_mse: Mean squared error of the value head against game outcomes.
        n_positions: Number of held-out positions evaluated.
    """

    policy_ce: float
    policy_kl: float
    target_entropy: float
    value_mse: float
    n_positions: int


def split_games_holdout(
    games: Sequence[GameExamples],
    holdout_fraction: float,
    seed: int,
) -> tuple[list[GameExamples], list[GameExamples]]:
    """Split self-play games into (train, holdout) at **game** granularity.

    Args:
        games: Per-game example lists (``SelfPlayStore.load_games`` shape).
        holdout_fraction: Fraction of *games* to hold out, in ``[0, 1)``. A
            non-zero fraction holds out at least one game.
        seed: RNG seed — the same (games, fraction, seed) always yields the
            same split, so every probe arm trains and evaluates on identical
            data.

    Returns:
        ``(train_games, holdout_games)`` — a partition of ``games`` (every game
        in exactly one side, original order preserved within each side).
    """
    if not 0.0 <= holdout_fraction < 1.0:
        raise ValueError(f"holdout_fraction must be in [0, 1), got {holdout_fraction}")
    if not games:
        raise ValueError("split_games_holdout needs at least one game")

    n_holdout = max(1, round(len(games) * holdout_fraction)) if holdout_fraction > 0.0 else 0
    rng = np.random.default_rng(seed)
    holdout_indices = set(rng.choice(len(games), size=n_holdout, replace=False).tolist())
    train = [game for i, game in enumerate(games) if i not in holdout_indices]
    holdout = [game for i, game in enumerate(games) if i in holdout_indices]
    return train, holdout


def evaluate_holdout(
    predictor: SupportsEncodedPrediction,
    examples: Sequence[ProcessedExample],
    *,
    encode_fn: Callable[[NDArray], NDArray],
    action_size: int,
    batch_size: int = 512,
) -> HoldoutMetrics:
    """Mean policy CE/KL + value MSE of ``predictor`` over held-out examples.

    Args:
        predictor: Anything with ``predict_encoded`` (e.g. a net wrapper).
        examples: Held-out ``(compact_board, sparse_policy, value)`` tuples.
        encode_fn: ``IGame.encode_compact`` for the game the boards came from.
        action_size: Dense action-space size the sparse policies index into.
        batch_size: Forward-pass batch size (memory knob only).
    """
    if not examples:
        raise ValueError("evaluate_holdout needs at least one example")

    ce_sum = 0.0
    entropy_sum = 0.0
    mse_sum = 0.0
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        planes = np.stack([encode_fn(board) for board, _pi, _value in batch])
        targets = np.stack([as_dense(pi, action_size) for _board, pi, _value in batch])
        outcomes = np.array([value for _board, _pi, value in batch], dtype=np.float64)

        policies, values = predictor.predict_encoded(planes)
        log_policies = np.log(np.clip(policies.astype(np.float64), _LOG_EPS, None))
        ce_sum += float(-(targets * log_policies).sum())
        positive = targets > 0.0
        entropy_sum += float(-(targets[positive] * np.log(targets[positive])).sum())
        mse_sum += float(((outcomes - values.astype(np.float64)) ** 2).sum())

    n = len(examples)
    policy_ce = ce_sum / n
    target_entropy = entropy_sum / n
    return HoldoutMetrics(
        policy_ce=policy_ce,
        policy_kl=policy_ce - target_entropy,
        target_entropy=target_entropy,
        value_mse=mse_sum / n,
        n_positions=n,
    )
