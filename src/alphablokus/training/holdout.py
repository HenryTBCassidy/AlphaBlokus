"""Game-level held-out splits + out-of-sample fit metrics for the capacity probe.

The `xl` demotion in ``docs/research/xl-training-scaleup.md`` A4 rested on
diagnostics computed against the net's *own* self-play targets — circular, and
structurally unable to detect a capacity limit (a capacity-bound net generates
weaker targets and then fits them comfortably). The decisive experiment is
out-of-sample: can a bigger net fit the *same frozen data* better on positions
it never trained on? (docs/research/regression-and-next-steps.md §3.1/§3.4,
docs/plans/post-regression-recovery.md P6.)

Three pieces live here so ``scripts/capacity_probe.py`` and
``scripts/distill_sl.py`` stay thin CLIs:

- :func:`split_games_holdout` — split **by game**, never by position: positions
  within a game are strongly correlated (shared trajectory, shared outcome
  label, symmetry-augmented pairs), so a position-level split leaks the
  held-out answers into training. Generic over the game item type so both
  self-play ``GameExamples`` and the distillation corpus's per-game row groups
  split through the same code.
- :func:`evaluate_holdout` — mean policy cross-entropy / KL and value MSE of a
  predictor over held-out examples. Typed against a structural protocol
  (anything with ``predict_encoded``) so this framework module never imports
  ``games.*``.
- :func:`evaluate_imitation_diagnostics` — the SL-distillation extras (plan
  ``docs/plans/pentobi-distillation.md`` D7): held-out top-1 accuracy against
  the expert's move (legal-restricted argmax) and value calibration split by
  side-to-move (colour-conditional — Blokus outcomes are heavily
  colour-skewed, so a pooled calibration curve hides a per-colour bias).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeVar

import numpy as np

from alphablokus.storage.sparse_policy import as_dense

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

    from alphablokus.selfplay.episode import ProcessedExample

# Probabilities are clipped here before the log so a zero predicted mass on a
# target action contributes a large-but-finite CE instead of inf.
_LOG_EPS = 1e-12

# The game-shaped item a holdout split partitions — self-play ``GameExamples``
# or any other per-game grouping (e.g. the distillation corpus's row groups).
TGameItem = TypeVar("TGameItem")

# Value-head reliability diagram resolution: predicted v ∈ [-1, 1] in 10 buckets
# (matches the training-loop calibration diagnostic in ``games/base_wrapper.py``).
_CALIBRATION_BUCKETS = 10


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
    games: Sequence[TGameItem],
    holdout_fraction: float,
    seed: int,
) -> tuple[list[TGameItem], list[TGameItem]]:
    """Split games into (train, holdout) at **game** granularity.

    Args:
        games: One item per game — self-play ``GameExamples`` lists
            (``SelfPlayStore.load_games`` shape) or any other per-game grouping.
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


@dataclass(frozen=True)
class ColourValueCalibration:
    """Value-head calibration over the held-out positions of one side-to-move.

    Attributes:
        player: Side to move this row conditions on (+1 White, -1 Black).
        n_positions: Held-out positions with this side to move.
        mean_predicted: Mean predicted value over those positions.
        mean_outcome: Mean actual outcome — ``mean_predicted − mean_outcome``
            is the colour's calibration bias in one number.
        value_mse: Value MSE restricted to this colour.
        bucket_centers: Centres of the 10 reliability buckets over predicted
            v ∈ [-1, 1] (same binning as the training-loop diagnostic).
        bucket_mean_outcomes: Mean actual outcome per bucket; ``None`` for an
            empty bucket (kept ``None`` rather than NaN so the row serialises
            to strict JSON).
        bucket_counts: Positions per bucket.
    """

    player: int
    n_positions: int
    mean_predicted: float
    mean_outcome: float
    value_mse: float
    bucket_centers: tuple[float, ...]
    bucket_mean_outcomes: tuple[float | None, ...]
    bucket_counts: tuple[int, ...]


@dataclass(frozen=True)
class ImitationDiagnostics:
    """Held-out imitation metrics for SL distillation (plan D7).

    Attributes:
        top1_accuracy: Fraction of positions where the predictor's best *legal*
            move is the expert's move. Legal-restricted on purpose: an illegal
            high-prior action never plays, so it should not cost the net a hit
            it would score at the board.
        n_positions: Held-out positions evaluated.
        calibration: One :class:`ColourValueCalibration` per side-to-move
            present, ordered by ``player`` ascending (Black -1 first).
        value_mse: Value MSE over all held-out positions.
        colour_only_value_mse: The MSE of a model that sees **only whose turn it
            is** — it predicts each colour's mean held-out outcome and never looks
            at the board. This is the floor a value head has to beat to be reading
            positions at all, and in a game with a large first-player advantage the
            floor is high: measured on real v2 data, White-to-move positions are 79%
            wins and Black-to-move 78% losses, so guessing from the colour alone
            scores 0.30 MSE against 0.84 for always predicting a draw. Without this
            number a value head that has learnt nothing but the colour prior looks
            like a value head that works.
    """

    top1_accuracy: float
    n_positions: int
    calibration: tuple[ColourValueCalibration, ...]
    value_mse: float = 0.0
    colour_only_value_mse: float = 0.0

    @property
    def value_skill(self) -> float:
        """How much of the colour-only baseline's error the value head removes.

        ``1 - mse / colour_only_mse``. Zero means the head has learnt the colour
        prior and nothing else; negative means it is worse than that.
        """
        if self.colour_only_value_mse <= 0.0:
            return 0.0
        return 1.0 - self.value_mse / self.colour_only_value_mse


def evaluate_imitation_diagnostics(
    predictor: SupportsEncodedPrediction,
    examples: Sequence[ProcessedExample],
    expert_actions: Sequence[int],
    players: Sequence[int],
    *,
    encode_fn: Callable[[NDArray], NDArray],
    batch_size: int = 512,
) -> ImitationDiagnostics:
    """Held-out top-1 accuracy vs the expert + colour-conditional value calibration.

    ``examples`` must be index-aligned with ``expert_actions`` and ``players``
    (the unaugmented output of the corpus dataloader has exactly this alignment).
    Each example's sparse policy support doubles as the position's **legal set**
    (label smoothing spreads over exactly the legal moves), so top-1 accuracy is
    computed as the argmax of the predicted policy *restricted to that support*
    — no move generation happens here, keeping this module game-free.

    Args:
        predictor: Anything with ``predict_encoded`` (e.g. a net wrapper).
        examples: Held-out ``(compact_board, sparse_policy, value)`` tuples with
            smoothed (legal-support) sparse policies.
        expert_actions: The expert's action index per position.
        players: Side to move per position (+1 / -1).
        encode_fn: ``IGame.encode_compact`` for the game the boards came from.
        batch_size: Forward-pass batch size (memory knob only).
    """
    if not examples:
        raise ValueError("evaluate_imitation_diagnostics needs at least one example")
    if not (len(examples) == len(expert_actions) == len(players)):
        raise ValueError(
            f"misaligned inputs: {len(examples)} examples, {len(expert_actions)} expert actions, "
            f"{len(players)} players",
        )

    top1_hits = 0
    predicted_values = np.empty(len(examples), dtype=np.float64)
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        planes = np.stack([encode_fn(board) for board, _pi, _value in batch])
        policies, values = predictor.predict_encoded(planes)
        predicted_values[start : start + len(batch)] = values.astype(np.float64)
        for row, (_board, pi, _value) in enumerate(batch):
            support = pi[0] if isinstance(pi, tuple) else np.flatnonzero(pi)
            best_legal = int(support[int(np.argmax(policies[row][support]))])
            top1_hits += int(best_legal == int(expert_actions[start + row]))

    player_arr = np.asarray(players, dtype=np.int64)
    outcomes = np.array([value for _board, _pi, value in examples], dtype=np.float64)
    calibration = tuple(
        _colour_calibration(colour, predicted_values[player_arr == colour], outcomes[player_arr == colour])
        for colour in sorted(set(player_arr.tolist()))
    )
    colour_means = {colour: float(outcomes[player_arr == colour].mean()) for colour in set(player_arr.tolist())}
    colour_only = np.array([colour_means[int(colour)] for colour in player_arr], dtype=np.float64)
    return ImitationDiagnostics(
        top1_accuracy=top1_hits / len(examples),
        n_positions=len(examples),
        calibration=calibration,
        value_mse=float(np.mean((predicted_values - outcomes) ** 2)),
        colour_only_value_mse=float(np.mean((colour_only - outcomes) ** 2)),
    )


def _colour_calibration(
    player: int,
    predicted: NDArray[np.float64],
    outcomes: NDArray[np.float64],
) -> ColourValueCalibration:
    """Reliability diagram + summary stats for one side-to-move's positions."""
    edges = np.linspace(-1.0, 1.0, _CALIBRATION_BUCKETS + 1)
    bucket_idx = np.clip(np.digitize(predicted, edges) - 1, 0, _CALIBRATION_BUCKETS - 1)
    mean_outcomes: list[float | None] = []
    counts: list[int] = []
    for bucket in range(_CALIBRATION_BUCKETS):
        mask = bucket_idx == bucket
        counts.append(int(mask.sum()))
        mean_outcomes.append(float(outcomes[mask].mean()) if counts[-1] else None)
    return ColourValueCalibration(
        player=player,
        n_positions=len(predicted),
        mean_predicted=float(predicted.mean()),
        mean_outcome=float(outcomes.mean()),
        value_mse=float(((predicted - outcomes) ** 2).mean()),
        bucket_centers=tuple(((edges[:-1] + edges[1:]) / 2.0).tolist()),
        bucket_mean_outcomes=tuple(mean_outcomes),
        bucket_counts=tuple(counts),
    )
