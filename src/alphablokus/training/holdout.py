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
  ``docs/plans/pentobi-distillation.md`` D7): held-out top-1 and top-3 accuracy
  against the expert's move (legal-restricted) and value calibration split by
  side-to-move (colour-conditional — Blokus outcomes are heavily
  colour-skewed, so a pooled calibration curve hides a per-colour bias).
- :func:`evaluate_score_head`, :func:`evaluate_ownership_head` and
  :func:`evaluate_reply_head` — one held-out metric set per **auxiliary** head
  (``docs/plans/score-auxiliary-target.md`` S6 and
  ``docs/plans/supervised-network-improvements.md`` N4/N5). Each answers "is this
  head learning at all", each returns ``None`` rather than a fabricated zero when the
  head does not exist, and each reports a **baseline** alongside its raw loss: a small
  loss against an easy baseline is not skill, and the A/B harness reads the skill.
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

# Rank cut-off of the second imitation-agreement figure. Three, not five: Blokus
# positions typically have hundreds of legal moves, so a wide window would report
# agreement that no player would recognise as "the same idea".
_TOP_K_AGREEMENT = 3


class SupportsEncodedPrediction(Protocol):
    """The one inference surface holdout evaluation needs (structural).

    Matched by ``BaseNNetWrapper.predict_encoded`` without this module ever
    importing ``games.*`` (the registry rule).
    """

    def predict_encoded(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(N, C, H, W) encoded boards → ((N, A) policy probs, (N,) values)."""
        ...


class SupportsAuxPrediction(Protocol):
    """The diagnostics-only surface that also returns the auxiliary heads' outputs.

    Deliberately *not* :class:`SupportsEncodedPrediction`: no auxiliary head is part of
    the surface anything uses to choose a move, and keeping the two protocols apart is
    what makes that visible in the type system.
    """

    def predict_encoded_aux(self, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """(N, C, H, W) encoded boards → (policy probs, values, {head: output})."""
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
class ScoreHeadMetrics:
    """Held-out fit of the auxiliary score head (plan S6).

    Attributes:
        score_mse: Mean squared error against ``tanh(margin / score_scale)``, over the
            held-out positions that *have* a margin.
        constant_mse: The MSE of predicting the held-out mean target for every position —
            the floor a head that has learnt nothing but "games are usually close" would
            score. Without it a small ``score_mse`` reads as success when it may just be a
            consequence of the tanh squashing everything toward the middle.
        score_skill: ``1 − score_mse / constant_mse``. Zero means the head has learnt the
            mean and nothing else; this is the number S8 reads to decide whether the head
            has an easy job (skill near zero at a tiny MSE ⇒ ``score_scale`` too small).
        n_positions: Held-out positions with a margin, i.e. those actually scored.
        n_skipped: Held-out positions with no margin (masked out, as in training).
    """

    score_mse: float
    constant_mse: float
    score_skill: float
    n_positions: int
    n_skipped: int


def evaluate_score_head(
    predictor: SupportsAuxPrediction,
    examples: Sequence[ProcessedExample],
    margins: Sequence[float | None],
    *,
    score_scale: float,
    encode_fn: Callable[[NDArray], NDArray],
    batch_size: int = 512,
) -> ScoreHeadMetrics | None:
    """Score-head MSE on held-out positions, against the same target training uses.

    Returns ``None`` when the predictor has no score head, or when no held-out position
    carries a margin — both "there is nothing to report", never a fabricated zero.

    Args:
        predictor: Anything with ``predict_encoded_aux``.
        examples: Held-out ``(compact_board, sparse_policy, value)`` tuples.
        margins: Raw margins index-aligned with ``examples``; ``None`` = no margin.
        score_scale: ``NetConfig.score_scale`` — must match the value trained with, or the
            reported MSE is against a different target than the one optimised.
        encode_fn: ``IGame.encode_compact`` for the game the boards came from.
        batch_size: Forward-pass batch size (memory knob only).
    """
    from alphablokus.training.score_target import scale_margins

    if len(examples) != len(margins):
        raise ValueError(f"{len(examples)} examples but {len(margins)} margins; they must be index-aligned")
    if not examples:
        raise ValueError("evaluate_score_head needs at least one example")

    predicted = np.empty(len(examples), dtype=np.float64)
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        planes = np.stack([encode_fn(board) for board, _pi, _value in batch])
        _, _, aux = predictor.predict_encoded_aux(planes)
        if "score" not in aux:
            return None
        predicted[start : start + len(batch)] = aux["score"].astype(np.float64)

    targets = scale_margins(margins, score_scale).astype(np.float64)
    scored = np.isfinite(targets)
    if not scored.any():
        return None

    errors = predicted[scored] - targets[scored]
    score_mse = float(np.mean(errors**2))
    constant_mse = float(np.mean((targets[scored] - targets[scored].mean()) ** 2))
    return ScoreHeadMetrics(
        score_mse=score_mse,
        constant_mse=constant_mse,
        score_skill=0.0 if constant_mse <= 0.0 else 1.0 - score_mse / constant_mse,
        n_positions=int(scored.sum()),
        n_skipped=int((~scored).sum()),
    )


@dataclass(frozen=True)
class OwnershipHeadMetrics:
    """Held-out fit of the auxiliary ownership head (plan N4).

    Attributes:
        cross_entropy: Mean per-cell cross-entropy against the final board's ownership,
            in nats, over the held-out cells that have a label.
        marginal_cross_entropy: The cross-entropy of predicting the held-out **marginal**
            class distribution for every cell — the floor a head that has learnt nothing
            but "most cells end up owned by somebody" would score. Without it a small
            cross-entropy reads as success when it may only reflect a skewed prior.
        skill: ``1 − cross_entropy / marginal_cross_entropy``. Zero means the head has
            learnt the marginal and nothing else.
        accuracy: Fraction of labelled cells whose argmax class is correct.
        n_positions: Held-out positions with a final board, i.e. those actually scored.
        n_skipped: Held-out positions with no final board (masked out, as in training).
    """

    cross_entropy: float
    marginal_cross_entropy: float
    skill: float
    accuracy: float
    n_positions: int
    n_skipped: int


def evaluate_ownership_head(
    predictor: SupportsAuxPrediction,
    examples: Sequence[ProcessedExample],
    ownership: Sequence[NDArray[np.int8] | None],
    *,
    encode_fn: Callable[[NDArray], NDArray],
    batch_size: int = 512,
) -> OwnershipHeadMetrics | None:
    """Per-cell ownership cross-entropy/accuracy on held-out positions.

    Returns ``None`` when the predictor has no ownership head, or when no held-out
    position has a final board — both "there is nothing to report", never a fabricated
    zero.

    Args:
        predictor: Anything with ``predict_encoded_aux``.
        examples: Held-out ``(compact_board, sparse_policy, value)`` tuples.
        ownership: ``{-1, 0, +1}`` maps in each position's own canonical frame,
            index-aligned with ``examples``; ``None`` = no final board.
        encode_fn: ``IGame.encode_compact`` for the game the boards came from.
        batch_size: Forward-pass batch size (memory knob only).
    """
    if len(examples) != len(ownership):
        raise ValueError(f"{len(examples)} examples but {len(ownership)} ownership maps; must be index-aligned")
    if not examples:
        raise ValueError("evaluate_ownership_head needs at least one example")

    scored = [index for index, target in enumerate(ownership) if target is not None]
    if not scored:
        return None

    log_probabilities: list[NDArray[np.float64]] = []
    for start in range(0, len(scored), batch_size):
        batch_indices = scored[start : start + batch_size]
        planes = np.stack([encode_fn(examples[index][0]) for index in batch_indices])
        _, _, aux = predictor.predict_encoded_aux(planes)
        if "ownership" not in aux:
            return None
        # (n, classes, rows, cols) → (n, classes, cells): the metric is per cell and
        # never needs the board's 2-D shape.
        probabilities = aux["ownership"].astype(np.float64)
        flat = probabilities.reshape(len(batch_indices), probabilities.shape[1], -1)
        log_probabilities.append(np.log(np.clip(flat, _LOG_EPS, None)))

    log_probs = np.concatenate(log_probabilities, axis=0)
    # ``+1`` maps the stored {-1, 0, +1} map onto the head's class order, exactly as
    # the training-side target source does.
    labels = np.stack([np.asarray(ownership[index]).reshape(-1) + 1 for index in scored]).astype(np.int64)
    rows = np.arange(labels.shape[0])[:, None]
    cells = np.arange(labels.shape[1])[None, :]
    cross_entropy = float(-log_probs[rows, labels, cells].mean())
    accuracy = float((log_probs.argmax(axis=1) == labels).mean())

    marginal = np.bincount(labels.reshape(-1), minlength=log_probs.shape[1]) / labels.size
    marginal_cross_entropy = float(-(marginal * np.log(np.clip(marginal, _LOG_EPS, None))).sum())
    return OwnershipHeadMetrics(
        cross_entropy=cross_entropy,
        marginal_cross_entropy=marginal_cross_entropy,
        skill=0.0 if marginal_cross_entropy <= 0.0 else 1.0 - cross_entropy / marginal_cross_entropy,
        accuracy=accuracy,
        n_positions=len(scored),
        n_skipped=len(examples) - len(scored),
    )


@dataclass(frozen=True)
class ReplyHeadMetrics:
    """Held-out fit of the auxiliary opponent-reply head (plan N5).

    Attributes:
        policy_ce: Mean cross-entropy of the predicted reply distribution against the
            opponent's actual next-ply target, in nats.
        policy_kl: ``policy_ce − target_entropy`` — the reducible part, directly
            comparable to :attr:`HoldoutMetrics.policy_kl` for the main policy head.
        target_entropy: Mean entropy of the reply targets, in nats.
        top1_accuracy: Fraction of scored positions where the head's most likely reply
            is the target's most likely reply.
        n_positions: Held-out positions with a next ply, i.e. those actually scored.
        n_skipped: Held-out positions with no next ply (each game's final position),
            masked out exactly as in training.
    """

    policy_ce: float
    policy_kl: float
    target_entropy: float
    top1_accuracy: float
    n_positions: int
    n_skipped: int


def evaluate_reply_head(
    predictor: SupportsAuxPrediction,
    examples: Sequence[ProcessedExample],
    replies: Sequence[object | None],
    *,
    action_size: int,
    encode_fn: Callable[[NDArray], NDArray],
    batch_size: int = 512,
) -> ReplyHeadMetrics | None:
    """Reply-head cross-entropy/KL and top-1 agreement on held-out positions.

    Returns ``None`` when the predictor has no reply head, or when no held-out position
    has a next ply — both "there is nothing to report", never a fabricated zero.

    Args:
        predictor: Anything with ``predict_encoded_aux``.
        examples: Held-out ``(compact_board, sparse_policy, value)`` tuples.
        replies: The opponent's next-ply distribution per position (sparse
            ``(indices, values)`` or dense), index-aligned with ``examples``;
            ``None`` = no next ply.
        action_size: Dense action-space size the sparse targets index into.
        encode_fn: ``IGame.encode_compact`` for the game the boards came from.
        batch_size: Forward-pass batch size (memory knob only).
    """
    if len(examples) != len(replies):
        raise ValueError(f"{len(examples)} examples but {len(replies)} reply targets; must be index-aligned")
    if not examples:
        raise ValueError("evaluate_reply_head needs at least one example")

    scored = [index for index, target in enumerate(replies) if target is not None]
    if not scored:
        return None

    ce_sum = 0.0
    entropy_sum = 0.0
    top1_hits = 0
    for start in range(0, len(scored), batch_size):
        batch_indices = scored[start : start + batch_size]
        planes = np.stack([encode_fn(examples[index][0]) for index in batch_indices])
        _, _, aux = predictor.predict_encoded_aux(planes)
        if "reply" not in aux:
            return None
        predicted = aux["reply"].astype(np.float64)
        targets = np.stack([as_dense(replies[index], action_size) for index in batch_indices]).astype(np.float64)
        log_predicted = np.log(np.clip(predicted, _LOG_EPS, None))
        ce_sum += float(-(targets * log_predicted).sum())
        positive = targets > 0.0
        entropy_sum += float(-(targets[positive] * np.log(targets[positive])).sum())
        top1_hits += int((predicted.argmax(axis=1) == targets.argmax(axis=1)).sum())

    n = len(scored)
    policy_ce = ce_sum / n
    target_entropy = entropy_sum / n
    return ReplyHeadMetrics(
        policy_ce=policy_ce,
        policy_kl=policy_ce - target_entropy,
        target_entropy=target_entropy,
        top1_accuracy=top1_hits / n,
        n_positions=n,
        n_skipped=len(examples) - n,
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
        top3_accuracy: Fraction of positions where the expert's move is in the
            predictor's top three *legal* moves. Less brittle than top-1 in a game
            with several near-equivalent good moves, so it moves earlier and more
            smoothly under a small improvement — which is why the A/B harness reads
            both (plan N1).
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
    top3_accuracy: float = 0.0

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
    top3_hits = 0
    predicted_values = np.empty(len(examples), dtype=np.float64)
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        planes = np.stack([encode_fn(board) for board, _pi, _value in batch])
        policies, values = predictor.predict_encoded(planes)
        predicted_values[start : start + len(batch)] = values.astype(np.float64)
        for row, (_board, pi, _value) in enumerate(batch):
            support = pi[0] if isinstance(pi, tuple) else np.flatnonzero(pi)
            expert = int(expert_actions[start + row])
            # One argsort over the legal support serves both ranks; ``[::-1]`` puts the
            # most probable legal move first.
            ranked = support[np.argsort(policies[row][support])[::-1]]
            top1_hits += int(int(ranked[0]) == expert)
            top3_hits += int(expert in {int(action) for action in ranked[:_TOP_K_AGREEMENT]})

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
        top3_accuracy=top3_hits / len(examples),
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
