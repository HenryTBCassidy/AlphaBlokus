"""Colour-conditional value diagnostic.

The question this answers: **does the value head know anything about the position,
or has it only learnt whose turn it is?**

In Blokus Duo the first mover wins the large majority of decisive games, so
"White is probably winning" is a strong prediction that requires no understanding
of the board at all. A value head that has learnt only that will look competent
on any pooled metric — its MSE is genuinely low — while contributing nothing to
search. And it does not contribute nothing harmlessly: Gumbel's training target
is ``softmax(logits + σ(completed Q))``, and completed-Q substitutes the value
net for every unvisited action, so a colour-prior value head corrupts the target
the policy trains on at every simulation count and every net size.

The measurement is *value skill* — how much of a colour-only baseline's error the
head removes:

    ``skill = 1 - mse / baseline_mse``

Zero means the head has learnt the colour prior and nothing else. Negative means
it is worse than guessing from the colour.

Two baselines are reported, because the colour-only one is too easy to beat for
the wrong reason: game phase also predicts the outcome (positions near the end of
a decided game are easy), so a head that has learnt only "colour plus how far
along we are" scores positive skill against the colour-only baseline. The
**colour×phase** baseline removes that too, and skill against it is the honest
number.

Every interval here is a game-cluster bootstrap
(:mod:`alphablokus.bootstrap`). Position-level intervals on this
statistic are roughly ``sqrt(positions per game)`` too narrow, which is enough to
turn "no demonstrable skill" into "skill" and back.

**One bias worth knowing about, because it runs the safe way.** The baseline is
re-estimated inside every bootstrap resample, which is what propagates the
uncertainty in the baseline itself into the interval. A refit baseline enjoys a
small in-sample advantage on each resample, so measured skill is biased very
slightly *downward* — a head with genuinely zero skill reports an interval whose
upper end sits a hair below zero rather than straddling it. The size of the effect
scales like (baseline cells / games): with 2 cells and 40 games it is under 0.001,
and with the colour×phase baseline's 6 cells it is still small. It is left in
deliberately, because the alternative (fitting the baseline once on the full
sample and holding it fixed) throws away the baseline's own sampling error and
produces intervals that are too narrow — the exact failure this module exists to
avoid. Read "skill ≤ 0" as a slightly conservative verdict, not an exact one.

**Known limitation.** Mover colour is *inferred* from the compact board's piece
counts, because self-play stores canonical (side-to-move perspective) boards and
discards the absolute mover. The inference is exact until the first pass and
ambiguous afterwards; ambiguous positions are excluded and counted, never
guessed. Threading the true side-to-move through the self-play example tuple
(``player``, as the Pentobi corpus already stores) would remove the caveat and is
tracked separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from alphablokus.bootstrap import BootstrapResult, game_cluster_bootstrap

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

# Mover-colour codes. 0 means "could not be determined" — see the module note.
WHITE_TO_MOVE = 1
BLACK_TO_MOVE = -1
COLOUR_UNKNOWN = 0

# Game phase by pieces placed across both players (42 max in Blokus Duo).
# Thirds, so each bucket holds a comparable span of the game.
PHASE_EARLY = "early"
PHASE_MID = "mid"
PHASE_LATE = "late"
_PHASE_BOUNDARIES = (14, 28)


def infer_mover_colour(compact_board: NDArray) -> int:
    """Infer whose turn it is from a canonical compact board.

    The compact board is signed by *perspective*: positive piece ids belong to the
    side to move, negative to the opponent. In a pass-free game the two sides have
    placed equal numbers of pieces when White is to move, and the opponent has one
    more when Black is to move. Once a player has passed, that parity breaks and
    the colour is genuinely unrecoverable from the board alone.

    Args:
        compact_board: Canonical compact board (signed piece ids).

    Returns:
        :data:`WHITE_TO_MOVE`, :data:`BLACK_TO_MOVE`, or :data:`COLOUR_UNKNOWN`.
    """
    values = np.unique(compact_board)
    mine = {int(v) for v in values if v > 0}
    theirs = {int(-v) for v in values if v < 0}
    if len(mine) == len(theirs):
        return WHITE_TO_MOVE
    if len(theirs) == len(mine) + 1:
        return BLACK_TO_MOVE
    return COLOUR_UNKNOWN


def game_phase(compact_board: NDArray) -> str:
    """Bucket a position by how far into the game it is.

    Uses pieces placed by both players, which is monotone in game progress and
    does not depend on knowing the mover.
    """
    placed = len({int(v) for v in np.unique(compact_board) if v != 0})
    if placed <= _PHASE_BOUNDARIES[0]:
        return PHASE_EARLY
    if placed <= _PHASE_BOUNDARIES[1]:
        return PHASE_MID
    return PHASE_LATE


def _group_mean_baseline(targets: NDArray, keys: NDArray) -> NDArray:
    """Predict each position's group mean — the baseline a value head must beat.

    ``keys`` labels the group (colour, or colour×phase). Every group present has
    at least one member by construction, so the baseline is always defined.
    """
    baseline = np.empty_like(targets, dtype=np.float64)
    for key in np.unique(keys):
        mask = keys == key
        baseline[mask] = targets[mask].mean()
    return baseline


def _skill(predictions: NDArray, targets: NDArray, keys: NDArray) -> float:
    """``1 - mse / baseline_mse`` for the baseline defined by ``keys``."""
    baseline = _group_mean_baseline(targets, keys)
    baseline_mse = float(np.mean((baseline - targets) ** 2))
    if baseline_mse <= 0.0:
        # A baseline with zero error cannot be improved on; skill is undefined
        # rather than infinite. The bootstrap drops non-finite resamples.
        return float("nan")
    mse = float(np.mean((predictions - targets) ** 2))
    return 1.0 - mse / baseline_mse


def _skill_statistic(
    predictions: NDArray,
    targets: NDArray,
    keys: NDArray,
) -> Callable[[NDArray], float]:
    """A ``statistic(position_indices) -> float`` closure for the bootstrap.

    A factory rather than an inline lambda so each call site's arrays are bound in
    their own scope — an inline closure inside a loop would evaluate against the
    last iteration's arrays if the call were ever deferred.
    """

    def statistic(indices: NDArray) -> float:
        return _skill(predictions[indices], targets[indices], keys[indices])

    return statistic


@dataclass(frozen=True)
class ColourSlice:
    """Value-head behaviour restricted to one side-to-move.

    Attributes:
        colour: :data:`WHITE_TO_MOVE` or :data:`BLACK_TO_MOVE`.
        n_positions: Positions with this side to move.
        n_games: Distinct source games contributing them — the effective n.
        mean_prediction: Mean predicted value.
        mean_target: Mean actual outcome. ``mean_prediction - mean_target`` is
            this colour's calibration bias in one number.
        value_mse: Value MSE restricted to this colour.
    """

    colour: int
    n_positions: int
    n_games: int
    mean_prediction: float
    mean_target: float
    value_mse: float

    @property
    def bias(self) -> float:
        """Signed calibration bias: positive means over-optimistic."""
        return self.mean_prediction - self.mean_target


@dataclass(frozen=True)
class ColourValueDiagnostic:
    """The full colour-conditional value diagnostic for one checkpoint.

    Attributes:
        n_positions: Positions used (ambiguous-colour positions excluded).
        n_games: Distinct source games — the effective sample size.
        n_excluded: Positions dropped because the mover could not be inferred.
        value_mse: Pooled value MSE.
        colour_only_mse: MSE of predicting each colour's mean outcome.
        colour_phase_mse: MSE of predicting each colour×phase cell's mean outcome.
        skill_vs_colour: Value skill against the colour-only baseline, with a
            game-cluster interval. An interval straddling zero means the head has
            no demonstrable skill beyond the colour prior.
        skill_vs_colour_phase: Value skill against the colour×phase baseline —
            the honest number, since phase is also freely available.
        colour_target_correlation: How strongly outcomes track mover colour.
        colour_prediction_correlation: How strongly the head's output tracks mover
            colour. When this materially exceeds the previous number, the net is
            more certain that colour decides the game than the game is.
        slices: One :class:`ColourSlice` per colour present, White first.
        phase_skill: Skill against the colour-only baseline computed within each
            phase, so a head that works early and fails late is visible.
    """

    n_positions: int
    n_games: int
    n_excluded: int
    value_mse: float
    colour_only_mse: float
    colour_phase_mse: float
    skill_vs_colour: BootstrapResult
    skill_vs_colour_phase: BootstrapResult
    colour_target_correlation: float
    colour_prediction_correlation: float
    slices: tuple[ColourSlice, ...]
    phase_skill: tuple[tuple[str, BootstrapResult], ...]

    def as_payload(self) -> dict[str, object]:
        """Serialise for the metrics store / report payload."""
        return {
            "n_positions": self.n_positions,
            "n_games": self.n_games,
            "n_excluded": self.n_excluded,
            "value_mse": round(self.value_mse, 5),
            "colour_only_mse": round(self.colour_only_mse, 5),
            "colour_phase_mse": round(self.colour_phase_mse, 5),
            "skill_vs_colour": self.skill_vs_colour.as_payload(),
            "skill_vs_colour_phase": self.skill_vs_colour_phase.as_payload(),
            "colour_target_correlation": round(self.colour_target_correlation, 4),
            "colour_prediction_correlation": round(self.colour_prediction_correlation, 4),
            "slices": [
                {
                    "colour": slice_.colour,
                    "n_positions": slice_.n_positions,
                    "n_games": slice_.n_games,
                    "mean_prediction": round(slice_.mean_prediction, 4),
                    "mean_target": round(slice_.mean_target, 4),
                    "value_mse": round(slice_.value_mse, 5),
                    "bias": round(slice_.bias, 4),
                }
                for slice_ in self.slices
            ],
            "phase_skill": [{"phase": phase, **result.as_payload()} for phase, result in self.phase_skill],
        }


def _safe_correlation(a: NDArray, b: NDArray) -> float:
    """Pearson correlation, returning nan for a degenerate input."""
    if a.size < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def compute_colour_value_diagnostic(
    predictions: NDArray,
    targets: NDArray,
    compact_boards: Sequence[NDArray] | NDArray,
    source_game_ids: NDArray,
    *,
    n_resamples: int = 2_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> ColourValueDiagnostic | None:
    """Compute the colour-conditional value diagnostic.

    Args:
        predictions: Value-head output per position.
        targets: Actual outcome per position, from the mover's perspective.
        compact_boards: Canonical compact boards, used to infer mover colour and
            game phase.
        source_game_ids: Which self-play game each position came from. Required —
            every interval is a cluster bootstrap over these, and computing one
            without them would produce intervals that are confidently wrong.
        n_resamples: Bootstrap resamples.
        seed: Resampling seed, so a generation's reported intervals reproduce.
        confidence: Nominal interval coverage.

    Returns:
        The diagnostic, or ``None`` when no position's mover colour could be
        inferred, or when only one colour is present (the colour-only baseline is
        then the global mean and the comparison is not the one being claimed).
    """
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    game_ids = np.asarray(source_game_ids).reshape(-1)
    boards = list(compact_boards)
    if not (len(predictions) == len(targets) == len(game_ids) == len(boards)):
        raise ValueError(
            "predictions, targets, compact_boards and source_game_ids must be the same length; "
            f"got {len(predictions)}, {len(targets)}, {len(boards)}, {len(game_ids)}"
        )

    colours = np.array([infer_mover_colour(board) for board in boards])
    phases = np.array([game_phase(board) for board in boards])
    keep = colours != COLOUR_UNKNOWN
    n_excluded = int((~keep).sum())
    if not keep.any():
        return None

    predictions, targets = predictions[keep], targets[keep]
    colours, phases, game_ids = colours[keep], phases[keep], game_ids[keep]
    if np.unique(colours).size < 2:
        return None

    colour_phase_keys = np.array([f"{c}:{p}" for c, p in zip(colours, phases, strict=True)])

    # Bootstrap the two skill numbers. The baseline is refit inside every
    # resample, because the baseline is part of the estimator being measured.
    #
    # Both bootstraps can legitimately fail to produce an interval: ``_skill``
    # returns nan when the baseline has zero error, and ``game_cluster_bootstrap``
    # raises once too few resamples are finite. A baseline has zero error whenever
    # every one of its groups is internally constant — which happens if the sample
    # is small enough that groups hold one position each (the colour x phase
    # baseline has many more groups, so it is the fragile one), or if the outcome
    # is perfectly predicted by mover colour. This is a **diagnostic**: an eval set
    # too degenerate to measure must return "no reading", never abort the training
    # run that called it.
    try:
        skill_vs_colour = game_cluster_bootstrap(
            _skill_statistic(predictions, targets, colours),
            game_ids,
            n_resamples=n_resamples,
            confidence=confidence,
            seed=seed,
        )
        skill_vs_colour_phase = game_cluster_bootstrap(
            _skill_statistic(predictions, targets, colour_phase_keys),
            game_ids,
            n_resamples=n_resamples,
            confidence=confidence,
            seed=seed + 1,
        )
    except ValueError as exc:
        logger.warning(
            "Colour-value diagnostic unavailable this generation ({} positions, {} games): {}. "
            "The baseline it measures against has no error to improve on, so skill is undefined. "
            "Training is unaffected; raise the eval-set size for a reading.",
            len(predictions),
            int(np.unique(game_ids).size),
            exc,
        )
        return None

    slices: list[ColourSlice] = []
    for colour in (WHITE_TO_MOVE, BLACK_TO_MOVE):
        mask = colours == colour
        if not mask.any():
            continue
        slices.append(
            ColourSlice(
                colour=colour,
                n_positions=int(mask.sum()),
                n_games=int(np.unique(game_ids[mask]).size),
                mean_prediction=float(predictions[mask].mean()),
                mean_target=float(targets[mask].mean()),
                value_mse=float(np.mean((predictions[mask] - targets[mask]) ** 2)),
            )
        )

    # Per-phase skill against the colour-only baseline, so a head that reads the
    # opening but not the endgame is visible rather than averaged away.
    phase_skill: list[tuple[str, BootstrapResult]] = []
    for offset, phase in enumerate((PHASE_EARLY, PHASE_MID, PHASE_LATE)):
        mask = phases == phase
        if mask.sum() < 2 or np.unique(game_ids[mask]).size < 2:
            continue
        try:
            result = game_cluster_bootstrap(
                _skill_statistic(predictions[mask], targets[mask], colours[mask]),
                game_ids[mask],
                n_resamples=n_resamples,
                confidence=confidence,
                seed=seed + 10 + offset,
            )
        except ValueError:
            # Too few usable resamples in this phase — report the phases that work
            # rather than failing the whole diagnostic.
            continue
        phase_skill.append((phase, result))

    return ColourValueDiagnostic(
        n_positions=int(keep.sum()),
        n_games=int(np.unique(game_ids).size),
        n_excluded=n_excluded,
        value_mse=float(np.mean((predictions - targets) ** 2)),
        colour_only_mse=float(np.mean((_group_mean_baseline(targets, colours) - targets) ** 2)),
        colour_phase_mse=float(np.mean((_group_mean_baseline(targets, colour_phase_keys) - targets) ** 2)),
        skill_vs_colour=skill_vs_colour,
        skill_vs_colour_phase=skill_vs_colour_phase,
        colour_target_correlation=_safe_correlation(colours.astype(np.float64), targets),
        colour_prediction_correlation=_safe_correlation(colours.astype(np.float64), predictions),
        slices=tuple(slices),
        phase_skill=tuple(phase_skill),
    )
