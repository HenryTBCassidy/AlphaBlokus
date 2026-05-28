"""Post-training diagnostic: does the network play symmetrically?

A trained AlphaZero-style model is expected to produce policy distributions
that are *equivariant* under the game's symmetry group — if a board is
reflected (or rotated, depending on the game), the network's raw policy on
the transformed board should be the symmetric image of its policy on the
original. Deviation from this is a real bug shape: the asymmetric
preferences Henry surfaced on TTT (the "favourite corner" issue) are
exactly what this diagnostic catches.

The diagnostic is **game-agnostic**: it uses each game's existing
:meth:`IGame.get_symmetries` to enumerate the symmetric variants. For
Blokus Duo that's 2 (identity + main-diagonal reflection); for TTT it's 8
(the full D₄ group); for other games whatever they expose.

Per-position outputs: one KL divergence per non-identity symmetry, plus
the mean across them. A perfectly equivariant network scores 0. Larger
values indicate the network has internalised arbitrary biases that the
augmented training signal isn't fully averaging out.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from core.interfaces import IBoard, IGame, INeuralNetWrapper


@dataclass(frozen=True)
class SymmetryDiagnosticResult:
    """Outcome of running the symmetry diagnostic on a single position.

    ``kl_divergences`` has one entry per non-identity symmetry of the game
    — for Blokus Duo there's just one (the main-diagonal transpose); for
    TTT there are seven. ``mean_kl`` is the average across them, the
    headline summary statistic.
    """

    position_index: int
    kl_divergences: list[float]
    mean_kl: float
    top1_matches: list[bool]


def build_diagnostic_positions(
    game: IGame, *, n: int = 100, seed: int = 2026,
) -> list[IBoard]:
    """Generate ``n`` reproducible reference board positions for the
    symmetry diagnostic.

    These are the boards on which the network's asymmetric-preference is
    measured each generation — i.e. for each one we ask "does the net's
    raw policy look the same after we mirror the board?". The same
    ``seed`` always produces the same ``n`` positions, which is what
    makes the per-gen KL metric comparable across generations.

    **What's in the set:**

    - Index 0 is always the **empty starting board** (no moves played).
      Acts as a sanity check — the starting position is its own
      transpose, so a correctly-implemented diagnostic must score KL ≈ 0
      regardless of how the net behaves.
    - The remaining ``n - 1`` positions are reached by playing a random
      number of legal moves from the empty board, alternating players.

    **Distribution of game phase** is deliberately weighted toward
    early/mid-game positions because late-game Blokus often leaves only
    1–2 legal placements, so any reasonable net will look "asymmetric"
    on those purely by virtue of having one dominant move — that would
    skew the metric without telling us anything useful about learned
    bias. A small slice of late-game positions is still included to
    confirm the net still chooses the strong move when one stands out.

    Concretely, of the ``n - 1`` non-start positions:

    - ~80% have rollout length sampled uniformly from ``[3, 22]`` —
      opening through mid-game, where strategy is most interesting and
      the legal-move set is broad enough that genuine policy asymmetry
      is detectable.
    - ~20% have rollout length sampled uniformly from ``[23, 32]`` —
      late-game, where the board is more constrained.

    Args:
        game: any :class:`IGame` implementation; method only needs
            ``initialise_board``, ``valid_move_masking``, and
            ``get_next_state``.
        n: number of reference positions to return. Default 100.
        seed: deterministic seed for both the rollout-length sampler and
            the per-rollout move sampler.
    """
    if n <= 0:
        return []

    rng = np.random.default_rng(seed)
    positions: list[IBoard] = [game.initialise_board()]

    remaining = n - 1
    n_early_mid = round(remaining * 0.8)
    n_late = remaining - n_early_mid
    rollout_lengths = list(rng.integers(3, 23, size=n_early_mid)) + list(
        rng.integers(23, 33, size=n_late),
    )

    for length in rollout_lengths:
        board = game.initialise_board()
        player = 1
        for _ in range(int(length)):
            mask = game.valid_move_masking(board, player)
            legal = np.flatnonzero(mask)
            if len(legal) == 0:
                break
            action = int(rng.choice(legal))
            board, player = game.get_next_state(board, player, action)
        positions.append(board)
    return positions


def compute_symmetry_diagnostic(
    nnet: INeuralNetWrapper, game: IGame, board: IBoard, position_index: int = 0,
) -> SymmetryDiagnosticResult:
    """Measure how equivariant ``nnet`` is on ``board``.

    For each non-identity symmetry ``s`` of the game:

    1. ``actual_pi``: ``nnet.predict(s(board))`` — the network's raw policy
       on the symmetric variant in *that variant's* coordinate frame.
    2. ``expected_pi``: ``s(nnet.predict(board))`` — what the policy
       *should* be if the network were perfectly equivariant. Produced
       directly by ``game.get_symmetries`` (which already knows how to
       transform a policy under each symmetry).
    3. KL divergence between the two.

    Returns one KL per non-identity symmetry plus their mean.
    """
    canonical = game.get_canonical_form(board, 1)
    raw_pi_array, _ = nnet.predict(canonical)
    raw_pi = np.asarray(raw_pi_array, dtype=np.float64)

    symmetries = game.get_symmetries(canonical, raw_pi)
    kls: list[float] = []
    top1_matches: list[bool] = []
    # Skip the identity (first entry) — by construction it has KL 0 and
    # would dilute the signal.
    for variant_board, expected_pi in symmetries[1:]:
        actual_pi_array, _ = nnet.predict(variant_board)
        actual_pi = np.asarray(actual_pi_array, dtype=np.float64)
        expected_pi_array = np.asarray(expected_pi, dtype=np.float64)
        kls.append(_kl(actual_pi, expected_pi_array))
        top1_matches.append(int(actual_pi.argmax()) == int(expected_pi_array.argmax()))

    mean_kl = float(np.mean(kls)) if kls else 0.0
    return SymmetryDiagnosticResult(
        position_index=position_index,
        kl_divergences=kls,
        mean_kl=mean_kl,
        top1_matches=top1_matches,
    )


def _kl(p: NDArray, q: NDArray, *, eps: float = 1e-12) -> float:
    """KL(p || q) over discrete distributions, with floor `eps` to keep the
    log finite when the network assigns zero probability to a legal action.
    """
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * (np.log(p) - np.log(q))))
