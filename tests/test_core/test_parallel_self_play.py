"""Determinism + correctness tests for the F1 parallel self-play worker.

The headline invariants this file enforces:

1. ``derive_episode_seed`` is a pure function: same input → same output,
   different inputs → (statistically) different outputs.
2. ``run_self_play_episodes_parallel`` with ``num_workers=1`` produces
   the same per-episode training examples as with ``num_workers=4`` at
   the same seed. This is the F1 P4 determinism check — proves the
   worker pool introduces no statistical drift relative to a serial
   walk through the same tasks.
3. The function returns one episode-worth of examples per
   ``range(num_eps)`` slot, in submission order.

Tests use TicTacToe + a small MCTS sim count so the whole file runs in
a few seconds even with multiple pool spawns.
"""
from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from core.config import MCTSConfig, NetConfig, RunConfig
from core.game_factory import instantiate_game_and_network
from core.parallel_self_play import (
    PHASE_ARENA,
    derive_episode_seed,
    run_self_play_episodes_parallel,
    run_two_player_games_parallel,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def parallel_test_config(tmp_path: Path) -> RunConfig:
    """RunConfig sized so the determinism test completes in a few
    seconds even with two pool spawns. The TTT 32f×1b net is fast on
    CPU; 25 MCTS sims is enough to exercise the search loop without
    making each episode expensive.
    """
    return RunConfig(
        game="tictactoe",
        run_name="parallel_self_play_test",
        seed=42,
        num_generations=1,
        num_eps=4,
        temp_threshold=3,
        update_threshold=0.55,
        max_queue_length=100,
        num_arena_matches=2,
        max_generations_lookback=1,
        root_directory=tmp_path,
        load_model=False,
        mcts_config=MCTSConfig(num_mcts_sims=25, cpuct=1.0),
        net_config=NetConfig(
            learning_rate=0.001,
            dropout=0.3,
            epochs=1,
            batch_size=4,
            cuda=False,
            num_filters=32,
            num_residual_blocks=1,
        ),
    )


class TestDeriveEpisodeSeed:
    """``derive_episode_seed`` is a pure function — no globals, no
    randomness. These tests pin that contract.
    """

    def test_same_input_same_output(self) -> None:
        assert derive_episode_seed(42, 1, 3) == derive_episode_seed(42, 1, 3)

    def test_different_episode_changes_seed(self) -> None:
        s_a = derive_episode_seed(42, 1, 3)
        s_b = derive_episode_seed(42, 1, 4)
        assert s_a != s_b

    def test_different_generation_changes_seed(self) -> None:
        s_a = derive_episode_seed(42, 1, 3)
        s_b = derive_episode_seed(42, 2, 3)
        assert s_a != s_b

    def test_different_base_seed_changes_seed(self) -> None:
        s_a = derive_episode_seed(42, 1, 3)
        s_b = derive_episode_seed(43, 1, 3)
        assert s_a != s_b

    def test_returns_python_int(self) -> None:
        # ``torch.manual_seed`` rejects numpy scalar types; we coerce
        # the SeedSequence output to ``int`` so workers can pass it
        # straight through.
        seed = derive_episode_seed(42, 1, 3)
        assert isinstance(seed, int)


def _normalise_examples_for_comparison(per_episode_examples):
    """Strip per-example floats down to a comparable shape.

    Each example is ``(board_ndarray, policy_ndarray, value_float)``.
    We hash each array's bytes (after coercing to a fixed dtype) and
    keep the value as a Python float — so the per-episode list becomes
    a tuple of ``(board_hash, policy_hash, value)`` triples that
    compares cleanly with ``==``.
    """
    import numpy as np

    normalised = []
    for episode in per_episode_examples:
        triples = []
        for board, policy, value in episode:
            board_bytes = np.asarray(board, dtype=np.float32).tobytes()
            policy_bytes = np.asarray(policy, dtype=np.float32).tobytes()
            triples.append((hash(board_bytes), hash(policy_bytes), float(value)))
        normalised.append(tuple(triples))
    return normalised


def _save_fixed_checkpoint(config: RunConfig, filename: str) -> str:
    """Build a net under the config, save its random-init weights to
    ``filename`` (relative to ``config.net_directory``), return the
    filename. Both pool spawns load this same checkpoint so worker
    networks are bit-identical regardless of process-local seeding.
    """
    _, nnet = instantiate_game_and_network(config)
    nnet.save_checkpoint(filename=filename)
    return filename


@pytest.mark.slow
def test_parallel_one_worker_matches_four_workers(
    parallel_test_config: RunConfig,
) -> None:
    """The headline determinism test.

    Running the same episodes through a 1-worker pool vs a 4-worker
    pool must produce the **same training examples in the same
    per-episode order**. Episode order is preserved because the
    orchestrator uses ``pool.map`` (not ``imap_unordered``); per-episode
    determinism is guaranteed by ``derive_episode_seed``.

    If this test fails, any speedup claim from F1 is suspect — we'd be
    getting "free" wall-clock at the cost of running different (worse)
    training data.

    Workers must load from the same checkpoint or their random-init
    weights diverge across process boundaries — in production
    ``Coach._run_self_play_parallel`` always passes a checkpoint, so
    this matches real usage.
    """
    checkpoint = _save_fixed_checkpoint(parallel_test_config, "test_init.pth.tar")

    examples_serial, _ = run_self_play_episodes_parallel(
        config=parallel_test_config,
        generation=1,
        checkpoint_path=checkpoint,
        num_workers=1,
    )
    examples_parallel, _ = run_self_play_episodes_parallel(
        config=replace(parallel_test_config, num_parallel_workers=4),
        generation=1,
        checkpoint_path=checkpoint,
        num_workers=4,
    )

    norm_serial = _normalise_examples_for_comparison(examples_serial)
    norm_parallel = _normalise_examples_for_comparison(examples_parallel)

    assert len(norm_serial) == parallel_test_config.num_eps
    assert len(norm_parallel) == parallel_test_config.num_eps
    assert norm_serial == norm_parallel, (
        "Parallel run with 4 workers produced different training data than "
        "the same run with 1 worker — determinism is broken."
    )


@pytest.mark.slow
def test_two_player_one_worker_matches_four_workers(
    parallel_test_config: RunConfig,
) -> None:
    """Determinism test for the two-net orchestrator (arena + Elo).

    Same idea as the self-play test: same checkpoints + same seed +
    different worker counts must yield identical outcome counts and
    identical per-game records. Game records compare exactly because
    ``Arena.play_game`` is deterministic at temp=0 once RNGs are
    pinned, and our worker seeds the RNGs from
    ``(base, gen, ep, phase)``.
    """
    checkpoint_a = _save_fixed_checkpoint(parallel_test_config, "two_player_a.pth.tar")
    checkpoint_b = _save_fixed_checkpoint(parallel_test_config, "two_player_b.pth.tar")

    a1, b1, d1, r1 = run_two_player_games_parallel(
        config=parallel_test_config,
        generation=1,
        checkpoint_a_path=checkpoint_a,
        checkpoint_b_path=checkpoint_b,
        num_games=4,
        num_workers=1,
        phase=PHASE_ARENA,
        record=True,
        top_k=3,
        desc="Arena (test)",
    )
    a4, b4, d4, r4 = run_two_player_games_parallel(
        config=replace(parallel_test_config, num_parallel_workers=4),
        generation=1,
        checkpoint_a_path=checkpoint_a,
        checkpoint_b_path=checkpoint_b,
        num_games=4,
        num_workers=4,
        phase=PHASE_ARENA,
        record=True,
        top_k=3,
        desc="Arena (test)",
    )

    assert (a1, b1, d1) == (a4, b4, d4), (
        f"Outcome counts diverged between 1-worker and 4-worker arena "
        f"runs: 1-worker={(a1, b1, d1)}, 4-worker={(a4, b4, d4)}."
    )
    assert len(r1) == len(r4) == 4
    # Move sequences should match game-for-game when seeds match.
    for game_idx, (rec1, rec4) in enumerate(zip(r1, r4, strict=False)):
        assert rec1.outcome == rec4.outcome, (
            f"Game {game_idx} outcome differs: {rec1.outcome} vs {rec4.outcome}"
        )
        assert rec1.player1_was_white == rec4.player1_was_white
        actions_1 = tuple(m.action for m in rec1.moves)
        actions_4 = tuple(m.action for m in rec4.moves)
        assert actions_1 == actions_4, (
            f"Game {game_idx} move sequence differs between worker counts"
        )


@pytest.mark.slow
def test_episode_count_matches_config(
    parallel_test_config: RunConfig,
) -> None:
    """The orchestrator returns one entry per ``range(num_eps)``."""
    checkpoint = _save_fixed_checkpoint(parallel_test_config, "test_init.pth.tar")
    examples, stats = run_self_play_episodes_parallel(
        config=parallel_test_config,
        generation=1,
        checkpoint_path=checkpoint,
        num_workers=2,
    )
    assert len(examples) == parallel_test_config.num_eps
    assert len(stats) == parallel_test_config.num_eps
    # Every episode should produce at least one example (the game has
    # to run for at least one move before ending).
    for ep in examples:
        assert len(ep) >= 1
    # Every episode should have non-trivial MCTS stats.
    for s in stats:
        assert s.num_moves >= 1
        assert s.total_sims >= 1
