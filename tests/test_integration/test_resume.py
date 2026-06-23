"""Resume + crash-safe-report behaviour (docs/plans/resumable-runs.md, R8).

A run must be continuable in place: after it stops, ``--resume`` picks up from the
last *completed* generation, keeps the original frozen Elo baseline, and never
overwrites the generations already on disk.
"""
import json
from dataclasses import replace

import pytest

from core.coach import PROGRESS_MARKER_FILENAME, Coach, read_progress_marker
from core.config import RunConfig
from games.tictactoe.game import TicTacToeGame
from games.tictactoe.neuralnets.wrapper import NNetWrapper


def test_read_progress_marker_roundtrip(test_config: RunConfig) -> None:
    """`read_progress_marker` returns None when absent and the payload when present."""
    assert read_progress_marker(test_config) is None

    test_config.log_directory.mkdir(parents=True, exist_ok=True)
    payload = {"last_completed_generation": 7, "wandb_run_id": "abc123"}
    (test_config.log_directory / PROGRESS_MARKER_FILENAME).write_text(json.dumps(payload))

    assert read_progress_marker(test_config) == payload


@pytest.mark.slow
def test_resume_continues_without_clobber(
    ttt_game: TicTacToeGame, test_config: RunConfig,
) -> None:
    """A 2-generation run, then `--resume` to 4 generations:

    - generation numbering continues at 3 (not restart at 1);
    - the frozen Elo baseline is byte-identical across the resume;
    - generations 1-2 on disk are untouched, 3-4 are appended;
    - the progress marker advances to the last completed generation.
    """
    # elo_games_per_gen > 0 so the baseline checkpoint is created (and must be
    # preserved on resume); lookback 2 so the resume history reload is exercised.
    cfg = replace(
        test_config, num_generations=2, elo_games_per_gen=2, max_generations_lookback=2,
    )

    # --- initial run: 2 generations ---------------------------------------
    coach = Coach(ttt_game, NNetWrapper(ttt_game, cfg), cfg)
    coach.learn()

    marker = read_progress_marker(cfg)
    assert marker is not None and marker["last_completed_generation"] == 2

    training_dir = cfg.training_data_directory
    gens_after_initial = {p.name for p in training_dir.glob("generation=*")}
    assert gens_after_initial == {"generation=1", "generation=2"}

    baseline_path = cfg.net_directory / "elo_baseline.pth.tar"
    assert baseline_path.exists()
    baseline_bytes = baseline_path.read_bytes()

    # --- resume: extend to 4 generations ----------------------------------
    cfg_resume = replace(cfg, num_generations=4)
    last_gen = read_progress_marker(cfg_resume)["last_completed_generation"]
    nnet = NNetWrapper(ttt_game, cfg_resume)
    nnet.load_checkpoint("latest.pth.tar")
    coach2 = Coach(ttt_game, nnet, cfg_resume, resume=True, resume_wandb_run_id=None)
    coach2.load_self_play_history_for_resume(last_gen)
    coach2.learn(start_generation=last_gen + 1)

    # Numbering continued: 3 and 4 appended, 1 and 2 still present.
    gens_after_resume = {p.name for p in training_dir.glob("generation=*")}
    assert gens_after_resume == {f"generation={g}" for g in (1, 2, 3, 4)}

    # Baseline preserved exactly — Elo stays comparable across the resume.
    assert baseline_path.read_bytes() == baseline_bytes

    # Marker advanced to the new last completed generation.
    assert read_progress_marker(cfg_resume)["last_completed_generation"] == 4
