"""Tests for run provenance and the committed-config guard (A5).

The guard exists because one run's committed config was edited five days after
the run finished and now describes a run that never happened. The point is not to
detect that edit later — it is to make it impossible for the file on disk to
differ from the commit at launch, so the commit is a faithful record.

The git-backed tests build a throwaway repository in ``tmp_path`` rather than
inspecting this one, so they neither depend on nor disturb the working tree.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from alphablokus.config import RunConfig
from alphablokus.provenance import (
    PROVENANCE_FILENAME,
    build_provenance,
    check_config_is_committed,
    code_version,
    config_commit_state,
    sha256_file,
    write_provenance,
)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A throwaway git repository holding one committed config file."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    config = root / "run.json"
    config.write_text(json.dumps({"learning_rate": 0.00025}), encoding="utf-8")
    _git(root, "add", "run.json")
    _git(root, "commit", "-m", "add config")
    return root


@pytest.fixture
def config(tmp_path: Path, mcts_config, net_config) -> RunConfig:
    return RunConfig(
        game="tictactoe",
        run_name="provenance_test",
        num_generations=1,
        num_eps=1,
        temp_threshold=5,
        update_threshold=0.55,
        num_arena_matches=2,
        root_directory=tmp_path / "runs",
        load_model=False,
        mcts_config=mcts_config,
        net_config=net_config,
    )


# --- the guard ------------------------------------------------------------


def test_committed_config_is_allowed(repo: Path) -> None:
    state = check_config_is_committed(repo / "run.json")

    assert state.is_clean
    assert state.tracked
    assert state.differs is False
    assert state.commit


def test_modified_config_is_refused(repo: Path) -> None:
    path = repo / "run.json"
    path.write_text(json.dumps({"learning_rate": 0.001}), encoding="utf-8")

    with pytest.raises(SystemExit, match="uncommitted changes"):
        check_config_is_committed(path)


def test_untracked_config_is_refused(repo: Path) -> None:
    path = repo / "untracked.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit, match="not tracked"):
        check_config_is_committed(path)


def test_config_outside_a_repository_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "loose.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit, match="could not be checked"):
        check_config_is_committed(path)


def test_refusal_names_the_override(repo: Path) -> None:
    path = repo / "run.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit, match="--allow-uncommitted-config"):
        check_config_is_committed(path)


def test_override_permits_a_modified_config(repo: Path) -> None:
    """The guard must be escapable, or it becomes a guard people delete."""
    path = repo / "run.json"
    path.write_text(json.dumps({"learning_rate": 0.001}), encoding="utf-8")

    state = check_config_is_committed(path, allow_uncommitted=True)

    assert not state.is_clean
    assert state.differs is True


def test_override_permits_a_config_outside_a_repository(tmp_path: Path) -> None:
    path = tmp_path / "loose.json"
    path.write_text("{}", encoding="utf-8")

    state = check_config_is_committed(path, allow_uncommitted=True)

    assert not state.is_clean


def test_committing_the_change_clears_the_refusal(repo: Path) -> None:
    path = repo / "run.json"
    path.write_text(json.dumps({"learning_rate": 0.001}), encoding="utf-8")
    with pytest.raises(SystemExit):
        check_config_is_committed(path)

    _git(repo, "commit", "-am", "change the rate")

    assert check_config_is_committed(path).is_clean


def test_state_of_a_modified_config(repo: Path) -> None:
    path = repo / "run.json"
    path.write_text("{}", encoding="utf-8")

    state = config_commit_state(path)

    assert state.tracked is True
    assert state.differs is True
    assert state.is_clean is False


# --- the record -----------------------------------------------------------


def test_write_provenance_stamps_code_config_and_data(config: RunConfig, repo: Path) -> None:
    config.run_directory.mkdir(parents=True, exist_ok=True)
    config_path = repo / "run.json"
    state = config_commit_state(config_path)

    path = write_provenance(config, config_path=config_path, config_state=state)

    assert path is not None and path.name == PROVENANCE_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["run_name"] == "provenance_test"
    assert set(payload) >= {"code", "config_commit", "data_manifest", "config_path"}
    assert payload["config_commit"]["commit"] == state.commit
    assert payload["config_commit"]["differs_from_head"] is False
    assert payload["config_commit"]["override_used"] is False
    # The config file itself is always in the manifest, with a hash.
    manifest = {Path(entry["path"]).name: entry for entry in payload["data_manifest"]}
    assert "run.json" in manifest
    assert manifest["run.json"]["sha256"] == sha256_file(config_path)
    assert manifest["run.json"]["size_bytes"] > 0


def test_provenance_records_the_override(config: RunConfig, tmp_path: Path) -> None:
    config.run_directory.mkdir(parents=True, exist_ok=True)
    loose = tmp_path / "loose.json"
    loose.write_text("{}", encoding="utf-8")

    payload = build_provenance(
        config,
        config_path=loose,
        config_state=config_commit_state(loose),
        override_used=True,
    )

    assert payload["config_commit"]["override_used"] is True


def test_data_manifest_includes_the_eval_set_when_present(config: RunConfig) -> None:
    import numpy as np

    eval_dir = config.eval_set_directory
    eval_dir.mkdir(parents=True, exist_ok=True)
    np.save(eval_dir / "boards.npy", np.zeros((2, 2)))
    np.save(eval_dir / "source_game_ids.npy", np.arange(2))

    payload = build_provenance(config, config_path=None)

    names = {Path(entry["path"]).name for entry in payload["data_manifest"]}
    assert {"boards.npy", "source_game_ids.npy"} <= names


def test_provenance_is_json_serialisable(config: RunConfig) -> None:
    config.run_directory.mkdir(parents=True, exist_ok=True)

    json.dumps(build_provenance(config, config_path=None), default=str)


def test_code_version_reports_a_commit_for_this_repository() -> None:
    version = code_version()

    assert set(version) == {"commit", "branch", "dirty"}


def test_sha256_of_a_missing_file_is_none(tmp_path: Path) -> None:
    assert sha256_file(tmp_path / "nope.bin") is None


def test_write_provenance_survives_an_unwritable_directory(config: RunConfig) -> None:
    """Best-effort: a provenance failure must never sink an otherwise-fine launch."""
    assert write_provenance(config, config_path=None) is None
