"""Run provenance: what code, what config, what data.

Reconstructing what a run actually did has been the single most expensive
recurring cost in this project's post-mortems. The committed config is not a
reliable record — one run's committed JSON was edited five days *after* the run
finished, and now describes a run that never happened: read it today and you
would conclude that run had weight decay. It did not.

Two mechanisms here, and they do different jobs:

- :func:`write_provenance` stamps the resolved config, the code version and a
  manifest of the input data into the run directory. This is the *record*.
- :func:`check_config_is_committed` refuses to start a run whose config file
  differs from the committed version in git. This is the *guard*, and it is what
  makes the record trustworthy: if the file on disk is always the committed file,
  then the commit is a faithful description of what ran, and a later edit shows
  up as a new commit rather than silently rewriting history.

The guard is overridable with an explicit flag (``--allow-uncommitted-config``),
because refusing outright would make it impossible to iterate on a config, and a
guard nobody can get past is a guard people delete.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from alphablokus.config import RunConfig

PROVENANCE_FILENAME = "run_provenance.json"

# Read in chunks so a multi-hundred-MB checkpoint doesn't land in RAM whole.
_HASH_CHUNK_BYTES = 1 << 20


def _git(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str] | None:
    """Run a git command, or return None when git cannot be run at all.

    Everything here is best-effort: the code may be running from a source tarball,
    an installed wheel, or a machine without git. A missing answer degrades the
    record; it must never crash a launch.
    """
    try:
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _run_git(args: list[str], *, cwd: Path) -> str | None:
    """Stripped stdout of a successful git command, else None.

    Only for commands whose *output* is the answer. Commands whose answer is the
    exit status (``diff --quiet``) must use :func:`_git_succeeds` — they succeed
    with empty output, which this function cannot distinguish from failure.
    """
    completed = _git(args, cwd=cwd)
    if completed is None or completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _git_succeeds(args: list[str], *, cwd: Path) -> bool | None:
    """Whether a git command exited 0, or None when git could not be run."""
    completed = _git(args, cwd=cwd)
    if completed is None:
        return None
    return completed.returncode == 0


def repo_root(start: Path | None = None) -> Path:
    """Best guess at the repository root containing the package source."""
    base = start or Path(__file__).resolve().parent
    found = _run_git(["rev-parse", "--show-toplevel"], cwd=base)
    return Path(found) if found else base


def code_version(root: Path | None = None) -> dict[str, object]:
    """Describe the code the run is about to execute.

    Returns:
        ``commit`` (full sha or None), ``dirty`` (whether the working tree has
        uncommitted changes to tracked files), and ``branch``. A dirty tree is
        recorded rather than refused — the config guard is the hard stop, because
        an uncommitted *config* silently changes the experiment whereas an
        uncommitted code change is usually the thing being tested.
    """
    base = root or repo_root()
    commit = _run_git(["rev-parse", "HEAD"], cwd=base)
    status = _run_git(["status", "--porcelain", "--untracked-files=no"], cwd=base)
    return {
        "commit": commit,
        "branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=base),
        "dirty": bool(status) if status is not None else None,
    }


def sha256_file(path: Path) -> str | None:
    """SHA-256 of a file's contents, or None when it can't be read."""
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(_HASH_CHUNK_BYTES):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _describe_file(path: Path) -> dict[str, object]:
    """One data-manifest entry."""
    try:
        size = path.stat().st_size
    except OSError:
        size = -1
    return {"path": str(path), "sha256": sha256_file(path), "size_bytes": size}


def data_manifest(config: RunConfig, config_path: Path | None) -> list[dict[str, object]]:
    """Hash the inputs that determine the run's starting state.

    Covers the config file itself, the donor checkpoint on a warm start (the
    single most consequential input — three experiments warm-started from a
    checkpoint nobody had verified was the best of its run), and any eval set
    already on disk. Files that don't exist are simply absent from the manifest.
    """
    candidates: list[Path] = []
    if config_path is not None:
        candidates.append(config_path)
    if config.load_model:
        candidates.append(config.net_directory / "best.pth.tar")
    # Every persisted eval-set component, not a subset: the policy targets and
    # metadata change what the diagnostics report, and source_fingerprints.json
    # decides which replay games are excluded from training. Hashing only some
    # of them lets two materially different runs share an eval-set manifest.
    eval_dir = config.eval_set_directory
    candidates.extend(
        [
            eval_dir / "boards.npy",
            eval_dir / "compact_boards.npy",
            eval_dir / "target_policies.npy",
            eval_dir / "target_values.npy",
            eval_dir / "targets_kind.txt",
            eval_dir / "source_game_ids.npy",
            eval_dir / "source_fingerprints.json",
            eval_dir / "metadata.json",
        ]
    )
    return [_describe_file(path) for path in candidates if path.exists()]


@dataclass(frozen=True)
class ConfigCommitState:
    """Whether the config file on disk matches its committed version.

    Attributes:
        tracked: True when git knows about the file. False for an untracked
            config — which is *not* the same as a modified one, and is treated as
            a violation too: a run whose config exists only on one machine cannot
            be reproduced from the repository.
        differs: True when the working-tree file differs from HEAD. None when git
            could not be consulted at all.
        commit: HEAD at the time of the check.
    """

    tracked: bool
    differs: bool | None
    commit: str | None

    @property
    def is_clean(self) -> bool:
        """True when the file is tracked and byte-identical to HEAD."""
        return self.tracked and self.differs is False


def config_commit_state(config_path: Path) -> ConfigCommitState:
    """Compare a config file against its committed version."""
    path = config_path.resolve()
    root = repo_root(path.parent)
    commit = _run_git(["rev-parse", "HEAD"], cwd=root)
    if commit is None:
        # No git available / not a repository: we cannot say anything.
        return ConfigCommitState(tracked=False, differs=None, commit=None)
    if not _git_succeeds(["ls-files", "--error-unmatch", str(path)], cwd=root):
        return ConfigCommitState(tracked=False, differs=None, commit=commit)
    # ``diff --quiet`` exits 0 when the file matches HEAD and 1 when it differs,
    # with no output either way — so the exit status is the answer.
    unchanged = _git_succeeds(["diff", "--quiet", "HEAD", "--", str(path)], cwd=root)
    if unchanged is None:
        return ConfigCommitState(tracked=True, differs=None, commit=commit)
    return ConfigCommitState(tracked=True, differs=not unchanged, commit=commit)


def check_config_is_committed(config_path: Path, *, allow_uncommitted: bool = False) -> ConfigCommitState:
    """Refuse to start a run whose config differs from the committed version.

    Args:
        config_path: The config file the run was launched with.
        allow_uncommitted: Explicit override. Logs loudly and proceeds — used
            when iterating on a config, or when running from a checkout without
            git.

    Returns:
        The state that was checked, so the caller can stamp it into the record.

    Raises:
        SystemExit: When the config is modified or untracked and the override was
            not given. ``SystemExit`` rather than ``ValueError`` to match the
            CLI's existing pre-flight refusals (see the ``--resume`` guard).
    """
    state = config_commit_state(config_path)
    if state.is_clean:
        logger.info("Config {} matches HEAD ({}).", config_path, (state.commit or "?")[:12])
        return state

    if state.differs is None and not state.tracked and state.commit is None:
        reason = f"{config_path} could not be checked against git (no repository or git unavailable)"
    elif not state.tracked:
        reason = f"{config_path} is not tracked by git, so this run could not be reproduced from the repository"
    else:
        reason = f"{config_path} has uncommitted changes, so the committed config does not describe this run"

    if allow_uncommitted:
        logger.warning(
            "{} — proceeding because --allow-uncommitted-config was given. The run's provenance record will say so.",
            reason,
        )
        return state

    raise SystemExit(
        f"Refusing to start: {reason}.\n"
        "Commit the config first — a run whose config is edited afterwards leaves a record that "
        "describes a run that never happened (this has already happened once, and cost a "
        "post-mortem).\n"
        "To run anyway, pass --allow-uncommitted-config."
    )


def build_provenance(
    config: RunConfig,
    *,
    config_path: Path | None,
    config_state: ConfigCommitState | None = None,
    override_used: bool = False,
) -> dict[str, object]:
    """Assemble the provenance record for a run."""
    return {
        "run_name": config.run_name,
        "config_path": str(config_path) if config_path else None,
        "code": code_version(),
        "config_commit": {
            "tracked": config_state.tracked if config_state else None,
            "differs_from_head": config_state.differs if config_state else None,
            "commit": config_state.commit if config_state else None,
            "override_used": override_used,
        },
        "data_manifest": data_manifest(config, config_path),
    }


def write_provenance(
    config: RunConfig,
    *,
    config_path: Path | None,
    config_state: ConfigCommitState | None = None,
    override_used: bool = False,
) -> Path | None:
    """Stamp the provenance record into the run directory.

    Best-effort, like ``persist_resolved_config``: a hashing or serialisation
    failure must not sink a launch that is otherwise fine.

    Returns:
        The path written, or None on failure.
    """
    path = config.run_directory / PROVENANCE_FILENAME
    try:
        payload = build_provenance(
            config,
            config_path=config_path,
            config_state=config_state,
            override_used=override_used,
        )
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        logger.info("Wrote run provenance to {}", path)
        return path
    except Exception as err:  # pragma: no cover - defensive; never blocks a run
        logger.warning("Could not write provenance to {} ({}); continuing.", path, err)
        return None
