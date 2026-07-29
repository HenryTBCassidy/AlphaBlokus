"""Turnkey wrap-up for a finished Pentobi distillation corpus.

Runs the four completion steps as one command, with a **verify-before-done**
gate so a corpus is never declared safe until it is confirmed present in
durable object storage (data-safety: local box disk + the laptop mirror are not
"safe" until R2 has it too):

1. **validate** every stored row on the box (replay-check legality/labels).
2. **analyze** the corpus on the box (diversity + zero-duplicate-openings).
3. **fetch** the shards down to the laptop mirror (rsync; the box can't push).
4. **sync to R2** and verify every local file is present remotely with a
   matching size before printing the all-clear.

Credentials come from the environment (the secrets convention) — source them
first, and on the Mac export the corp CA bundle for R2's TLS:

    set -a; source local/secrets.env; set +a
    export AWS_CA_BUNDLE="$HOME/.corp-ca-bundle.pem"
    uv run --extra s3 python scripts/corpus_wrapup.py \
        --config run_configurations/blokus_cloud_v2.json

The R2 bucket/endpoint are read from the config's ``object_store`` block; only
the keys come from the environment.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from loguru import logger

from alphablokus.storage.object_store import ObjectStore


class CorpusWrapupError(RuntimeError):
    """A wrap-up step failed; the corpus is NOT confirmed safe."""


def _run_remote(host: str, command: str) -> None:
    """Run a shell command on the box over SSH; raise on non-zero exit."""
    result = subprocess.run(  # noqa: S603 — fixed argv, host/command are operator-supplied
        ["ssh", "-o", "ConnectTimeout=20", host, command],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise CorpusWrapupError(f"remote command failed on {host} (exit {result.returncode}):\n{result.stderr.strip()}")
    if result.stdout.strip():
        logger.info("[{}] {}", host, result.stdout.strip().splitlines()[-1])


def _is_v2_corpus(host: str, remote_dir: str) -> bool:
    """A v2 corpus is the one with a store; v1 is bare shards in a directory."""
    result = subprocess.run(  # noqa: S603 — fixed argv, host/dir are operator-supplied
        ["ssh", "-o", "ConnectTimeout=20", host, f"test -f {remote_dir}/store.sqlite && echo v2 || echo v1"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() == "v2"


def validate_and_analyze_remote(host: str, remote_repo: str, remote_dir: str) -> None:
    """Replay-validate and analyze the corpus in place on the box.

    The two generators have separate scripts and incompatible layouts — v1 keeps bare
    ``corpus_*.parquet`` in one directory, v2 keeps ``games/``, ``opening/`` and
    ``store.sqlite`` — so the right one has to be chosen. Pointing v1's validator at a v2
    corpus is the dangerous case rather than a loud one: it finds no shards at the top
    level, prints "No corpus shards found" and **exits 0**, so the gate that is supposed
    to prove a corpus correct passes having checked nothing at all.
    """
    base = f"export PATH=$HOME/.local/bin:$PATH && cd {remote_repo}"
    script = "scripts/pentobi_corpus_v2.py" if _is_v2_corpus(host, remote_dir) else "scripts/pentobi_corpus.py"
    flag = "--corpus" if script.endswith("_v2.py") else "--data"
    logger.info("Detected {} corpus; validating every row on {} ...", "v2" if "_v2" in script else "v1", host)
    _run_remote(host, f"{base} && uv run python -m {script[:-3].replace('/', '.')} validate {flag} {remote_dir}")
    logger.info("Analyzing corpus on {} ...", host)
    _run_remote(
        host,
        f"{base} && uv run python -m {script[:-3].replace('/', '.')} analyze {flag} {remote_dir} "
        f"--json {remote_dir}/diversity.json",
    )


def fetch_corpus(host: str, remote_dir: str, local_dir: Path) -> None:
    """rsync the corpus shards down to the laptop mirror (laptop pulls)."""
    local_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Fetching corpus {}:{} -> {} ...", host, remote_dir, local_dir)
    result = subprocess.run(  # noqa: S603 — fixed argv
        ["rsync", "-az", "--exclude", "*.tmp", f"{host}:{remote_dir}/", f"{local_dir}/"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise CorpusWrapupError(f"rsync failed (exit {result.returncode}):\n{result.stderr.strip()}")


def push_to_r2_verified(store: ObjectStore, local_dir: Path) -> tuple[int, int]:
    """Upload the corpus to R2, then verify every local file is present remotely.

    Returns ``(files_uploaded, files_verified_remote)``. Raises
    ``CorpusWrapupError`` if any local file is missing remotely or differs in
    size — the corpus is only "safe" once this returns cleanly.
    """
    local_files = {
        p.relative_to(local_dir).as_posix(): p.stat().st_size
        for p in local_dir.rglob("*")
        if p.is_file() and p.suffix != ".tmp"
    }
    if not local_files:
        raise CorpusWrapupError(f"no files to sync under {local_dir}")

    uploaded = store.sync_up(local_dir)
    remote = dict(store.remote_files())

    missing = [rel for rel, size in local_files.items() if remote.get(rel) != size]
    if missing:
        preview = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
        raise CorpusWrapupError(
            f"R2 verify FAILED — {len(missing)} local file(s) missing/mismatched remotely: {preview}"
        )
    return uploaded, len(local_files)


def _object_store_from_config(config_path: Path, prefix: str) -> ObjectStore:
    """Build an ObjectStore from a run config's ``object_store`` block + a prefix."""
    block = json.loads(config_path.read_text()).get("object_store")
    if not block or "bucket" not in block:
        raise CorpusWrapupError(f"{config_path} has no usable object_store block (need at least a bucket).")
    return ObjectStore(
        bucket=block["bucket"],
        prefix=prefix,
        endpoint_url=block.get("endpoint_url"),
        region=block.get("region"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="gpu-linux", help="SSH host of the box (default gpu-linux)")
    parser.add_argument("--remote-repo", default="~/AlphaBlokus", help="Repo path on the box")
    parser.add_argument("--remote-dir", default="~/corpora/pentobi_l9_stage1", help="Corpus dir on the box")
    parser.add_argument("--local-dir", type=Path, default=Path("temp/corpora/pentobi_l9_stage1"), help="Laptop mirror")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("run_configurations/blokus_cloud_v2.json"),
        help="Run config whose object_store block gives the R2 bucket/endpoint",
    )
    parser.add_argument("--prefix", default=None, help="R2 key prefix (default corpora/<corpus dir name>)")
    parser.add_argument("--skip-validate", action="store_true", help="Skip the on-box validate + analyze")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip the rsync-down (use the existing local mirror)")
    parser.add_argument("--skip-r2", action="store_true", help="Skip the R2 sync (NOT data-safe — for dry runs only)")
    args = parser.parse_args()

    prefix = args.prefix or f"corpora/{args.remote_dir.rstrip('/').split('/')[-1]}"

    if not args.skip_validate:
        validate_and_analyze_remote(args.host, args.remote_repo, args.remote_dir)
    if not args.skip_fetch:
        fetch_corpus(args.host, args.remote_dir, args.local_dir)

    if args.skip_r2:
        logger.warning("Skipping R2 sync — corpus is NOT confirmed in durable storage.")
        return

    store = _object_store_from_config(args.config, prefix)
    uploaded, verified = push_to_r2_verified(store, args.local_dir)
    logger.success(
        "CORPUS SAFE IN R2 — bucket prefix {!r}: {} file(s) verified ({} newly uploaded).",
        prefix,
        verified,
        uploaded,
    )


if __name__ == "__main__":
    try:
        main()
    except CorpusWrapupError as err:
        logger.error("Wrap-up aborted: {}", err)
        sys.exit(1)
