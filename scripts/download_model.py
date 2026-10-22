"""Download the published best network so you can play or benchmark it locally.

Fetches the checkpoint attached to the GitHub release, checks its SHA-256, and puts it in the
net directory of ``run_configurations/best_net_v3_gen40.json`` (``models/runs/blokus/v3_gen40/Nets/``),
where ``alphablokus-play``, ``export_web_assets.py`` and ``pentobi_benchmark.py`` expect it.

Usage:
    uv run python scripts/download_model.py
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

from loguru import logger

from alphablokus.config import load_args

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "run_configurations" / "best_net_v3_gen40.json"
CHECKPOINT_NAME = "v3_gen40.pth.tar"
RELEASE_URL = f"https://github.com/HenryTBCassidy/AlphaBlokus/releases/download/v3-gen40/{CHECKPOINT_NAME}"
SHA256 = "4c918e37079d824dce3832f371e5016c5da4e2a65d6409afc94ba973b1994adb"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_path() -> Path:
    """Where the checkpoint lives, resolved from the release config the same way the tools resolve it."""
    config = load_args(CONFIG_PATH)
    net_dir = config.net_directory
    if not net_dir.is_absolute():
        net_dir = REPO_ROOT / net_dir
    return net_dir / CHECKPOINT_NAME


def download(dest: Path, *, force: bool = False) -> Path:
    """Download the checkpoint to ``dest`` unless a verified copy is already there.

    Args:
        dest: Target file path.
        force: Re-download even if a verified copy exists.

    Returns:
        The path of the verified checkpoint.

    Raises:
        RuntimeError: If the downloaded file's hash doesn't match the published one.
    """
    if dest.exists() and not force:
        if _sha256(dest) == SHA256:
            logger.info("Already have a verified copy at {}", dest)
            return dest
        logger.warning("Existing file at {} doesn't match the published hash, downloading again", dest)

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    logger.info("Downloading {} (~98 MB)", RELEASE_URL)
    with urllib.request.urlopen(RELEASE_URL) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)

    actual = _sha256(partial)
    if actual != SHA256:
        partial.unlink()
        raise RuntimeError(f"SHA-256 mismatch: expected {SHA256}, got {actual}")
    partial.replace(dest)
    logger.info("Saved and verified {}", dest)
    return dest


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="Download again even if a verified copy exists.")
    args = parser.parse_args()
    download(checkpoint_path(), force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
