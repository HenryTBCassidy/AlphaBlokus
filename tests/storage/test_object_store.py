"""Object-store sync + remote resume (storage/object_store.py, cli restore).

Uses an in-memory fake with boto3's client surface — no network, no boto3
dependency — so these run in every CI job.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from alphablokus.cli import restore_run_from_object_store
from alphablokus.config import ObjectStoreConfig
from alphablokus.storage.object_store import ObjectStore, create_object_store, sync_up_guarded
from alphablokus.training.coach import PROGRESS_MARKER_FILENAME

if TYPE_CHECKING:
    from alphablokus.config import RunConfig


class FakeS3Client:
    """In-memory stand-in for the slice of boto3's S3 client that we use."""

    def __init__(self, page_size: int = 1000) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.page_size = page_size

    def upload_file(self, filename: str, bucket: str, key: str) -> None:
        self.objects[(bucket, key)] = Path(filename).read_bytes()

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        Path(filename).write_bytes(self.objects[(bucket, key)])

    def list_objects_v2(self, **kwargs: str) -> dict:
        bucket, prefix = kwargs["Bucket"], kwargs["Prefix"]
        keys = sorted(k for (b, k) in self.objects if b == bucket and k.startswith(prefix))
        start = int(kwargs.get("ContinuationToken") or 0)
        page = keys[start : start + self.page_size]
        truncated = start + self.page_size < len(keys)
        response: dict = {
            "Contents": [{"Key": k, "Size": len(self.objects[(bucket, k)])} for k in page],
            "IsTruncated": truncated,
        }
        if truncated:
            response["NextContinuationToken"] = str(start + self.page_size)
        return response


def _store(client: FakeS3Client) -> ObjectStore:
    return ObjectStore(bucket="test-bucket", prefix="runs/tictactoe/test_run", client=client)


def _make_run_dir(root: Path) -> Path:
    run_dir = root / "run"
    (run_dir / "Nets").mkdir(parents=True)
    (run_dir / "Logs").mkdir(parents=True)
    (run_dir / "Nets" / "latest.pth.tar").write_bytes(b"weights-v1")
    (run_dir / "Logs" / "progress.json").write_text('{"last_completed_generation": 1}')
    return run_dir


def test_sync_up_is_incremental(tmp_path: Path) -> None:
    client = FakeS3Client()
    store = _store(client)
    run_dir = _make_run_dir(tmp_path)

    assert store.sync_up(run_dir) == 2
    assert store.sync_up(run_dir) == 0  # unchanged files skipped

    marker = run_dir / "Logs" / "progress.json"
    marker.write_text('{"last_completed_generation": 2}')
    assert store.sync_up(run_dir) == 1  # only the changed file re-uploads
    assert client.objects[("test-bucket", "runs/tictactoe/test_run/Logs/progress.json")] == marker.read_bytes()


def test_sync_up_skips_tmp_files(tmp_path: Path) -> None:
    client = FakeS3Client()
    run_dir = _make_run_dir(tmp_path)
    (run_dir / "Logs" / "progress.json.tmp").write_text("half-written")
    _store(client).sync_up(run_dir)
    assert not any(key.endswith(".tmp") for (_b, key) in client.objects)


def test_sync_down_fills_gaps_and_force_overwrites(tmp_path: Path) -> None:
    client = FakeS3Client()
    store = _store(client)
    store.sync_up(_make_run_dir(tmp_path))

    # Fresh machine: everything downloads.
    fresh = tmp_path / "fresh"
    assert store.sync_down(fresh) == 2
    assert (fresh / "Nets" / "latest.pth.tar").read_bytes() == b"weights-v1"

    # Same-size stale local file: the plain sync can't see it, force can.
    (fresh / "Nets" / "latest.pth.tar").write_bytes(b"weights-v2")  # same size, different bytes
    assert store.sync_down(fresh) == 0
    assert store.sync_down(fresh, force=True) == 2
    assert (fresh / "Nets" / "latest.pth.tar").read_bytes() == b"weights-v1"


def test_list_remote_follows_pagination(tmp_path: Path) -> None:
    client = FakeS3Client(page_size=1)  # every object on its own page
    store = _store(client)
    store.sync_up(_make_run_dir(tmp_path))
    target = tmp_path / "paged"
    assert store.sync_down(target) == 2


def test_download_file_returns_false_when_missing(tmp_path: Path) -> None:
    store = _store(FakeS3Client())
    assert store.download_file("Logs/progress.json", tmp_path / "marker.json") is False


def test_create_object_store_none_when_unconfigured(test_config: RunConfig) -> None:
    assert create_object_store(test_config) is None


def test_create_object_store_default_prefix_mirrors_local_layout(test_config: RunConfig) -> None:
    config = replace(test_config, object_store=ObjectStoreConfig(bucket="b"))
    store = create_object_store(config, client=FakeS3Client())
    assert store is not None
    assert store._prefix == "runs/tictactoe/test_run"


def test_sync_up_guarded_swallows_failures(tmp_path: Path) -> None:
    class ExplodingClient(FakeS3Client):
        def upload_file(self, filename: str, bucket: str, key: str) -> None:
            raise ConnectionError("endpoint unreachable")

    store = ObjectStore(bucket="b", prefix="p", client=ExplodingClient())
    run_dir = _make_run_dir(tmp_path)
    sync_up_guarded(store, run_dir, "generation 1")  # must not raise
    sync_up_guarded(None, run_dir, "no store")  # None store is a no-op


def _configured(test_config: RunConfig) -> RunConfig:
    return replace(test_config, object_store=ObjectStoreConfig(bucket="test-bucket"))


def test_restore_noop_without_store(test_config: RunConfig) -> None:
    restore_run_from_object_store(test_config)  # no object_store configured — no-op


def test_restore_noop_when_remote_has_no_marker(test_config: RunConfig) -> None:
    config = _configured(test_config)
    restore_run_from_object_store(config, client=FakeS3Client())
    assert not config.run_directory.exists() or not any(config.run_directory.iterdir())


def test_restore_pulls_run_when_local_is_behind(test_config: RunConfig) -> None:
    config = _configured(test_config)
    client = FakeS3Client()
    # Seed the bucket as a completed-gen-3 run would have left it.
    prefix = "runs/tictactoe/test_run"
    client.objects[("test-bucket", f"{prefix}/Logs/{PROGRESS_MARKER_FILENAME}")] = json.dumps(
        {"last_completed_generation": 3, "wandb_run_id": None}
    ).encode()
    client.objects[("test-bucket", f"{prefix}/Nets/latest.pth.tar")] = b"weights-gen3"

    restore_run_from_object_store(config, client=client)

    assert (config.run_directory / "Nets" / "latest.pth.tar").read_bytes() == b"weights-gen3"
    marker = json.loads((config.log_directory / PROGRESS_MARKER_FILENAME).read_text())
    assert marker["last_completed_generation"] == 3
    assert not (config.log_directory / "progress.remote.json").exists()  # comparison temp cleaned up


def test_restore_keeps_local_when_up_to_date(test_config: RunConfig) -> None:
    config = _configured(test_config)
    client = FakeS3Client()
    prefix = "runs/tictactoe/test_run"
    client.objects[("test-bucket", f"{prefix}/Logs/{PROGRESS_MARKER_FILENAME}")] = json.dumps(
        {"last_completed_generation": 2, "wandb_run_id": None}
    ).encode()
    client.objects[("test-bucket", f"{prefix}/Nets/latest.pth.tar")] = b"remote-stale"

    config.log_directory.mkdir(parents=True)
    (config.log_directory / PROGRESS_MARKER_FILENAME).write_text(
        json.dumps({"last_completed_generation": 5, "wandb_run_id": None})
    )
    (config.run_directory / "Nets").mkdir(parents=True)
    (config.run_directory / "Nets" / "latest.pth.tar").write_bytes(b"local-gen5!!")

    restore_run_from_object_store(config, client=client)

    assert (config.run_directory / "Nets" / "latest.pth.tar").read_bytes() == b"local-gen5!!"
