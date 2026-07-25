"""Tests for the corpus wrap-up's data-safety gate.

The load-bearing guarantee is ``push_to_r2_verified``: it must FAIL (not silently
pass) when an upload does not actually land in the bucket, because that is the
exact failure mode the verify step exists to catch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from alphablokus.storage.object_store import ObjectStore
from scripts.corpus_wrapup import CorpusWrapupError, _object_store_from_config, push_to_r2_verified


class _FakeS3:
    """Minimal in-memory S3 stand-in that can be told to drop specific keys."""

    def __init__(self, drop: set[str] | None = None) -> None:
        self.objects: dict[tuple[str, str], int] = {}
        self._drop = drop or set()

    def upload_file(self, filename: str, bucket: str, key: str) -> None:
        if key in self._drop:
            return  # simulate an upload that "succeeds" locally but never lands
        self.objects[(bucket, key)] = Path(filename).stat().st_size

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
        bucket, prefix = kwargs["Bucket"], kwargs["Prefix"]
        contents = [{"Key": k, "Size": s} for (b, k), s in self.objects.items() if b == bucket and k.startswith(prefix)]
        return {"Contents": contents, "IsTruncated": False}


def _corpus(tmp_path: Path) -> Path:
    d = tmp_path / "corpus"
    d.mkdir()
    (d / "corpus_00000.parquet").write_bytes(b"aaa")
    (d / "corpus_00001.parquet").write_bytes(b"bbbb")
    (d / "scratch.tmp").write_bytes(b"x")  # excluded from sync + verify
    return d


def test_push_verified_happy_path(tmp_path: Path) -> None:
    store = ObjectStore(bucket="b", prefix="corpora/x", client=_FakeS3())
    uploaded, verified = push_to_r2_verified(store, _corpus(tmp_path))
    assert uploaded == 2  # the .tmp file is excluded
    assert verified == 2


def test_push_verified_detects_upload_that_did_not_land(tmp_path: Path) -> None:
    # The client accepts the upload call but never stores one shard → verify must fail.
    store = ObjectStore(bucket="b", prefix="corpora/x", client=_FakeS3(drop={"corpora/x/corpus_00001.parquet"}))
    with pytest.raises(CorpusWrapupError, match="verify FAILED"):
        push_to_r2_verified(store, _corpus(tmp_path))


def test_push_empty_dir_raises(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    store = ObjectStore(bucket="b", prefix="corpora/x", client=_FakeS3())
    with pytest.raises(CorpusWrapupError, match="no files"):
        push_to_r2_verified(store, tmp_path / "empty")


def test_object_store_from_config_requires_a_block(tmp_path: Path) -> None:
    cfg = tmp_path / "no_store.json"
    cfg.write_text('{"game": "blokusduo"}')
    with pytest.raises(CorpusWrapupError, match="object_store"):
        _object_store_from_config(cfg, prefix="corpora/x")
