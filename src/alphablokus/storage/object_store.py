"""S3-compatible object storage for run artefacts (cloud-scale C6/C7).

Mirrors a run directory (checkpoints, parquet metrics, HTML reports, the
resume marker) to any S3-compatible bucket — AWS S3, Cloudflare R2, Backblaze
B2, MinIO, a neocloud's built-in store — so an interrupted cloud instance
loses at most its in-flight generation. Opt-in via ``RunConfig.object_store``;
absent config means pure local-FS behaviour. ``boto3`` is imported only when a
real client is built, so it stays an optional extra (``uv sync --extra s3``)
and importing this module costs nothing on unconfigured runs.

Credentials come from the standard AWS env vars / config chain
(``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY``), never from run JSON.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from pathlib import Path

    from alphablokus.config import RunConfig

# Files never worth mirroring: atomic-write leftovers.
_EXCLUDED_SUFFIXES = (".tmp",)


class ObjectStore:
    """Incremental up/down sync between a local run directory and a bucket prefix.

    ``client`` is anything with boto3's ``upload_file`` / ``download_file`` /
    ``list_objects_v2`` surface — injected in tests, a real boto3 S3 client in
    production (built lazily so importing this module never requires boto3).
    """

    def __init__(
        self,
        bucket: str,
        prefix: str,
        endpoint_url: str | None = None,
        region: str | None = None,
        client: Any | None = None,
    ) -> None:
        self._bucket = bucket
        self._prefix = prefix.strip("/")
        self._client = client if client is not None else self._create_client(endpoint_url, region)
        # (relative path) -> (mtime_ns, size) at last successful upload. Scoped
        # to this process: a resumed process re-uploads once, then goes
        # incremental — correct, at worst slightly wasteful on the first sync.
        self._uploaded: dict[str, tuple[int, int]] = {}

    @staticmethod
    def _create_client(endpoint_url: str | None, region: str | None) -> Any:
        try:
            import boto3
        except ImportError as err:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "object_store is configured but boto3 is not installed — install with: uv sync --extra s3",
            ) from err
        return boto3.client("s3", endpoint_url=endpoint_url, region_name=region)

    def _key_for(self, relative_path: str) -> str:
        return f"{self._prefix}/{relative_path}"

    def sync_up(self, local_dir: Path) -> int:
        """Upload files under ``local_dir`` that changed since the last sync.

        Returns the number of files uploaded. Change detection is
        (mtime_ns, size) against this process's last successful upload.
        """
        uploaded = 0
        for path in sorted(local_dir.rglob("*")):
            if not path.is_file() or path.suffix in _EXCLUDED_SUFFIXES:
                continue
            relative = path.relative_to(local_dir).as_posix()
            stat = path.stat()
            fingerprint = (stat.st_mtime_ns, stat.st_size)
            if self._uploaded.get(relative) == fingerprint:
                continue
            self._client.upload_file(str(path), self._bucket, self._key_for(relative))
            self._uploaded[relative] = fingerprint
            uploaded += 1
        return uploaded

    def sync_down(self, local_dir: Path, force: bool = False) -> int:
        """Download objects under the prefix into ``local_dir``; returns the count.

        By default, existing same-size local files are kept — fill-the-gaps
        semantics for a fresh machine. ``force=True`` downloads everything
        unconditionally: required when the caller knows the bucket is *ahead*
        of the local state (a stale ``latest.pth.tar`` has the same byte size
        as the remote one, so a size check cannot see the difference).
        """
        downloaded = 0
        for key, size in self._list_remote():
            relative = key[len(self._prefix) + 1 :]
            target = local_dir / relative
            if not force and target.exists() and target.stat().st_size == size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            self._client.download_file(self._bucket, key, str(target))
            downloaded += 1
        return downloaded

    def download_file(self, relative_path: str, target: Path) -> bool:
        """Fetch one run-relative file; False if it doesn't exist remotely."""
        for key, _size in self._list_remote():
            if key == self._key_for(relative_path):
                target.parent.mkdir(parents=True, exist_ok=True)
                self._client.download_file(self._bucket, key, str(target))
                return True
        return False

    def _list_remote(self) -> list[tuple[str, int]]:
        """All (key, size) pairs under the prefix, following pagination."""
        results: list[tuple[str, int]] = []
        continuation: str | None = None
        while True:
            kwargs: dict[str, Any] = {"Bucket": self._bucket, "Prefix": f"{self._prefix}/"}
            if continuation:
                kwargs["ContinuationToken"] = continuation
            response = self._client.list_objects_v2(**kwargs)
            results.extend((obj["Key"], int(obj["Size"])) for obj in response.get("Contents", []))
            if not response.get("IsTruncated"):
                return results
            continuation = response.get("NextContinuationToken")


def create_object_store(config: RunConfig, client: Any | None = None) -> ObjectStore | None:
    """Build the run's ``ObjectStore`` from config, or ``None`` when not configured.

    The default prefix mirrors the local layout under the output root
    (``runs/<group>/<run_name>``), so one bucket can hold many runs the same
    way one ``temp/`` does.
    """
    store_config = config.object_store
    if store_config is None:
        return None
    prefix = store_config.prefix or config.run_directory.relative_to(config.root_directory).as_posix()
    return ObjectStore(
        bucket=store_config.bucket,
        prefix=prefix,
        endpoint_url=store_config.endpoint_url,
        region=store_config.region,
        client=client,
    )


def sync_up_guarded(store: ObjectStore | None, local_dir: Path, context: str) -> None:
    """Best-effort upload sync: object-storage trouble must never kill training.

    Same policy as report rendering — everything is already safe on local disk,
    so log the failure and carry on; the next generation's sync retries.
    """
    if store is None:
        return
    try:
        uploaded = store.sync_up(local_dir)
        logger.info("Object store sync ({}): {} file(s) uploaded", context, uploaded)
    except Exception:
        logger.exception("Object store sync failed ({}) — training continues; will retry next sync.", context)
