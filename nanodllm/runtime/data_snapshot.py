"""Explicit prepared dataset snapshots and validation of shards as they are read."""

from __future__ import annotations

import hashlib
import os
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from .data_inventory_codec import read_inventory, require, require_digest

VERIFICATION_POLICY = "pinned_inventory_and_opened_shards_v1"


def strip_disabled_snapshot(value: dict) -> None:
    """Preserve frozen projections only when this additive feature is disabled."""
    data = value["data"]
    if data.get("dataset_snapshot", "") == "" and data.get("dataset_snapshot_sha256", "") == "":
        data.pop("dataset_snapshot", None)
        data.pop("dataset_snapshot_sha256", None)


def validate_snapshot_config(data) -> None:
    path, digest = data.dataset_snapshot, data.dataset_snapshot_sha256
    require(isinstance(path, str) and isinstance(digest, str), "dataset snapshot settings must be strings")
    require(bool(path) == bool(digest), "dataset_snapshot and dataset_snapshot_sha256 must be set together")
    if path:
        require_digest(digest)
        require(
            not data.synthetic_dataset and data.identity_mode != "off",
            "dataset snapshot requires Parquet identity_mode",
        )


def _patterns(value) -> list[str]:
    return [value] if isinstance(value, str) else list(value or [])


@dataclass(frozen=True)
class DatasetSnapshot:
    dataset: dict
    raw: bytes
    sha256: str
    full_sha256: bool

    def file_records(self) -> dict[str, dict]:
        return {record["path"]: record for record in self.dataset["files"]}


def load_snapshot(data) -> DatasetSnapshot:
    """Read one pinned object, with no discovery, stat or footer reads on the corpus."""
    validate_snapshot_config(data)
    require(bool(data.dataset_snapshot), "dataset snapshot is not configured")
    dataset, raw = read_inventory(Path(data.dataset_snapshot), data.dataset_snapshot_sha256)
    selection = dataset["selection"]
    for field, key in (
        ("train_data", "requested_train_patterns"),
        ("val_data", "requested_validation_patterns"),
        ("train_exclude_data", "requested_train_exclude_patterns"),
    ):
        require(_patterns(getattr(data, field)) == selection.get(key, []), f"snapshot selection differs: {field}")
    require(
        float(data.val_fraction) == dataset["split_policy"]["configured_val_fraction"], "snapshot val_fraction differs"
    )
    full = data.identity_mode == "full_sha256"
    has_full = all("full_sha256" in record for record in dataset["files"])
    require(full == has_full, "snapshot hashing strength differs from identity_mode; prepare a matching snapshot")
    return DatasetSnapshot(dataset, raw, data.dataset_snapshot_sha256, full)


def _stat_identity(info) -> tuple[int, ...]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


class CheckedParquetFile:
    """Keep the validated file descriptor open and detect ordinary changes on reads."""

    def __init__(self, record: dict, *, full_sha256: bool):
        self.path = record["path"]
        self._resources = ExitStack()
        # The descriptor must outlive construction; close() owns its lifetime.
        self.stream = self._resources.enter_context(open(self.path, "rb"))  # noqa: SIM115
        try:
            info = os.fstat(self.stream.fileno())
            self.identity = _stat_identity(info)
            require(
                (info.st_size, info.st_mtime_ns) == (record["size_bytes"], record["mtime_ns"]),
                f"dataset shard changed: {self.path}",
            )
            if full_sha256:
                digest = hashlib.sha256()
                for block in iter(lambda: self.stream.read(8 * 1024**2), b""):
                    digest.update(block)
                require(digest.hexdigest() == record.get("full_sha256"), f"dataset shard content changed: {self.path}")
                self.stream.seek(0)
            self.parquet = pq.ParquetFile(self.stream)
            meta = self.parquet.metadata
            observed = {
                "num_rows": meta.num_rows,
                "num_row_groups": meta.num_row_groups,
                "num_columns": meta.num_columns,
                "row_group_num_rows": [meta.row_group(i).num_rows for i in range(meta.num_row_groups)],
                "row_group_total_byte_sizes": [meta.row_group(i).total_byte_size for i in range(meta.num_row_groups)],
            }
            require(observed == record["parquet_footer"], f"dataset shard footer changed: {self.path}")
            self._check_unchanged()
        except BaseException:
            self._resources.close()
            raise

    def _check_unchanged(self) -> None:
        require(
            _stat_identity(os.fstat(self.stream.fileno())) == self.identity, f"open dataset shard changed: {self.path}"
        )
        require(_stat_identity(os.stat(self.path)) == self.identity, f"dataset shard replaced: {self.path}")

    def read_row_group(self, index: int, *, columns: list[str]):
        self._check_unchanged()
        result = self.parquet.read_row_group(index, columns=columns)
        self._check_unchanged()
        return result

    def close(self) -> None:
        self.parquet.close()
        self._resources.close()
