"""Parquet discovery, row-group streaming and prepared-snapshot admission."""

import random
from collections import OrderedDict
from dataclasses import dataclass

import pyarrow.parquet as pq

from .data_identity import resolve_parquet_files, resolve_parquet_selection
from .data_snapshot import CheckedParquetFile, load_snapshot


@dataclass(frozen=True)
class RowGroupRef:
    path: str
    row_group: int


def _normalize_patterns(patterns: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(patterns, str):
        return [patterns]
    return list(patterns)


def _resolve_parquet_files(patterns: str | list[str] | tuple[str, ...]) -> list[str]:
    return resolve_parquet_files(_normalize_patterns(patterns))


def _row_groups_for_files(files: list[str]) -> list[RowGroupRef]:
    row_groups: list[RowGroupRef] = []
    for path in files:
        pf = pq.ParquetFile(path)
        row_groups.extend(RowGroupRef(path, rg_idx) for rg_idx in range(pf.num_row_groups))
    return row_groups


def _split_row_groups(data_config):
    train_files, _excluded_train_files = resolve_parquet_selection(
        data_config.train_data,
        getattr(data_config, "train_exclude_data", None),
    )
    train_row_groups = _row_groups_for_files(train_files)
    if not train_row_groups:
        raise ValueError("train dataset has no parquet row groups")

    if data_config.val_data:
        val_files = _resolve_parquet_files(data_config.val_data)
        overlap = sorted(set(train_files) & set(val_files))
        if overlap:
            raise ValueError(
                "explicit train and validation datasets overlap after canonicalization: " + ", ".join(overlap)
            )
        val_row_groups = _row_groups_for_files(val_files)
        if not val_row_groups:
            raise ValueError("val dataset has no parquet row groups")
        return train_row_groups, val_row_groups

    if data_config.val_fraction <= 0:
        return train_row_groups, []
    n_val = max(1, round(len(train_row_groups) * data_config.val_fraction))
    if n_val >= len(train_row_groups):
        raise ValueError("validation split would consume all row groups; reduce data.val_fraction or provide more data")
    return train_row_groups[:-n_val], train_row_groups[-n_val:]


class ParquetTextSource:
    def __init__(self, data_config):
        self.text_column = data_config.text_column
        self.tokenizer_batch_size = int(data_config.tokenizer_batch_size)
        self.shuffle_row_groups = bool(data_config.shuffle_row_groups)
        self.shuffle_seed = int(data_config.shuffle_seed)
        snapshot = load_snapshot(data_config) if getattr(data_config, "dataset_snapshot", "") else None
        self.snapshot_files = snapshot.file_records() if snapshot else {}
        self.snapshot_full_sha256 = snapshot.full_sha256 if snapshot else False
        if snapshot:
            membership = snapshot.dataset["membership"]
            self.train_row_groups, self.val_row_groups = (
                [RowGroupRef(group["path"], group["row_group"]) for group in membership[name]["row_groups"]]
                for name in ("train", "validation")
            )
        else:
            self.train_row_groups, self.val_row_groups = _split_row_groups(data_config)
        self._pf_cache: OrderedDict[str, pq.ParquetFile | CheckedParquetFile] = OrderedDict()

    def _get_parquet_file(self, path: str) -> pq.ParquetFile | CheckedParquetFile:
        pf = self._pf_cache.get(path)
        if pf is not None:
            self._pf_cache.move_to_end(path)
            return pf
        pf = (
            CheckedParquetFile(self.snapshot_files[path], full_sha256=self.snapshot_full_sha256)
            if self.snapshot_files
            else pq.ParquetFile(path)
        )
        self._pf_cache[path] = pf
        if len(self._pf_cache) > 16:
            _, evicted = self._pf_cache.popitem(last=False)
            evicted.close()
        return pf

    def _epoch_row_groups(self, split: str, epoch: int, rank: int, world_size: int) -> list[RowGroupRef]:
        if split == "train":
            row_groups = list(self.train_row_groups)
        elif split == "val":
            row_groups = list(self.val_row_groups)
        else:
            raise ValueError(f"unknown split: {split!r}")
        if not row_groups:
            raise ValueError(f"no row groups available for split {split!r}")
        if self.shuffle_row_groups and split == "train":
            random.Random(self.shuffle_seed + epoch).shuffle(row_groups)
        return row_groups[rank::world_size]

    def document_batches(self, split: str, rank: int = 0, world_size: int = 1):
        epoch = 1
        while True:
            row_groups = self._epoch_row_groups(split, epoch, rank, world_size)
            if not row_groups:
                raise ValueError(f"rank {rank} received no row groups for split {split!r}")
            for ref in row_groups:
                pf = self._get_parquet_file(ref.path)
                table = pf.read_row_group(ref.row_group, columns=[self.text_column])
                texts = [t for t in table.column(self.text_column).to_pylist() if isinstance(t, str) and t]
                for i in range(0, len(texts), self.tokenizer_batch_size):
                    yield texts[i : i + self.tokenizer_batch_size], epoch
            epoch += 1
