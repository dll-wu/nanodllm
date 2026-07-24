"""Deterministic data, tokenizer, and packing identity records.

The default dataset digest is intentionally metadata-only: it covers canonical
paths, sizes, mtimes, Parquet footer counts, and exact split membership, but it
does not read every dataset byte. Pass ``full_sha256=True`` when byte-level
dataset identity is worth the additional I/O.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections.abc import Mapping, Sequence
from glob import glob
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

DATA_IDENTITY_VERSION = 1
HASH_CHUNK_BYTES = 8 * 1024 * 1024

PACKING_DATA_FIELDS = (
    "text_column",
    "tokenizer_batch_size",
    "prepend_bos",
    "append_eos",
    "loss_on_bos",
    "loss_on_eos",
    "random_length_ratio",
    "random_length_min",
    "continuation_ratio",
    "continuation_min_prefix_frac",
    "continuation_max_prefix_frac",
    "shuffle_row_groups",
    "shuffle_seed",
    "boundary_aware_packing",
    "content_budget_segmented_packing",
    "short_document_curriculum",
    "short_document_max_tokens",
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_patterns(patterns: str | Sequence[str] | None) -> list[str]:
    if patterns is None:
        return []
    if isinstance(patterns, str):
        return [patterns]
    return [str(pattern) for pattern in patterns]


def _canonical_existing_file(path: str | os.PathLike[str]) -> str:
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(f"expected a file, got: {path!s}")
    return str(resolved)


def resolve_parquet_files(patterns: str | Sequence[str]) -> list[str]:
    """Resolve paths exactly once into a stable, deduplicated file order."""

    resolved_files: set[str] = set()
    normalized = _normalize_patterns(patterns)
    if not normalized:
        raise FileNotFoundError("no dataset paths were provided")
    for raw in normalized:
        expanded = os.path.expanduser(os.path.expandvars(raw))
        path = Path(expanded)
        matches: list[Path]
        if path.is_dir():
            matches = list(path.rglob("*.parquet"))
        elif any(character in expanded for character in "*?[]"):
            matches = [Path(match) for match in glob(expanded, recursive=True)]
        elif path.is_file():
            matches = [path]
        else:
            raise FileNotFoundError(f"dataset path did not match any parquet files: {raw!r}")
        parquet_matches = [match for match in matches if match.is_file() and match.name.endswith(".parquet")]
        if not parquet_matches:
            raise FileNotFoundError(f"dataset path did not match any parquet files: {raw!r}")
        resolved_files.update(_canonical_existing_file(match) for match in parquet_matches)
    if not resolved_files:
        raise FileNotFoundError(f"no parquet files found for dataset patterns: {normalized!r}")
    return sorted(resolved_files)


def resolve_parquet_selection(
    include_patterns: str | Sequence[str],
    exclude_patterns: str | Sequence[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Resolve a fail-closed whole-file include/exclude selection."""

    included = resolve_parquet_files(include_patterns)
    requested_excludes = _normalize_patterns(exclude_patterns)
    if not requested_excludes:
        return included, []

    excluded = resolve_parquet_files(requested_excludes)
    included_set = set(included)
    outside = sorted(set(excluded) - included_set)
    if outside:
        raise ValueError("excluded parquet files are outside the resolved train selection: " + ", ".join(outside))
    excluded_set = set(excluded)
    selected = [path for path in included if path not in excluded_set]
    if not selected:
        raise ValueError("train exclusions removed every resolved parquet file")
    return selected, excluded


def _stat_identity(stat_result: os.stat_result) -> tuple[int, int]:
    return int(stat_result.st_size), int(stat_result.st_mtime_ns)


def _inspect_parquet_file(path: str, *, full_sha256: bool) -> dict[str, Any]:
    before = os.stat(path)
    parquet_file = pq.ParquetFile(path)
    metadata = parquet_file.metadata
    if metadata is None:
        raise ValueError(f"parquet file has no footer metadata: {path}")
    row_group_num_rows = [int(metadata.row_group(index).num_rows) for index in range(metadata.num_row_groups)]
    row_group_total_byte_sizes = [
        int(metadata.row_group(index).total_byte_size) for index in range(metadata.num_row_groups)
    ]
    record: dict[str, Any] = {
        "path": path,
        "size_bytes": int(before.st_size),
        "mtime_ns": int(before.st_mtime_ns),
        "parquet_footer": {
            "num_rows": int(metadata.num_rows),
            "num_row_groups": int(metadata.num_row_groups),
            "num_columns": int(metadata.num_columns),
            "row_group_num_rows": row_group_num_rows,
            "row_group_total_byte_sizes": row_group_total_byte_sizes,
        },
    }
    if full_sha256:
        record["full_sha256"] = sha256_file(path)
    after = os.stat(path)
    if _stat_identity(before) != _stat_identity(after):
        raise RuntimeError(f"dataset file changed while its identity was being recorded: {path}")
    return record


def _row_groups(file_records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    membership: list[dict[str, Any]] = []
    for file_record in file_records:
        footer = file_record["parquet_footer"]
        for row_group, num_rows in enumerate(footer["row_group_num_rows"]):
            membership.append(
                {
                    "path": str(file_record["path"]),
                    "row_group": int(row_group),
                    "num_rows": int(num_rows),
                }
            )
    return membership


def _split_record(row_groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    paths = sorted({str(row_group["path"]) for row_group in row_groups})
    return {
        "files": paths,
        "file_count": len(paths),
        "row_groups": [dict(row_group) for row_group in row_groups],
        "row_group_count": len(row_groups),
        "row_count": sum(int(row_group["num_rows"]) for row_group in row_groups),
    }


def _without_dataset_content_hashes(dataset: Mapping[str, Any]) -> dict[str, Any]:
    files = [
        {key: value for key, value in file_record.items() if key != "full_sha256"} for file_record in dataset["files"]
    ]
    return {
        "split_policy": dataset["split_policy"],
        "files": files,
        "membership": dataset["membership"],
    }


def build_dataset_inventory(
    train_data: str | Sequence[str],
    val_data: str | Sequence[str] | None = None,
    *,
    train_exclude_data: str | Sequence[str] | None = None,
    val_fraction: float = 0.0,
    full_sha256: bool = False,
) -> dict[str, Any]:
    """Build exact train/validation file and row-group membership.

    With no explicit validation files, row groups are ordered by canonical file
    path and footer index, then the deterministic tail is assigned to
    validation. This matches the current loader's split rule while making the
    membership auditable.
    """

    if val_fraction < 0 or val_fraction >= 1:
        raise ValueError("val_fraction must be in [0, 1)")
    requested_train = _normalize_patterns(train_data)
    requested_train_excludes = _normalize_patterns(train_exclude_data)
    requested_val = _normalize_patterns(val_data)
    train_files, excluded_train_files = resolve_parquet_selection(requested_train, requested_train_excludes)
    explicit_validation = bool(requested_val)
    val_files = resolve_parquet_files(requested_val) if explicit_validation else []
    overlap = sorted(set(train_files) & set(val_files))
    if overlap:
        rendered = ", ".join(overlap)
        raise ValueError(f"explicit train and validation datasets overlap after canonicalization: {rendered}")

    all_files = sorted(set(train_files) | set(val_files))
    file_records = [_inspect_parquet_file(path, full_sha256=full_sha256) for path in all_files]
    by_path = {str(record["path"]): record for record in file_records}
    train_row_groups = _row_groups([by_path[path] for path in train_files])
    if not train_row_groups:
        raise ValueError("train dataset has no parquet row groups")

    if explicit_validation:
        validation_row_groups = _row_groups([by_path[path] for path in val_files])
        if not validation_row_groups:
            raise ValueError("validation dataset has no parquet row groups")
        split_policy = {
            "mode": "explicit_files",
            "configured_val_fraction": float(val_fraction),
            "val_fraction_applied": False,
            "ordering": "canonical_file_path_then_parquet_footer_row_group_index",
            "explicit_overlap_check": "canonical_file_intersection_rejected",
        }
    elif val_fraction > 0:
        validation_count = max(1, round(len(train_row_groups) * val_fraction))
        if validation_count >= len(train_row_groups):
            raise ValueError("validation split would consume all row groups; reduce val_fraction or provide more data")
        validation_row_groups = train_row_groups[-validation_count:]
        train_row_groups = train_row_groups[:-validation_count]
        split_policy = {
            "mode": "deterministic_tail_row_groups",
            "configured_val_fraction": float(val_fraction),
            "val_fraction_applied": True,
            "validation_row_group_count": int(validation_count),
            "rounding": "max(1, python_round(total_row_groups * val_fraction))",
            "ordering": "canonical_file_path_then_parquet_footer_row_group_index",
        }
    else:
        validation_row_groups = []
        split_policy = {
            "mode": "none",
            "configured_val_fraction": float(val_fraction),
            "val_fraction_applied": False,
            "ordering": "canonical_file_path_then_parquet_footer_row_group_index",
        }

    train_keys = {(str(row_group["path"]), int(row_group["row_group"])) for row_group in train_row_groups}
    validation_keys = {(str(row_group["path"]), int(row_group["row_group"])) for row_group in validation_row_groups}
    if train_keys & validation_keys:
        raise RuntimeError("internal error: train and validation row-group membership overlap")

    selection: dict[str, Any] = {
        "requested_train_patterns": requested_train,
        "requested_validation_patterns": requested_val,
    }
    if requested_train_excludes:
        selection["requested_train_exclude_patterns"] = requested_train_excludes
        selection["resolved_train_excluded_files"] = excluded_train_files
    dataset: dict[str, Any] = {
        "selection": selection,
        "split_policy": split_policy,
        "files": file_records,
        "membership": {
            "train": _split_record(train_row_groups),
            "validation": _split_record(validation_row_groups),
        },
    }
    dataset["metadata_inventory_sha256"] = _json_sha256(_without_dataset_content_hashes(dataset))
    dataset["metadata_inventory_sha256_is_content_hash"] = False
    if full_sha256:
        dataset["full_content_inventory_sha256"] = _json_sha256(
            [{"path": record["path"], "full_sha256": record["full_sha256"]} for record in file_records]
        )
    return dataset


def assign_row_groups(
    row_groups: Sequence[Mapping[str, Any]],
    *,
    rank: int,
    world_size: int,
    shuffle_seed: int | None = None,
    epoch: int = 1,
) -> list[dict[str, Any]]:
    """Reproduce the loader's optional epoch shuffle followed by rank striding."""

    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    ordered = [dict(row_group) for row_group in row_groups]
    if shuffle_seed is not None:
        random.Random(int(shuffle_seed) + int(epoch)).shuffle(ordered)
    return ordered[rank::world_size]


def _distributed_membership_summary(
    dataset: Mapping[str, Any],
    *,
    world_size: int,
    train_shuffle: bool,
    shuffle_seed: int,
    assignment_epoch: int,
) -> dict[str, Any]:
    train_row_groups = dataset["membership"]["train"]["row_groups"]
    validation_row_groups = dataset["membership"]["validation"]["row_groups"]
    ranks = []
    for rank in range(world_size):
        rank_train = assign_row_groups(
            train_row_groups,
            rank=rank,
            world_size=world_size,
            shuffle_seed=shuffle_seed if train_shuffle else None,
            epoch=assignment_epoch,
        )
        rank_validation = assign_row_groups(validation_row_groups, rank=rank, world_size=world_size)
        ranks.append(
            {
                "rank": rank,
                "train_row_group_count": len(rank_train),
                "train_membership_sha256": _json_sha256(rank_train),
                "validation_row_group_count": len(rank_validation),
                "validation_membership_sha256": _json_sha256(rank_validation),
            }
        )
    return {
        "world_size": int(world_size),
        "assignment_epoch": int(assignment_epoch),
        "assignment": "optional_epoch_shuffle_then_python_rank_stride_[rank::world_size]",
        "train_shuffle_row_groups": bool(train_shuffle),
        "train_shuffle_seed": int(shuffle_seed),
        "train_epoch_seed_rule": "shuffle_seed + epoch",
        "rank_summaries": ranks,
    }


def _stable_tokenizer_file_record(root: Path, path: Path) -> dict[str, Any]:
    relative_path = path.relative_to(root).as_posix()
    symlink_target: str | None = None
    if path.is_symlink():
        target = path.resolve(strict=True)
        if not target.is_file():
            raise ValueError(f"tokenizer symlink does not point to a regular file: {path}")
        symlink_target = os.readlink(path)
    elif not path.is_file():
        raise ValueError(f"tokenizer inventory entry is not a regular file: {path}")
    before = path.stat()
    record: dict[str, Any] = {
        "relative_path": relative_path,
        "size_bytes": int(before.st_size),
        "mtime_ns": int(before.st_mtime_ns),
        "sha256": sha256_file(path),
    }
    if symlink_target is not None:
        record["symlink_target"] = symlink_target
    after = path.stat()
    if _stat_identity(before) != _stat_identity(after):
        raise RuntimeError(f"tokenizer file changed while its identity was being recorded: {path}")
    return record


def build_tokenizer_inventory(tokenizer_kind: str, tokenizer_path: str | os.PathLike[str] | None) -> dict[str, Any]:
    if tokenizer_kind == "byte":
        inventory = {
            "kind": "byte",
            "path": None,
            "files": [],
            "token_bytes_source": "computed_byte_tokenizer_table",
            "token_bytes_pt_sha256": None,
        }
        inventory["inventory_sha256"] = _json_sha256(inventory)
        return inventory
    if tokenizer_kind != "hf":
        raise ValueError(f"unsupported tokenizer kind for identity: {tokenizer_kind!r}")
    if not tokenizer_path:
        raise ValueError("tokenizer_path is required for an HF tokenizer identity")
    requested_path = os.path.abspath(os.path.expanduser(os.path.expandvars(str(tokenizer_path))))
    root = Path(requested_path)
    if not root.is_dir():
        raise FileNotFoundError(f"tokenizer path is not a directory: {tokenizer_path!s}")
    files: list[Path] = []
    for candidate in root.rglob("*"):
        if candidate.is_symlink() and candidate.resolve(strict=True).is_dir():
            raise ValueError(
                f"tokenizer directory contains a symlinked directory, which is not inventoried: {candidate}"
            )
        if candidate.is_file() or candidate.is_symlink():
            files.append(candidate)
    file_records = [
        _stable_tokenizer_file_record(root, path)
        for path in sorted(files, key=lambda item: item.relative_to(root).as_posix())
    ]
    token_bytes_record = next(
        (record for record in file_records if record["relative_path"] == "token_bytes.pt"),
        None,
    )
    token_bytes_json_record = next(
        (record for record in file_records if record["relative_path"] == "token_bytes.json"),
        None,
    )
    token_bytes_source = "tokenizer_decode_fallback"
    if token_bytes_record:
        token_bytes_source = "token_bytes.pt"
    if token_bytes_json_record:
        token_bytes_source = "token_bytes.json"
    inventory = {
        "kind": "hf",
        "requested_path": requested_path,
        "path": str(root.resolve(strict=True)),
        "files": file_records,
        "file_count": len(file_records),
        "token_bytes_source": token_bytes_source,
        "token_bytes_pt_sha256": token_bytes_record["sha256"] if token_bytes_record else None,
    }
    if token_bytes_json_record:
        inventory["token_bytes_json_sha256"] = token_bytes_json_record["sha256"]
    inventory["inventory_sha256"] = _json_sha256(inventory)
    return inventory


def _tokenizer_metadata_projection(tokenizer: Mapping[str, Any]) -> dict[str, Any]:
    files = []
    for file_record in tokenizer["files"]:
        files.append({key: value for key, value in file_record.items() if key != "sha256"})
    return {
        key: value
        for key, value in tokenizer.items()
        if key not in {"files", "inventory_sha256", "token_bytes_pt_sha256", "token_bytes_json_sha256"}
    } | {"files": files}


def packing_settings_from_config(config: Any) -> dict[str, Any]:
    data_config = config.data
    if bool(getattr(data_config, "short_document_curriculum", False)):
        mode = "short_document_curriculum"
    elif bool(getattr(data_config, "boundary_aware_packing", False)):
        mode = "boundary_aware"
    elif bool(getattr(data_config, "content_budget_segmented_packing", False)):
        mode = "content_budget_segmented"
    else:
        mode = "stream"
    resolved_training_seed = int(
        config.runtime.training_seed if int(config.runtime.training_seed) >= 0 else config.runtime.seed
    )
    return {
        "mode": mode,
        "sequence_length": int(config.model.max_seq_len),
        "device_batch_size": int(config.train.device_batch_size),
        "resolved_training_seed": resolved_training_seed,
        "rank_training_seed_rule": "resolved_training_seed + rank",
        "mask_augmentation_rng": "loader-local random.Random",
        "mask_augmentation_seed_rule": "resolved_training_seed + rank",
        "data": {field: getattr(data_config, field) for field in PACKING_DATA_FIELDS},
    }


def build_data_identity(
    config: Any,
    *,
    full_sha256: bool = False,
    world_size: int = 1,
    assignment_epoch: int = 1,
) -> dict[str, Any]:
    """Build the standalone identity artifact for a resolved experiment config."""

    if getattr(config.data, "synthetic_dataset", ""):
        raise ValueError("data identity currently requires parquet train_data, not a synthetic dataset")
    dataset = build_dataset_inventory(
        config.data.train_data,
        config.data.val_data,
        train_exclude_data=getattr(config.data, "train_exclude_data", None),
        val_fraction=float(config.data.val_fraction),
        full_sha256=full_sha256,
    )
    tokenizer = build_tokenizer_inventory(config.data.tokenizer, config.data.tokenizer_path)
    packing = packing_settings_from_config(config)
    distributed = _distributed_membership_summary(
        dataset,
        world_size=int(world_size),
        train_shuffle=bool(config.data.shuffle_row_groups),
        shuffle_seed=int(config.data.shuffle_seed),
        assignment_epoch=int(assignment_epoch),
    )
    identity_material = {
        "version": DATA_IDENTITY_VERSION,
        "dataset": dataset,
        "tokenizer": tokenizer,
        "packing": packing,
        "distributed_membership": distributed,
    }
    metadata_material = {
        "version": DATA_IDENTITY_VERSION,
        "dataset": _without_dataset_content_hashes(dataset),
        "tokenizer": _tokenizer_metadata_projection(tokenizer),
        "packing": packing,
        "distributed_membership": distributed,
    }
    hashes: dict[str, Any] = {
        "metadata_inventory_sha256": _json_sha256(metadata_material),
        "metadata_inventory_sha256_is_content_hash": False,
        "metadata_inventory_sha256_scope": (
            "canonical paths, sizes, mtimes, parquet footer counts, exact split/DDP membership, "
            "packing settings, and tokenizer file metadata"
        ),
        "identity_sha256": _json_sha256(identity_material),
        "identity_sha256_strength": (
            "dataset_full_content_and_tokenizer_content" if full_sha256 else "dataset_metadata_and_tokenizer_content"
        ),
        "dataset_full_content_hashing_enabled": bool(full_sha256),
    }
    if full_sha256:
        hashes["full_content_inventory_sha256"] = _json_sha256(
            {
                "dataset_files": [
                    {"path": record["path"], "full_sha256": record["full_sha256"]} for record in dataset["files"]
                ],
                "tokenizer_files": [
                    {"relative_path": record["relative_path"], "sha256": record["sha256"]}
                    for record in tokenizer["files"]
                ],
            }
        )
    return {**identity_material, "hashes": hashes}
