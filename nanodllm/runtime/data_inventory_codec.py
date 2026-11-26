"""Lossless, bounded storage for prepared Parquet inventories (no corpus I/O)."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import tempfile
from pathlib import Path

SCHEMA = "nanodllm.dataset_inventory.v1"
MAX_COMPRESSED_BYTES = 64 * 1024**2
MAX_JSON_BYTES = 128 * 1024**2
MAX_ROW_GROUPS = 4_000_000


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def require_digest(value: str) -> None:
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None, "invalid inventory SHA-256")


def strict_json(raw: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"nonfinite JSON value: {value}")

    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    require(isinstance(value, dict), "inventory must be a JSON object")
    return value


def _integer(value: object, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _validate_files(files: object) -> None:
    require(isinstance(files, list) and 0 < len(files) <= 100_000, "invalid inventory files")
    paths = set()
    groups = 0
    for record in files:
        require(isinstance(record, dict), "invalid file record")
        path = record.get("path")
        require(isinstance(path, str) and Path(path).is_absolute() and path not in paths, "invalid/duplicate file path")
        paths.add(path)
        require(_integer(record.get("size_bytes")) and _integer(record.get("mtime_ns")), "invalid file stat")
        footer = record.get("parquet_footer")
        require(isinstance(footer, dict), "missing Parquet footer")
        count = footer.get("num_row_groups")
        require(_integer(count) and _integer(footer.get("num_columns"), 1), "invalid footer count")
        groups += count
        require(groups <= MAX_ROW_GROUPS, "inventory row-group limit exceeded")
        for key in ("row_group_num_rows", "row_group_total_byte_sizes"):
            values = footer.get(key)
            require(isinstance(values, list) and len(values) == count, "invalid footer arrays")
            require(all(_integer(v) for v in values), "invalid footer value")
        require(footer.get("num_rows") == sum(footer["row_group_num_rows"]), "footer row count differs")
        if "full_sha256" in record:
            require_digest(record["full_sha256"])


def compact_dataset(dataset: dict) -> dict:
    """Encode ordered membership as file-index/half-open row-group ranges."""
    files = dataset["files"]
    _validate_files(files)
    ids = {record["path"]: index for index, record in enumerate(files)}
    result = {key: value for key, value in dataset.items() if key != "membership"}
    result["membership"] = {}
    for name, split in dataset["membership"].items():
        spans = []
        for group in split["row_groups"]:
            index, group_index = ids[group["path"]], group["row_group"]
            require(set(group) == {"path", "row_group", "num_rows"}, "unsupported row-group fields")
            require(
                _integer(group_index) and group_index < files[index]["parquet_footer"]["num_row_groups"],
                "invalid group index",
            )
            require(
                group["num_rows"] == files[index]["parquet_footer"]["row_group_num_rows"][group_index],
                "group row count differs",
            )
            if spans and spans[-1][0] == index and spans[-1][2] == group_index:
                spans[-1][2] += 1
            else:
                spans.append([index, group_index, group_index + 1])
        result["membership"][name] = {
            **{key: value for key, value in split.items() if key not in {"files", "row_groups"}},
            "file_ids": [ids[path] for path in split["files"]],
            "row_group_spans": spans,
        }
    return {"schema": SCHEMA, "dataset": result}


def expand_dataset(payload: dict) -> dict:
    require(set(payload) == {"schema", "dataset"} and payload["schema"] == SCHEMA, "unsupported inventory schema")
    dataset = payload["dataset"]
    require(isinstance(dataset, dict), "invalid dataset inventory")
    files = dataset.get("files")
    _validate_files(files)
    membership = dataset.get("membership")
    require(isinstance(membership, dict) and set(membership) == {"train", "validation"}, "invalid split membership")
    result = {key: value for key, value in dataset.items() if key != "membership"}
    result["membership"] = {}
    seen = set()
    for name, split in membership.items():
        require(isinstance(split, dict), "invalid split record")
        ids, spans = split.get("file_ids"), split.get("row_group_spans")
        require(isinstance(ids, list) and all(_integer(i) and i < len(files) for i in ids), "invalid split file IDs")
        require(isinstance(spans, list) and len(spans) <= MAX_ROW_GROUPS, "invalid split spans")
        groups = []
        for span in spans:
            require(
                isinstance(span, list) and len(span) == 3 and all(_integer(v) for v in span), "invalid row-group span"
            )
            index, start, end = span
            require(index < len(files), "invalid span file ID")
            footer = files[index]["parquet_footer"]
            require(start < end <= footer["num_row_groups"], "invalid span bounds")
            require(len(seen) + end - start <= MAX_ROW_GROUPS, "expanded membership limit exceeded")
            for group_index in range(start, end):
                key = (index, group_index)
                require(key not in seen, "duplicate or overlapping split membership")
                seen.add(key)
                groups.append(
                    {
                        "path": files[index]["path"],
                        "row_group": group_index,
                        "num_rows": footer["row_group_num_rows"][group_index],
                    }
                )
        paths = [files[index]["path"] for index in ids]
        require(paths == sorted({group["path"] for group in groups}), "split files differ from membership")
        require(split.get("file_count") == len(paths), "split file count differs")
        require(split.get("row_group_count") == len(groups), "split group count differs")
        require(split.get("row_count") == sum(group["num_rows"] for group in groups), "split row count differs")
        result["membership"][name] = {
            **{key: value for key, value in split.items() if key not in {"file_ids", "row_group_spans"}},
            "files": paths,
            "row_groups": groups,
        }
    require(bool(result["membership"]["train"]["row_groups"]), "empty training membership")
    return result


def encode_inventory(dataset: dict) -> bytes:
    payload = compact_dataset(dataset)
    require(expand_dataset(payload) == dataset, "inventory codec is not lossless")
    raw = canonical_bytes(payload)
    require(len(raw) <= MAX_JSON_BYTES, "inventory JSON limit exceeded")
    # GzipFile fixes OS/header fields across supported Python versions and platforms.
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", filename="", compresslevel=6, mtime=0) as stream:
        stream.write(raw)
    compressed = output.getvalue()
    require(len(compressed) <= MAX_COMPRESSED_BYTES, "compressed inventory limit exceeded")
    return compressed


def decode_inventory(raw: bytes, expected_sha256: str) -> dict:
    require_digest(expected_sha256)
    require(len(raw) <= MAX_COMPRESSED_BYTES and sha256(raw) == expected_sha256, "inventory size/hash mismatch")
    with gzip.GzipFile(fileobj=io.BytesIO(raw), mode="rb") as stream:
        decoded = stream.read(MAX_JSON_BYTES + 1)
    require(len(decoded) <= MAX_JSON_BYTES, "expanded inventory limit exceeded")
    return expand_dataset(strict_json(decoded))


def read_inventory(path: Path, expected_sha256: str) -> tuple[dict, bytes]:
    with path.open("rb") as stream:
        raw = stream.read(MAX_COMPRESSED_BYTES + 1)
    return decode_inventory(raw, expected_sha256), raw


def publish_inventory(root: Path, raw: bytes) -> Path:
    """Atomically publish immutable bytes; never overwrite a damaged existing object."""
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{sha256(raw)}.json.gz"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=root, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        try:
            os.link(temporary, target)
        except FileExistsError:
            require(not target.is_symlink() and target.read_bytes() == raw, f"existing inventory is corrupt: {target}")
    finally:
        if temporary is not None:
            temporary.unlink()
    return target
