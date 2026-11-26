"""Legacy identity reads and portable compact run records with pinned inventories."""

from __future__ import annotations

from pathlib import Path

from .data_inventory_codec import (
    canonical_bytes,
    encode_inventory,
    publish_inventory,
    read_inventory,
    require,
    require_digest,
    sha256,
    strict_json,
)
from .telemetry import write_json_atomic

SCHEMA = "nanodllm.data_identity_storage.v2"
LEGACY_FILENAME = "data_identity.v1.json"
COMPACT_FILENAME = "data_identity.v2.json"


def identity_path(directory: Path) -> Path:
    candidates = [directory / name for name in (LEGACY_FILENAME, COMPACT_FILENAME)]
    present = [path for path in candidates if path.exists()]
    require(len(present) == 1, f"expected one data identity in {directory}, found {len(present)}")
    return present[0]


def load_data_identity(path: Path) -> dict:
    """Return the unchanged logical v1 structure, independently of its disk format."""
    path = identity_path(path) if path.is_dir() else path
    with path.open("rb") as stream:
        raw = stream.read(256 * 1024**2 + 1)
    require(len(raw) <= 256 * 1024**2, "data identity size limit exceeded")
    record = strict_json(raw)
    if record.get("schema") == SCHEMA:
        require(
            set(record) == {"schema", "identity", "dataset_inventory", "verification"},
            "unsupported compact identity fields",
        )
        reference = record["dataset_inventory"]
        require(
            isinstance(reference, dict) and set(reference) == {"sha256", "size_bytes", "path"},
            "invalid inventory reference",
        )
        digest = reference["sha256"]
        require_digest(digest)
        relative = f"data_inventory/{digest}.json.gz"
        require(reference["path"] == relative, "unsafe inventory reference")
        inventory_path = path.parent / relative
        require(
            not inventory_path.is_symlink() and not inventory_path.parent.is_symlink(), "symlinked inventory reference"
        )
        dataset, inventory_raw = read_inventory(inventory_path, digest)
        require(len(inventory_raw) == reference["size_bytes"], "inventory reference size differs")
        record = {**record["identity"], "dataset": dataset}
    require(record.get("version") == 1, "unsupported logical data identity version")
    actual = sha256(canonical_bytes({key: value for key, value in record.items() if key != "hashes"}))
    require(record.get("hashes", {}).get("identity_sha256") == actual, "data identity digest mismatch")
    return record


def write_data_identity(directory: Path, record: dict, *, compact: bool, verification: dict) -> Path:
    """Retain a portable dependency copy; existing legacy attempts keep their format."""
    present = [directory / name for name in (LEGACY_FILENAME, COMPACT_FILENAME) if (directory / name).exists()]
    require(len(present) <= 1, "ambiguous existing data identities")
    if present:
        existing = load_data_identity(present[0])
        require(
            existing["hashes"]["identity_sha256"] == record["hashes"]["identity_sha256"],
            "refusing to resume with a changed data identity",
        )
        # Preserve original bytes and their verification provenance.
        return present[0]
    if not compact:
        path = directory / LEGACY_FILENAME
        write_json_atomic(str(path), record)
        return path
    raw = encode_inventory(record["dataset"])
    inventory = publish_inventory(directory / "data_inventory", raw)
    path = directory / COMPACT_FILENAME
    write_json_atomic(
        str(path),
        {
            "schema": SCHEMA,
            "identity": {key: value for key, value in record.items() if key != "dataset"},
            "dataset_inventory": {
                "sha256": sha256(raw),
                "size_bytes": len(raw),
                "path": inventory.relative_to(directory).as_posix(),
            },
            "verification": verification,
        },
    )
    return path
