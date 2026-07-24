#!/usr/bin/env python3
"""Build the deterministic Parquet fixture used by the CPU smoke recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPOSITORY_ROOT / "examples" / "tiny_corpus.txt"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "data" / "tiny_corpus.parquet"


def read_documents(path: Path) -> list[str]:
    """Treat each non-empty UTF-8 line as one document."""

    documents = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(documents) < 2:
        raise ValueError(f"{path} must contain at least two non-empty document lines")
    return documents


def corpus_sha256(documents: list[str]) -> str:
    digest = hashlib.sha256(b"nanodllm.tiny_corpus.v1\0")
    for document in documents:
        encoded = document.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def write_parquet(path: Path, documents: list[str], row_group_size: int) -> None:
    if row_group_size <= 0:
        raise ValueError("row_group_size must be positive")
    content_sha256 = corpus_sha256(documents)
    schema = pa.schema(
        [pa.field("text", pa.string(), nullable=False)],
        metadata={
            b"nanodllm.format": b"tiny_corpus.v1",
            b"nanodllm.content_sha256": content_sha256.encode("ascii"),
        },
    )
    table = pa.Table.from_arrays([pa.array(documents, type=pa.string())], schema=schema)
    pq.write_table(
        table,
        path,
        version="2.6",
        compression="NONE",
        use_dictionary=False,
        write_statistics=True,
        data_page_version="1.0",
        row_group_size=row_group_size,
    )


def prepare_corpus(
    input_path: Path, output_path: Path, row_group_size: int, *, force: bool = False
) -> dict[str, object]:
    documents = read_documents(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        write_parquet(temporary_path, documents, row_group_size)
        generated_bytes = temporary_path.read_bytes()
        status = "created"
        if output_path.exists():
            if output_path.read_bytes() == generated_bytes:
                status = "unchanged"
                temporary_path.unlink()
            elif not force:
                raise FileExistsError(
                    f"refusing to replace different output at {output_path}; pass --force to overwrite it"
                )
            else:
                os.replace(temporary_path, output_path)
                status = "replaced"
        else:
            os.replace(temporary_path, output_path)
        return {
            "content_sha256": corpus_sha256(documents),
            "documents": len(documents),
            "output": str(output_path),
            "parquet_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
            "row_group_size": row_group_size,
            "status": status,
        }
    finally:
        temporary_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--row-group-size", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = prepare_corpus(args.input, args.output, args.row_group_size, force=args.force)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
