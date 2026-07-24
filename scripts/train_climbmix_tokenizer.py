#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#   "blobfile==3.2.0",
#   "pyarrow==21.0.0",
#   "rustbpe==0.1.0",
#   "tiktoken==0.11.0",
#   "tokenizers==0.22.2",
#   "transformers==5.10.2",
# ]
# ///
"""Train an auditable NanoChat-style byte BPE tokenizer on ClimbMix.

The fitting recipe is adapted from karpathy/nanochat at commit
92d63d4e8bb4df75c3b71618f31ddde2378b2bcd (MIT). Unlike NanoChat's
pickle artifact, this script exports a normal Hugging Face fast tokenizer,
an ordinary-text backend that never activates control tokens, and a safe JSON
token-byte table. Torch is intentionally not required.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import rustbpe
import tiktoken
from tokenizers import Tokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers.integrations.tiktoken import convert_tiktoken_to_fast

CLIMBMIX_REPO_ID = "karpathy/climbmix-400b-shuffle"
CLIMBMIX_REVISION = "915333b4f8b8684f39aeaafea600fea6f43fb703"
CLIMBMIX_LICENSE = "MIT"
NANOCHAT_REPO_ID = "karpathy/nanochat"
NANOCHAT_REVISION = "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd"
SCRIPT_RELATIVE_PATH = "scripts/train_climbmix_tokenizer.py"
SPLIT_PATTERN = (
    r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}|"
    r" ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
)
SPECIAL_TOKENS = [
    "<|bos|>",
    "<|eos|>",
    "<|mask|>",
    "<|pad|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
]
GUARDRAILS = [
    "Hello world! Numbers: 123, 4567, 89. I'm testing contractions.\n",
    "def fib(n: int) -> int:\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n",
    "For $x \\in \\mathbb{R}$, solve x^2 + 2x + 1 = 0.",
    "Unicode: 你好世界 안녕하세요 مرحبا नमस्ते 🌍 e\u0301 é",
    "Whitespace:\t  leading\n\ntrailing  \r\n",
    "Literal controls stay text: <|mask|> <|eos|> <|assistant_start|>",
]
SHARD_PATTERN = re.compile(r"^shard_(\d{5})\.parquet$")
HASH_CHUNK_BYTES = 8 * 1024 * 1024


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_canonical_json_bytes(value) + b"\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    names = ["blobfile", "pyarrow", "rustbpe", "tiktoken", "tokenizers", "transformers"]
    return {name: importlib.metadata.version(name) for name in names}


def _resolve_corpus(
    data_dir: Path,
    validation_shard: str,
    expected_first_shard: int,
    expected_last_shard: int,
    dataset_revision: str,
) -> tuple[list[Path], Path, dict[str, Any]]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"ClimbMix directory does not exist: {data_dir}")
    indexed: list[tuple[int, Path]] = []
    for path in data_dir.iterdir():
        match = SHARD_PATTERN.fullmatch(path.name)
        if match and path.is_file():
            indexed.append((int(match.group(1)), path.resolve()))
    indexed.sort()
    expected_ids = list(range(expected_first_shard, expected_last_shard + 1))
    actual_ids = [index for index, _path in indexed]
    if actual_ids != expected_ids:
        missing = sorted(set(expected_ids) - set(actual_ids))
        unexpected = sorted(set(actual_ids) - set(expected_ids))
        raise ValueError(
            "ClimbMix shard inventory mismatch: "
            f"missing={missing[:20]} ({len(missing)} total), "
            f"unexpected={unexpected[:20]} ({len(unexpected)} total)"
        )
    by_name = {path.name: path for _index, path in indexed}
    if validation_shard not in by_name:
        raise FileNotFoundError(f"validation shard is absent: {validation_shard}")
    validation_path = by_name[validation_shard]
    train_paths = [path for _index, path in indexed if path != validation_path]
    if not train_paths:
        raise ValueError("validation reservation left no tokenizer-training shards")
    records = []
    total_bytes = 0
    for index, path in indexed:
        stat = path.stat()
        total_bytes += stat.st_size
        records.append(
            {
                "index": index,
                "name": path.name,
                "size_bytes": stat.st_size,
            }
        )
    listing_material = {
        "format": "nanodllm.climbmix_inventory.v1",
        "source": {
            "provider": "huggingface",
            "repo_type": "dataset",
            "repo_id": CLIMBMIX_REPO_ID,
            "revision_declared": dataset_revision,
            "license_declared_by_dataset_card": CLIMBMIX_LICENSE,
        },
        "validation_shard": validation_shard,
        "files": records,
    }
    listing_material["listing_sha256"] = hashlib.sha256(_canonical_json_bytes(listing_material)).hexdigest()
    listing_material["file_count"] = len(records)
    listing_material["total_size_bytes"] = total_bytes
    return train_paths, validation_path, listing_material


class FitStream:
    def __init__(
        self,
        paths: list[Path],
        *,
        text_column: str,
        max_chars: int,
        document_char_cap: int,
        parity_documents: int,
    ):
        self.paths = paths
        self.text_column = text_column
        self.max_chars = max_chars
        self.document_char_cap = document_char_cap
        self.parity_document_limit = parity_documents
        self.started = False
        self.complete = False
        self.documents = 0
        self.codepoints = 0
        self.utf8_bytes = 0
        self.sample_hasher = hashlib.sha256(b"nanodllm.climbmix_tokenizer_fit.v1\0")
        self.per_shard: dict[str, dict[str, int]] = {}
        self.terminal: dict[str, Any] | None = None
        self.parity_documents: list[str] = []

    def __iter__(self):
        if self.started:
            raise RuntimeError("the deterministic tokenizer fit stream may only be consumed once")
        self.started = True
        for path in self.paths:
            parquet_file = pq.ParquetFile(path)
            if self.text_column not in parquet_file.schema_arrow.names:
                raise ValueError(f"{path} does not contain text column {self.text_column!r}")
            shard_stats = {"documents": 0, "codepoints": 0, "utf8_bytes": 0}
            self.per_shard[path.name] = shard_stats
            for row_group in range(parquet_file.num_row_groups):
                table = parquet_file.read_row_group(row_group, columns=[self.text_column])
                for row_index, document in enumerate(table.column(self.text_column).to_pylist()):
                    if not isinstance(document, str):
                        raise TypeError(f"non-string document in {path.name}, row_group={row_group}, row={row_index}")
                    cropped = document[: self.document_char_cap]
                    raw = cropped.encode("utf-8")
                    self.documents += 1
                    self.codepoints += len(cropped)
                    self.utf8_bytes += len(raw)
                    shard_stats["documents"] += 1
                    shard_stats["codepoints"] += len(cropped)
                    shard_stats["utf8_bytes"] += len(raw)
                    self.sample_hasher.update(len(raw).to_bytes(8, "big"))
                    self.sample_hasher.update(raw)
                    self.terminal = {
                        "shard": path.name,
                        "row_group": row_group,
                        "row_index_in_row_group": row_index,
                    }
                    if len(self.parity_documents) < self.parity_document_limit:
                        self.parity_documents.append(cropped)
                    yield cropped
                    # This deliberately matches NanoChat's first-crossing rule.
                    if self.codepoints > self.max_chars:
                        self.complete = True
                        return
        raise RuntimeError(
            f"tokenizer fit stream ended at {self.codepoints:,} characters before crossing {self.max_chars:,}"
        )

    def record(self) -> dict[str, Any]:
        if not self.complete or self.terminal is None:
            raise RuntimeError("tokenizer trainer did not consume the complete deterministic fit stream")
        return {
            "selection": "sorted training shards, stored row-group order, stored row order",
            "stop_rule": "stop after the first complete capped document for which cumulative codepoints > max_chars",
            "max_chars": self.max_chars,
            "document_char_cap": self.document_char_cap,
            "documents": self.documents,
            "codepoints": self.codepoints,
            "utf8_bytes": self.utf8_bytes,
            "sample_sha256": self.sample_hasher.hexdigest(),
            "sample_hash_framing": "sha256(domain || uint64_be(utf8_length) || utf8_bytes for each document)",
            "terminal": self.terminal,
            "per_shard": self.per_shard,
        }


def _source_record(path: Path) -> dict[str, Any]:
    parquet_file = pq.ParquetFile(path)
    metadata = parquet_file.metadata
    if metadata is None:
        raise ValueError(f"Parquet file has no footer metadata: {path}")
    return {
        "name": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "schema": str(parquet_file.schema_arrow),
        "rows": metadata.num_rows,
        "row_groups": metadata.num_row_groups,
    }


def _ordinary_tokenizer_file(converted_path: Path, ordinary_path: Path) -> None:
    payload = json.loads(converted_path.read_text(encoding="utf-8"))
    payload["added_tokens"] = []
    _write_json(ordinary_path, payload)


def _gpt2_byte_decoder() -> dict[str, int]:
    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(ord("¡"), ord("¬") + 1))
    byte_values += list(range(ord("®"), ord("ÿ") + 1))
    codepoints = list(byte_values)
    extension = 0
    for byte_value in range(256):
        if byte_value not in byte_values:
            byte_values.append(byte_value)
            codepoints.append(256 + extension)
            extension += 1
    return {chr(codepoint): byte_value for byte_value, codepoint in zip(byte_values, codepoints, strict=True)}


def _validate_export(
    staging: Path,
    encoding: tiktoken.Encoding,
    expected_special_ids: dict[str, int],
    probes: list[str],
) -> dict[str, Any]:
    hf = AutoTokenizer.from_pretrained(
        staging,
        use_fast=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    ordinary = Tokenizer.from_file(str(staging / "tokenizer_ordinary.json"))
    if len(hf) != encoding.n_vocab:
        raise ValueError(f"exported vocabulary has {len(hf)} entries, expected {encoding.n_vocab}")
    observed_special_ids = {token: int(hf.convert_tokens_to_ids(token)) for token in SPECIAL_TOKENS}
    if observed_special_ids != expected_special_ids:
        raise ValueError(
            f"exported special-token IDs changed: expected={expected_special_ids}, observed={observed_special_ids}"
        )
    hf_base_vocab = hf.backend_tokenizer.get_vocab(with_added_tokens=False)
    ordinary_base_vocab = ordinary.get_vocab(with_added_tokens=False)
    if hf_base_vocab != ordinary_base_vocab:
        raise ValueError("HF and ordinary backends have different base token-to-ID mappings")
    if len(ordinary_base_vocab) != encoding.n_vocab - len(expected_special_ids):
        raise ValueError("ordinary backend base vocabulary has an unexpected size")
    tokens_by_id = [None] * len(ordinary_base_vocab)
    for token, token_id in ordinary_base_vocab.items():
        if token_id < 0 or token_id >= len(tokens_by_id) or tokens_by_id[token_id] is not None:
            raise ValueError("ordinary backend base vocabulary IDs are not contiguous and unique")
        tokens_by_id[token_id] = token
    byte_decoder = _gpt2_byte_decoder()
    for token_id, token in enumerate(tokens_by_id):
        if token is None or any(character not in byte_decoder for character in token):
            raise ValueError(f"ordinary backend token {token_id} is not a valid byte-level token")
        observed_bytes = bytes(byte_decoder[character] for character in token)
        if observed_bytes != encoding.decode_single_token_bytes(token_id):
            raise ValueError(f"ordinary backend raw bytes differ from tiktoken at token ID {token_id}")
    for index, text in enumerate(probes):
        reference_ids = encoding.encode_ordinary(text)
        ordinary_ids = ordinary.encode(text, add_special_tokens=False).ids
        if reference_ids != ordinary_ids:
            raise ValueError(f"reference/HF ordinary token IDs differ for parity probe {index}")
        if encoding.decode(reference_ids) != text:
            raise ValueError(f"reference tokenizer round-trip failed for parity probe {index}")
        if ordinary.decode(ordinary_ids, skip_special_tokens=False) != text:
            raise ValueError(f"HF ordinary tokenizer round-trip failed for parity probe {index}")
    return {
        "probe_count": len(probes),
        "reference_hf_token_id_parity": True,
        "reference_roundtrip": True,
        "hf_ordinary_roundtrip": True,
        "literal_control_tokens_included": True,
        "exhaustive_base_vocab_id_parity": True,
        "exhaustive_mergeable_token_bytes_parity": True,
        "mergeable_tokens_checked": len(ordinary_base_vocab),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and validate a Hugging Face byte-BPE tokenizer on a deterministic ClimbMix prefix.",
        epilog=(
            f"Corpus: https://huggingface.co/datasets/{CLIMBMIX_REPO_ID} "
            f"at revision {CLIMBMIX_REVISION}. See docs/tokenizer-training.md."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing only the expected shard_NNNNN.parquet inventory (other file types are ignored).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory for the HF tokenizer and build records; an existing path is never overwritten.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32_768,
        help="Total vocabulary size, including the reserved special tokens.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2_000_000_000,
        help="Fit-sample threshold in Unicode code points; the complete document that first crosses it is included.",
    )
    parser.add_argument(
        "--document-char-cap",
        type=int,
        default=10_000,
        help="Maximum Unicode code points retained from the start of each document.",
    )
    parser.add_argument("--text-column", default="text", help="Parquet column containing document strings.")
    parser.add_argument(
        "--validation-shard",
        default="shard_06542.parquet",
        help="Shard reserved from tokenizer fitting and recorded as validation data.",
    )
    parser.add_argument(
        "--expected-first-shard",
        type=int,
        default=0,
        help="First inclusive shard index required in data-dir.",
    )
    parser.add_argument(
        "--expected-last-shard",
        type=int,
        default=6542,
        help="Last inclusive shard index required in data-dir.",
    )
    parser.add_argument(
        "--dataset-revision",
        default=CLIMBMIX_REVISION,
        help=(
            "Hugging Face dataset revision recorded as declared provenance. "
            "Local files are verified independently by inventory and consumed-file hashes."
        ),
    )
    parser.add_argument(
        "--model-max-length",
        type=int,
        default=2048,
        help="model_max_length written to the exported Hugging Face tokenizer configuration.",
    )
    parser.add_argument(
        "--parity-documents",
        type=int,
        default=256,
        help="Number of consumed documents retained in memory for tiktoken/HF parity checks.",
    )
    parser.add_argument(
        "--nanochat-revision",
        default=NANOCHAT_REVISION,
        help="NanoChat source revision recorded for the adapted fitting recipe.",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = _parse_args()
    if args.vocab_size < 256 + len(SPECIAL_TOKENS):
        raise ValueError("vocab size is too small for the 256 byte tokens and reserved specials")
    if args.max_chars <= 0 or args.document_char_cap <= 0:
        raise ValueError("max chars and document character cap must be positive")
    if args.model_max_length <= 0:
        raise ValueError("model max length must be positive")
    if args.parity_documents < 0:
        raise ValueError("parity documents must be non-negative")
    if args.expected_first_shard < 0 or args.expected_last_shard < args.expected_first_shard:
        raise ValueError("expected shard bounds must be non-negative and ordered")
    validation_match = SHARD_PATTERN.fullmatch(args.validation_shard)
    if validation_match is None:
        raise ValueError("validation shard must have the form shard_NNNNN.parquet")
    validation_index = int(validation_match.group(1))
    if not args.expected_first_shard <= validation_index <= args.expected_last_shard:
        raise ValueError("validation shard must fall within the expected shard range")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing tokenizer directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.building-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"staging directory already exists: {staging}")
    staging.mkdir()

    train_paths, validation_path, corpus_inventory = _resolve_corpus(
        args.data_dir,
        args.validation_shard,
        args.expected_first_shard,
        args.expected_last_shard,
        args.dataset_revision,
    )
    _write_json(staging / "corpus_inventory.v1.json", corpus_inventory)

    fit_stream = FitStream(
        train_paths,
        text_column=args.text_column,
        max_chars=args.max_chars,
        document_char_cap=args.document_char_cap,
        parity_documents=args.parity_documents,
    )
    mergeable_vocab_size = args.vocab_size - len(SPECIAL_TOKENS)
    trainer = rustbpe.Tokenizer()
    started = time.monotonic()
    trainer.train_from_iterator(iter(fit_stream), mergeable_vocab_size, pattern=SPLIT_PATTERN)
    train_seconds = time.monotonic() - started
    fit_record = fit_stream.record()

    mergeable_ranks = {bytes(token): int(rank) for token, rank in trainer.get_mergeable_ranks()}
    if len(mergeable_ranks) != mergeable_vocab_size:
        raise ValueError(f"RustBPE returned {len(mergeable_ranks)} mergeable tokens, expected {mergeable_vocab_size}")
    if sorted(mergeable_ranks.values()) != list(range(mergeable_vocab_size)):
        raise ValueError("RustBPE returned non-contiguous mergeable token ranks")
    if any(bytes([byte]) not in mergeable_ranks for byte in range(256)):
        raise ValueError("RustBPE output does not contain all 256 base byte tokens")
    trained_pattern = trainer.get_pattern()
    if trained_pattern != SPLIT_PATTERN:
        raise ValueError("RustBPE returned a different pretokenization pattern")
    special_ids = {token: mergeable_vocab_size + index for index, token in enumerate(SPECIAL_TOKENS)}
    encoding = tiktoken.Encoding(
        name=f"nanodllm-climbmix-{args.vocab_size}",
        pat_str=SPLIT_PATTERN,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_ids,
        explicit_n_vocab=args.vocab_size,
    )
    if encoding.n_vocab != args.vocab_size:
        raise ValueError(f"tiktoken vocabulary has {encoding.n_vocab} entries, expected {args.vocab_size}")

    convert_tiktoken_to_fast(encoding, str(staging))
    converted_path = staging / "tokenizer.json"
    _ordinary_tokenizer_file(converted_path, staging / "tokenizer_ordinary.json")
    hf = PreTrainedTokenizerFast(
        tokenizer_file=str(converted_path),
        bos_token=SPECIAL_TOKENS[0],
        eos_token=SPECIAL_TOKENS[1],
        mask_token=SPECIAL_TOKENS[2],
        pad_token=SPECIAL_TOKENS[3],
        additional_special_tokens=SPECIAL_TOKENS[4:],
        model_max_length=args.model_max_length,
        clean_up_tokenization_spaces=False,
    )
    hf.save_pretrained(staging)

    token_bytes = [len(encoding.decode_single_token_bytes(token_id)) for token_id in range(mergeable_vocab_size)]
    token_bytes.extend([0] * len(SPECIAL_TOKENS))
    _write_json(
        staging / "token_bytes.json",
        {
            "format": "nanodllm.token_bytes.v1",
            "dtype": "int32",
            "vocab_size": args.vocab_size,
            "special_token_ids": list(special_ids.values()),
            "values": token_bytes,
        },
    )

    parity = _validate_export(
        staging,
        encoding,
        special_ids,
        fit_stream.parity_documents + GUARDRAILS,
    )
    consumed_names = set(fit_record["per_shard"])
    consumed_sources = [_source_record(path) for path in train_paths if path.name in consumed_names]
    validation_source = _source_record(validation_path)

    artifact_records = []
    for path in sorted(staging.rglob("*")):
        if path.is_file():
            artifact_records.append(
                {
                    "relative_path": path.relative_to(staging).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    manifest = {
        "format": "nanodllm.tokenizer_build.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "invocation": {
            "entrypoint": SCRIPT_RELATIVE_PATH,
            "path_arguments_recorded": False,
            "arguments": {
                "vocab_size": args.vocab_size,
                "max_chars": args.max_chars,
                "document_char_cap": args.document_char_cap,
                "text_column": args.text_column,
                "validation_shard": args.validation_shard,
                "expected_first_shard": args.expected_first_shard,
                "expected_last_shard": args.expected_last_shard,
                "dataset_revision": args.dataset_revision,
                "model_max_length": args.model_max_length,
                "parity_documents": args.parity_documents,
                "nanochat_revision": args.nanochat_revision,
            },
        },
        "implementation": {
            "script": SCRIPT_RELATIVE_PATH,
            "script_sha256": _sha256_file(Path(__file__).resolve()),
        },
        "recipe": {
            "upstream": NANOCHAT_REPO_ID,
            "upstream_revision_declared": args.nanochat_revision,
            "adaptation": "NanoChat RustBPE recipe with explicit MDLM and future chat/tool specials",
            "algorithm": "regex-pretokenized byte-level BPE",
            "split_pattern": SPLIT_PATTERN,
            "split_pattern_sha256": hashlib.sha256(SPLIT_PATTERN.encode("utf-8")).hexdigest(),
            "normalization": "none",
            "vocab_size": args.vocab_size,
            "mergeable_vocab_size": mergeable_vocab_size,
            "special_tokens": SPECIAL_TOKENS,
            "special_token_ids": special_ids,
            "model_max_length": args.model_max_length,
        },
        "corpus": {
            "inventory_file": "corpus_inventory.v1.json",
            "inventory_sha256": _sha256_file(staging / "corpus_inventory.v1.json"),
            "validation_reservation": validation_source,
            "consumed_source_shards": consumed_sources,
        },
        "fit_sample": fit_record,
        "training": {
            "seconds": train_seconds,
            "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS"),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        },
        "validation": parity,
        "environment": {
            "python": ".".join(str(part) for part in sys.version_info[:3]),
            "packages": _package_versions(),
        },
        "artifacts": artifact_records,
    }
    _write_json(staging / "tokenizer_build.v1.json", manifest)
    shutil.move(str(staging), str(output_dir))
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "vocab_size": args.vocab_size,
                "fit_documents": fit_record["documents"],
                "fit_codepoints": fit_record["codepoints"],
                "fit_utf8_bytes": fit_record["utf8_bytes"],
                "fit_sha256": fit_record["sample_sha256"],
                "terminal": fit_record["terminal"],
                "train_seconds": train_seconds,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
