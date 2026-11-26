"""Admission and atomic publication of local CLI report files."""

import json
import os
from itertools import chain
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch

from .evaluation.checkpoint import TOKENIZER_FILES
from .runtime.data_identity_io import COMPACT_FILENAME, LEGACY_FILENAME, SCHEMA
from .runtime.data_inventory_codec import require_digest, strict_json


def _identity_files(checkpoint, data_identity):
    if data_identity is None or data_identity.is_dir():
        directory = checkpoint.parent if data_identity is None else data_identity
        sidecars = (directory / COMPACT_FILENAME, directory / LEGACY_FILENAME)
    else:
        sidecars = (data_identity,)
    yield from sidecars
    for path in sidecars:
        if path.is_file() and (path.parent / "data_inventory").is_dir():
            with path.open("rb") as stream:
                raw = stream.read(256 * 1024**2 + 1)
            if len(raw) > 256 * 1024**2:
                raise ValueError("data identity size limit exceeded")
            record = strict_json(raw)
            if record.get("schema") == SCHEMA:
                reference = record["dataset_inventory"]
                digest = reference["sha256"]
                require_digest(digest)
                relative = f"data_inventory/{digest}.json.gz"
                if reference["path"] != relative:
                    raise ValueError("unsafe inventory reference")
                yield path.parent / relative


def _tokenizer_files(root):
    if root is None:
        return
    yield from (root / name for name in TOKENIZER_FILES)
    yield root / "chat_template.jinja"
    yield from (root / "chat_templates").glob("*.jinja")
    config = root / "tokenizer_config.json"
    if config.is_file():
        settings = json.loads(config.read_bytes())
        tokenizer_class = settings.get("tokenizer_class")
        if tokenizer_class:
            from transformers.models.auto.tokenization_auto import tokenizer_class_from_name

            fast_class = tokenizer_class.removesuffix("Fast") + "Fast"
            backend = tokenizer_class_from_name(fast_class) or tokenizer_class_from_name(tokenizer_class)
        else:
            # AutoTokenizer may choose a model-specific vocabulary from config.json.
            from transformers import AutoTokenizer

            backend = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
        if backend:
            yield from (root / name for name in backend.vocab_files_names.values())
        versions = settings.get("fast_tokenizer_files")
        if versions:
            from transformers.tokenization_utils_base import get_fast_tokenizer_file

            yield root / get_fast_tokenizer_file(versions)


def _reference_files(root):
    if root is None or not root.is_dir():
        return
    # score_outputs hashes these top-level files, including existing JSON reports.
    yield from (
        p for p in root.iterdir() if p.suffix in {".json", ".model", ".safetensors", ".bin", ".txt", ".tiktoken"}
    )
    yield from _tokenizer_files(root)
    # HF sharded checkpoints can name weights in subdirectories. Follow the index
    # of the selected format without traversing unrelated directories.
    for name in ("model.safetensors", "pytorch_model.bin"):
        if (root / name).is_file():
            break
        index = root / (name + ".index.json")
        if index.is_file():
            shards = json.loads(index.read_bytes())["weight_map"].values()
            yield from (root / shard for shard in set(shards))
            break


def check_output(
    parser,
    output,
    *,
    files=(),
    checkpoint=None,
    training=False,
    data_identity=None,
    tokenizer=None,
    reference_model=None,
):
    """Reject input aliases and return the checked destination for writers that follow output symlinks."""
    if output is None:
        return
    target = output.resolve()
    if target.is_dir():
        parser.error(f"--output must name a file: {output}")
    exists = target.exists()
    for path in chain(files, (checkpoint, data_identity)):
        if path is not None and (target == path.resolve() or (exists and path.exists() and target.samefile(path))):
            parser.error(f"--output must differ from input file: {path}")
    if exists:
        identities = _identity_files(checkpoint, data_identity) if training and tokenizer is not None else ()
        for path in chain(identities, _tokenizer_files(tokenizer), _reference_files(reference_model)):
            if path.is_file() and target.samefile(path):
                parser.error(f"--output must differ from input file: {path}")
    return target


def save_result(path, value, *, binary=False):
    """Expose either a complete output file or no new result if serialization fails."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, mode="wb" if binary else "w", delete=False) as f:
        try:
            if binary:
                torch.save(value, f)
            else:
                json.dump(value, f, indent=2, allow_nan=False)
                f.write("\n")
            f.close()
            os.replace(f.name, path)
        finally:
            Path(f.name).unlink(missing_ok=True)
