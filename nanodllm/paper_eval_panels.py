"""Prepare versioned evaluation panels from local Parquet rows or replay their exact token tensors."""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import torch

from .paper_eval_bos import prepare_panel as prepare_bos
from .paper_eval_checkpoint import stream_sha256, tokenizer_identity
from .paper_runtime.core_evaluation_cells import require
from .paper_runtime.data import DLMDataBatch, build_token_bytes, encode_document_batch
from .paper_runtime.eval_nelbo_records import _nelbo_input_identity_sha256, _ordered_sha256_root
from .paper_runtime.replication_eval_support import digest
from .release_assets import release_asset

FULL_SEEDS = (2026091101, 2026091102, 2026091103, 2026091104)
KINDS = ("packed", "bos", "dev32", "holdout256")
SEGMENT_ATTENTION = "same_segment_bidirectional_v1"
PANEL_SCHEMA = "nanodllm.paper.corpus-panel.v1"
REFERENCE_ASSET = release_asset("evaluation/panels/corpus-v1.json")
MANIFEST_SHA256 = "289fe3d6eb83a828cb4242d06a0771952bc2d0078574c3e1739ca6f924ac3082"


def reference_manifest(path=None):
    with Path(path or REFERENCE_ASSET).open("rb") as f:
        require(stream_sha256(f) == MANIFEST_SHA256, "reference manifest differs from v1")
        f.seek(0)
        return json.load(f)


def packed_root(tensors):
    fields = ("x_0", "attention_mask", "loss_mask", "position_ids", "document_start_mask")
    rows = [_nelbo_input_identity_sha256(*row) for row in zip(*(tensors[k] for k in fields), strict=True)]
    return _ordered_sha256_root(b"nanodllm-nelbo-input-identity-root-v2", rows)


@dataclass
class FrozenInputs:
    tensors: dict

    @property
    def metadata(self):
        count, width = self.tensors["x_0"].shape
        return dict(
            sequence_count=count,
            sequence_length=width,
            attention_layout=SEGMENT_ATTENTION,
            input_identity_root_sha256=packed_root(self.tensors),
        )

    def batches(self, db, device):
        count = len(self.tensors["x_0"])
        for i in range(0, count, db):
            fields = {k: v[i : i + db].to(device) for k, v in self.tensors.items()}
            segments = fields["position_ids"].eq(0).long().cumsum(dim=1)
            yield DLMDataBatch(**fields, document_boundary_mask=segments[:, :, None] == segments[:, None, :])


def pack_documents(text_batches, tokenizer, count, width):
    """Fill complete rows in source order, carrying the unconsumed suffix across row boundaries."""
    require(count > 0 and width > 0, "positive row count and width required")
    bos = tokenizer.get_bos_token_id()
    require(bos is not None and tokenizer.get_eos_token_id() is not None, "BOS and EOS required")
    tokens = torch.empty(count * width, dtype=torch.long)
    starts_flat = torch.zeros_like(tokens, dtype=torch.bool)
    cursor = 0
    for batch in text_batches:
        docs = encode_document_batch(tokenizer, batch, SimpleNamespace(prepend_bos=True, append_eos=True))
        for doc in docs:
            n = min(len(doc), tokens.numel() - cursor)
            starts_flat[cursor] = True
            tokens[cursor : cursor + n] = torch.tensor(doc[:n])
            cursor += n
            if cursor == tokens.numel():
                x = tokens.view(count, width)
                starts = starts_flat.view_as(x)
                require(torch.equal(starts, x.eq(bos)), "reserved BOS occurred inside ordinary text")
                positions = torch.arange(width).expand_as(x)
                reset = torch.where(starts, positions, 0).cummax(dim=1).values
                return dict(
                    x_0=x,
                    position_ids=positions - reset,
                    document_start_mask=starts,
                    loss_mask=~starts,
                    attention_mask=torch.ones_like(starts),
                )
    raise ValueError(f"insufficient corpus for {count} complete rows of {width} tokens")


def block_prompt(text, ordinal, tokenizer, byte_table):
    if not isinstance(text, str) or not text.strip():
        return None
    text = text.strip()
    body = tokenizer.encode(text)
    if any(byte_table[t] <= 0 for t in body):
        return None
    tokens = [tokenizer.get_bos_token_id(), *body, tokenizer.get_eos_token_id()]
    remaining = len(tokens) - 192
    group = "short" if 32 <= remaining <= 128 else "long" if remaining >= 512 else None
    if group is None:
        return None
    document = hashlib.sha256(text.encode()).hexdigest()
    gold = tokens[192:320]
    return dict(
        id="fresh-" + document,
        raw_document_index=ordinal,
        length_stratum=group,
        source_remaining_tokens=remaining,
        prefix_ids=tokens[:192],
        gold_ids=gold,
        gold_token_bytes=[byte_table[t] for t in gold],
    )


def block_root(rows):
    return digest([[r["id"], r["prefix_ids"], r["gold_ids"], r["gold_token_bytes"]] for r in rows])


def block_reference(source, tokenizer, manifest):
    ends = list(accumulate(source.metadata.row_group(i).num_rows for i in range(source.num_row_groups)))
    by_group = {}
    for ordinal, doc_hash, group, remaining in manifest["rows"]:
        rg = bisect_right(ends, ordinal)
        by_group.setdefault(rg, []).append((ordinal, doc_hash, group, remaining))
    table = build_token_bytes(tokenizer).tolist()
    by_index = {}
    for rg, addresses in by_group.items():
        texts = source.read_row_group(rg, columns=["text"]).column(0)
        start = ends[rg - 1] if rg else 0
        for ordinal, doc_hash, group, remaining in addresses:
            row = block_prompt(texts[ordinal - start].as_py(), ordinal, tokenizer, table)
            require(
                row is not None
                and row["id"] == "fresh-" + doc_hash
                and row["length_stratum"] == group
                and row["source_remaining_tokens"] == remaining,
                "reference document or tokenization differs",
            )
            by_index[ordinal] = row
    rows = [by_index[spec[0]] for spec in manifest["rows"]]
    require(block_root(rows) == manifest["targets_sha256"], "reference block targets differ")
    return rows


def verify_reference_exclusions(kind, rows, packed, exclusions):
    """Check the retained preselection against earlier source segments without releasing their tokens."""
    width = exclusions["body_tokens"]
    prefixes = set()
    for tokens, positions, starts in zip(
        packed["x_0"], packed["position_ids"], packed["document_start_mask"], strict=True
    ):
        offsets = (positions == 0).nonzero().flatten().tolist()
        for start, end in zip(offsets, offsets[1:] + [len(tokens)], strict=True):
            if starts[start] and end - start > width:
                prefixes.add(digest(tokens[start + 1 : start + 1 + width].tolist()))
    require(
        len(prefixes) == exclusions["bos_unique_prefixes"]
        and digest(sorted(prefixes)) == exclusions["bos_prefix_set_sha256"],
        "BOS exclusion prefixes changed",
    )
    early = set(exclusions["development_predictive_and_generation_prefix_sha256"])
    selected = {digest(r["prefix_ids"][1 : 1 + width]) for r in rows}
    require(
        len(selected) == len(rows) and selected.isdisjoint(prefixes), "duplicate or previously scored corpus prefix"
    )
    if kind == "holdout256":
        require(selected.isdisjoint(early), "holdout overlaps an earlier prompt prefix")
    else:
        require(selected <= early, "development prefix differs from the retained prompt panel")


def block_custom(text_batches, tokenizer, *, per_stratum, skip_per_stratum):
    """A finite, deterministic split. Reserve the first 16 of each stratum for development by default."""
    table = build_token_bytes(tokenizer).tolist()
    counts = Counter()
    rows, prefixes = [], set()
    for ordinal, text in enumerate(t for batch in text_batches for t in batch):
        row = block_prompt(text, ordinal, tokenizer, table)
        if row is None:
            continue
        group = row["length_stratum"]
        key = tuple(row["prefix_ids"][1:65])
        if key in prefixes:
            continue
        prefixes.add(key)
        if counts[group] >= skip_per_stratum + per_stratum:
            continue
        counts[group] += 1
        if counts[group] > skip_per_stratum:
            rows.append(row)
            if len(rows) == per_stratum * 2:
                return rows
    raise ValueError("insufficient distinct eligible documents for both requested length strata")


def bos_from_packed(tensors, tokenizer):
    settings = dict(
        sequence_length=tensors["x_0"].shape[1],
        device_batch=4,
        drop_prefix_tokens=128,
        conditions=["document_start"],
        dtype="float32",
        time_sampling="stratified_per_sequence",
        seeds=list(FULL_SEEDS),
    )
    return prepare_bos(FrozenInputs(tensors), tokenizer, settings, conditions=("document_start",))


def panel_digest(kind, data):
    if kind == "packed":
        return packed_root(data)
    if kind == "bos":
        return digest({k: v for k, v in data.items() if k != "panel_sha256"})
    # Strata and ordinals affect the paired bootstrap and replay audit.
    return digest(data)


def prepare(kind, parquet, tokenizer, *, reference=True, manifest_path=None, count=None, width=None, per_stratum=None):
    require(kind in KINDS, "unknown panel kind")
    if reference and any(x is not None for x in (count, width, per_stratum)):
        raise ValueError("reference panels have fixed sizes")
    if (
        kind in ("packed", "bos")
        and per_stratum is not None
        or kind in ("dev32", "holdout256")
        and (count is not None or width is not None)
    ):
        raise ValueError("size option is incompatible with the requested panel")
    ident = tokenizer_identity(tokenizer)
    with Path(parquet).open("rb") as f:
        manifest = reference_manifest(manifest_path) if reference else {}
        if reference:
            require(ident == manifest["tokenizer_sha256"], "reference tokenizer files differ")
            require(stream_sha256(f) == manifest["shard_sha256"], "reference validation shard hash differs")
            f.seek(0)
        source = pq.ParquetFile(f)
        batches = (batch.column(0).to_pylist() for batch in source.iter_batches(batch_size=128, columns=["text"]))
        require("text" in source.schema_arrow.names, "expected a Parquet text column")
        if kind in ("packed", "bos"):
            data = pack_documents(
                batches, tokenizer, 1024 if count is None else count, 1024 if width is None else width
            )
            if reference:
                require(packed_root(data) == manifest["packed_input_root"], "packed row identity differs")
            if kind == "bos":
                data = bos_from_packed(data, tokenizer)
                if reference:
                    require(data["panel_sha256"] == manifest["bos_panel_sha256"], "BOS panel identity differs")
        elif reference:
            data = block_reference(source, tokenizer, manifest["block_panels"][kind])
            packed = pack_documents(batches, tokenizer, 1024, 1024)
            verify_reference_exclusions(kind, data, packed, manifest["prefix_exclusions"])
        else:
            n = (16 if kind == "dev32" else 128) if per_stratum is None else per_stratum
            require(n > 0 and (kind != "dev32" or n <= 16), "custom development allows at most 16 per stratum")
            data = block_custom(batches, tokenizer, per_stratum=n, skip_per_stratum=16 if kind == "holdout256" else 0)
    return dict(
        schema=PANEL_SCHEMA,
        kind=kind,
        reference=reference,
        tokenizer_sha256=ident,
        panel_sha256=panel_digest(kind, data),
        data=data,
    )


def load_panel(path, tokenizer, *, manifest_path=None):
    panel = torch.load(path, weights_only=True, map_location="cpu")
    require(
        isinstance(panel, dict) and panel.get("schema") == PANEL_SCHEMA and panel.get("kind") in KINDS,
        "unknown panel schema or kind",
    )
    kind, data = panel["kind"], panel["data"]
    require(panel["tokenizer_sha256"] == tokenizer_identity(tokenizer), "panel tokenizer differs")
    require(panel["panel_sha256"] == panel_digest(kind, data), "panel replay hash differs")
    if panel["reference"]:
        manifest = reference_manifest(manifest_path)
        require(panel["tokenizer_sha256"] == manifest["tokenizer_sha256"], "reference tokenizer differs")
        if kind in ("dev32", "holdout256"):
            spec = manifest["block_panels"][kind]
            addresses = [
                [r["raw_document_index"], r["id"][6:], r["length_stratum"], r["source_remaining_tokens"]] for r in data
            ]
            require(
                addresses == spec["rows"] and block_root(data) == spec["targets_sha256"],
                "reference block panel differs",
            )
        else:
            key = "packed_input_root" if kind == "packed" else "bos_panel_sha256"
            require(panel["panel_sha256"] == manifest[key], "reference sequence panel differs")
    return panel
