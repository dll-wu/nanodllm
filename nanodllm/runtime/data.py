import json
import os
import random
from dataclasses import dataclass

import torch

from .data_identity import resolve_parquet_files as resolve_parquet_files
from .data_identity import resolve_parquet_selection as resolve_parquet_selection
from .data_parquet import ParquetTextSource as ParquetTextSource
from .data_parquet import RowGroupRef as RowGroupRef
from .data_parquet import _normalize_patterns as _normalize_patterns
from .data_parquet import _resolve_parquet_files as _resolve_parquet_files
from .data_parquet import _row_groups_for_files as _row_groups_for_files
from .data_parquet import _split_row_groups as _split_row_groups

MAX_SEQ_LEN = 2048
SYNTHETIC_DATASET_EOS_OVERFIT_CANARY = "eos_overfit_canary"
EOS_OVERFIT_CANARY_DOCUMENTS = ("A", "AA", "AAA", "AAAA")
EOS_OVERFIT_CANARY_DOCUMENTS_UNIQUE_NON_EOS = ("A", "BC", "DEF", "GHIJ", "K", "LM", "NOP", "QRST")
_EOS_OVERFIT_CANARY_DOCUMENT_LENGTHS = (1, 2, 3, 4, 1, 2, 3, 4)


def _build_eos_overfit_canary_documents_for_non_eos_token_count(
    unique_token_count: int,
    document_lengths: tuple[int, ...] | list[int] | None = None,
) -> tuple[str, ...]:
    if unique_token_count <= 0:
        raise ValueError("unique_token_count must be a positive integer")
    if unique_token_count > 62:
        raise ValueError("unique_token_count must be <= 62")
    lengths = tuple(document_lengths) if document_lengths else _EOS_OVERFIT_CANARY_DOCUMENT_LENGTHS
    if len(lengths) == 0:
        raise ValueError("document_lengths must be non-empty")
    token_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    token_pool = tuple(token_alphabet[:unique_token_count])
    tokens = [token_pool[idx % unique_token_count] for idx in range(sum(lengths))]
    start = 0
    documents: list[str] = []
    for length in lengths:
        documents.append("".join(tokens[start : start + length]))
        start += length
    return tuple(documents)


@dataclass
class DLMDataBatch:
    x_0: torch.Tensor
    loss_mask: torch.Tensor
    attention_mask: torch.Tensor
    epoch: int = 0
    position_ids: torch.Tensor | None = None
    document_boundary_mask: torch.Tensor | None = None
    document_start_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        assert self.x_0.dtype == torch.long
        assert self.loss_mask.dtype == torch.bool
        assert self.attention_mask.dtype == torch.bool
        assert self.x_0.shape == self.loss_mask.shape == self.attention_mask.shape
        if self.position_ids is not None:
            assert self.position_ids.dtype == torch.long
            assert self.position_ids.shape == self.x_0.shape
        if self.document_boundary_mask is not None:
            assert self.document_boundary_mask.dtype == torch.bool
            b, t = self.x_0.shape
            assert self.document_boundary_mask.shape == (b, t, t)
        if self.document_start_mask is not None:
            assert self.document_start_mask.dtype == torch.bool
            assert self.document_start_mask.shape == self.x_0.shape


class ByteTokenizer:
    BOS_ID = 256
    MASK_ID = 257
    EOS_ID = 258
    PAD_ID = 259
    VOCAB_SIZE = 260

    def get_vocab_size(self) -> int:
        return self.VOCAB_SIZE

    def get_mask_token_id(self) -> int:
        return self.MASK_ID

    def get_bos_token_id(self) -> int:
        return self.BOS_ID

    def get_eos_token_id(self) -> int:
        return self.EOS_ID

    def get_pad_token_id(self) -> int:
        return self.PAD_ID

    def get_special_token_ids(self) -> set[int]:
        return {self.BOS_ID, self.MASK_ID, self.EOS_ID, self.PAD_ID}

    def get_generation_blocked_token_ids(self) -> set[int]:
        return {self.BOS_ID, self.MASK_ID, self.PAD_ID}

    def encode(self, text):
        if isinstance(text, str):
            return list(text.encode("utf-8"))
        if isinstance(text, list):
            return [list(t.encode("utf-8")) for t in text]
        raise ValueError(f"invalid input type: {type(text)}")

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        """Return UTF-8 byte IDs with offsets in Python character coordinates."""
        if not isinstance(text, str):
            raise ValueError("offset encoding requires a single text string")
        offsets = [(index, index + 1) for index, char in enumerate(text) for _ in char.encode("utf-8")]
        return self.encode(text), offsets

    def decode(self, ids) -> str:
        ids = list(ids)
        if self.EOS_ID in ids:
            ids = ids[: ids.index(self.EOS_ID)]
        raw = bytes(i for i in ids if 0 <= i <= 255)
        return raw.decode("utf-8", errors="replace")


class PretrainedHFTokenizer:
    def __init__(self, tokenizer, tokenizer_path: str):
        self._tok = tokenizer
        self._tokenizer_path = tokenizer_path
        ordinary_path = os.path.join(tokenizer_path, "tokenizer_ordinary.json")
        self._ordinary_tok = None
        if os.path.exists(ordinary_path):
            from tokenizers import Tokenizer

            self._ordinary_tok = Tokenizer.from_file(ordinary_path)
            hf_backend = getattr(self._tok, "backend_tokenizer", None)
            if hf_backend is None:
                raise ValueError(
                    f"tokenizer at {tokenizer_path!r} has tokenizer_ordinary.json but no fast-tokenizer backend"
                )
            hf_base_vocab = hf_backend.get_vocab(with_added_tokens=False)
            ordinary_base_vocab = self._ordinary_tok.get_vocab(with_added_tokens=False)
            if ordinary_base_vocab != hf_base_vocab:
                raise ValueError(
                    f"tokenizer_ordinary.json at {tokenizer_path!r} does not have the same base token-to-ID mapping"
                )
        if self._tok.mask_token_id is None:
            raise ValueError(
                f"tokenizer at {tokenizer_path!r} must define mask_token_id; "
                "adapt or train the tokenizer before using it for MDLM training"
            )
        if self._tok.eos_token_id is None:
            raise ValueError(f"tokenizer at {tokenizer_path!r} must define eos_token_id")

    @classmethod
    def from_pretrained(cls, tokenizer_path: str | os.PathLike):
        from transformers import AutoTokenizer

        tokenizer_path = os.fspath(tokenizer_path)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                backend="tokenizers",
                use_fast=True,
                local_files_only=True,
                trust_remote_code=False,
            )
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                use_fast=True,
                local_files_only=True,
                trust_remote_code=False,
            )
        return cls(tokenizer, tokenizer_path)

    def get_vocab_size(self) -> int:
        return len(self._tok)

    def get_mask_token_id(self) -> int:
        return int(self._tok.mask_token_id)

    def get_eos_token_id(self) -> int:
        return int(self._tok.eos_token_id)

    def get_pad_token_id(self) -> int | None:
        return int(self._tok.pad_token_id) if self._tok.pad_token_id is not None else None

    def get_bos_token_id(self) -> int | None:
        return int(self._tok.bos_token_id) if self._tok.bos_token_id is not None else None

    def get_special_token_ids(self) -> set[int]:
        return {int(i) for i in self._tok.all_special_ids if i is not None}

    def get_generation_blocked_token_ids(self) -> set[int]:
        blocked = self.get_special_token_ids()
        blocked.discard(self.get_eos_token_id())
        return blocked

    def encode(self, text):
        if self._ordinary_tok is not None:
            if isinstance(text, str):
                return self._ordinary_tok.encode(text, add_special_tokens=False).ids
            if isinstance(text, list):
                return [encoding.ids for encoding in self._ordinary_tok.encode_batch(text, add_special_tokens=False)]
            raise ValueError(f"invalid input type: {type(text)}")
        if isinstance(text, str):
            return self._tok.encode(text, add_special_tokens=False)
        if isinstance(text, list):
            enc = self._tok(text, add_special_tokens=False, return_attention_mask=False)
            return enc["input_ids"]
        raise ValueError(f"invalid input type: {type(text)}")

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        """Use the same ordinary-token path as encode, preserving character offsets."""
        if not isinstance(text, str):
            raise ValueError("offset encoding requires a single text string")
        if self._ordinary_tok is not None:
            encoded = self._ordinary_tok.encode(text, add_special_tokens=False)
            return encoded.ids, encoded.offsets
        encoded = self._tok(text, add_special_tokens=False, return_offsets_mapping=True, return_attention_mask=False)
        return encoded["input_ids"], encoded["offset_mapping"]

    def decode(self, ids) -> str:
        ids = list(ids)
        eos_id = self.get_eos_token_id()
        if eos_id in ids:
            ids = ids[: ids.index(eos_id)]
        return self._tok.decode(ids, skip_special_tokens=True)


def make_tokenizer(name: str, tokenizer_path: str | os.PathLike | None = None):
    if name == "byte":
        return ByteTokenizer()
    if name == "hf":
        if tokenizer_path is None:
            raise ValueError("tokenizer_path is required when data.tokenizer is 'hf'")
        return PretrainedHFTokenizer.from_pretrained(tokenizer_path)
    raise ValueError(f"unknown tokenizer name: {name!r}")


def build_tokenizer(config):
    return make_tokenizer(config.data.tokenizer, config.data.tokenizer_path)


def build_token_bytes(tokenizer) -> torch.Tensor:
    vocab_size = tokenizer.get_vocab_size()
    mask_id = tokenizer.get_mask_token_id()
    if isinstance(tokenizer, ByteTokenizer):
        return torch.tensor([1 if i < 256 else 0 for i in range(vocab_size)], dtype=torch.int32)
    if isinstance(tokenizer, PretrainedHFTokenizer):
        token_bytes_json_path = os.path.join(tokenizer._tokenizer_path, "token_bytes.json")
        if os.path.exists(token_bytes_json_path):
            with open(token_bytes_json_path, encoding="utf-8") as handle:
                payload = json.load(handle)
            if payload.get("format") != "nanodllm.token_bytes.v1":
                raise ValueError(f"unsupported token byte table format in {token_bytes_json_path!r}")
            if payload.get("dtype") != "int32" or payload.get("vocab_size") != vocab_size:
                raise ValueError(f"token byte table metadata in {token_bytes_json_path!r} does not match tokenizer")
            values = payload.get("values")
            if not isinstance(values, list) or len(values) != vocab_size:
                raise ValueError(f"token byte table in {token_bytes_json_path!r} must contain {vocab_size} values")
            if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values):
                raise ValueError(f"token byte table in {token_bytes_json_path!r} contains an invalid value")
            special_ids = tokenizer.get_special_token_ids()
            declared_special_ids = payload.get("special_token_ids")
            if (
                not isinstance(declared_special_ids, list)
                or any(
                    not isinstance(token_id, int)
                    or isinstance(token_id, bool)
                    or token_id < 0
                    or token_id >= vocab_size
                    for token_id in declared_special_ids
                )
                or len(declared_special_ids) != len(set(declared_special_ids))
                or set(declared_special_ids) != special_ids
            ):
                raise ValueError(
                    f"token byte table special-token IDs in {token_bytes_json_path!r} do not match tokenizer"
                )
            if any(values[token_id] != 0 for token_id in special_ids):
                raise ValueError(f"token byte table in {token_bytes_json_path!r} assigns bytes to a special token")
            return torch.tensor(values, dtype=torch.int32)
        token_bytes_path = os.path.join(tokenizer._tokenizer_path, "token_bytes.pt")
        if os.path.exists(token_bytes_path):
            return torch.load(token_bytes_path, map_location="cpu", weights_only=True).to(dtype=torch.int32)
        special_ids = tokenizer.get_special_token_ids()
        lengths: list[int] = []
        for new_id in range(vocab_size):
            if new_id in special_ids or new_id == mask_id:
                lengths.append(0)
            else:
                text = tokenizer._tok.decode([new_id], skip_special_tokens=False)
                lengths.append(len(text.encode("utf-8")))
        return torch.tensor(lengths, dtype=torch.int32)
    raise TypeError(f"unknown tokenizer type: {type(tokenizer)}")


def build_frequency_token_classes(
    data_config,
    tokenizer,
    batch_size: int,
    seq_len: int,
    device,
    batches: int,
    num_deciles: int = 7,
) -> tuple[torch.Tensor, dict]:
    """Build fixed output classes: special/EOS plus train-frequency deciles."""
    vocab_size = int(tokenizer.get_vocab_size())
    if num_deciles <= 0:
        raise ValueError("num_deciles must be positive")
    batches = int(batches)
    if batches <= 0:
        raise ValueError("batches must be positive")

    special_ids = {int(i) for i in tokenizer.get_special_token_ids() if 0 <= int(i) < vocab_size}
    token_bytes = build_token_bytes(tokenizer).to(device)
    counts = torch.zeros(vocab_size, dtype=torch.long)
    loader = make_dataloader(data_config, tokenizer, int(batch_size), int(seq_len), "train", device=device)
    observed_tokens = 0
    for _ in range(batches):
        batch = next(loader)
        content = batch.attention_mask & batch.loss_mask & (token_bytes[batch.x_0] > 0)
        for token_id in special_ids:
            content &= batch.x_0 != int(token_id)
        if not content.any():
            continue
        ids = batch.x_0[content].detach().cpu()
        counts += torch.bincount(ids, minlength=vocab_size)
        observed_tokens += int(ids.numel())

    token_class_ids = torch.zeros(vocab_size, dtype=torch.long)
    non_special_ids = [token_id for token_id in range(vocab_size) if token_id not in special_ids]
    non_special_ids.sort(key=lambda token_id: (int(counts[token_id].item()), token_id))
    base_size = len(non_special_ids) // num_deciles
    remainder = len(non_special_ids) % num_deciles
    offset = 0
    class_token_counts = {"0": len(special_ids)}
    for decile in range(1, num_deciles + 1):
        size = base_size + (1 if decile <= remainder else 0)
        ids = non_special_ids[offset : offset + size]
        if ids:
            token_class_ids[torch.tensor(ids, dtype=torch.long)] = decile
        class_token_counts[str(decile)] = len(ids)
        offset += size

    top_count = int(counts.max().item()) if vocab_size else 0
    nonzero = int((counts > 0).sum().item())
    metadata = {
        "output_factorization": "hierarchical_frequency_decile",
        "output_class_count": int(num_deciles + 1),
        "output_class_0_name": "eos_special",
        "frequency_decile_count": int(num_deciles),
        "frequency_batches": batches,
        "frequency_observed_tokens": int(observed_tokens),
        "frequency_nonzero_token_types": nonzero,
        "frequency_top_count": top_count,
        "special_token_count": len(special_ids),
        "class_token_counts": class_token_counts,
    }
    return token_class_ids, metadata


def encode_document(tokenizer, text: str, data_config) -> list[int]:
    encoded = encode_document_batch(tokenizer, [text], data_config)
    return encoded[0] if encoded else []


def encode_document_batch(tokenizer, texts: list[str], data_config) -> list[list[int]]:
    cleaned = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
    if not cleaned:
        return []
    encoded = tokenizer.encode(cleaned)
    out: list[list[int]] = []
    for ids in encoded:
        if not ids:
            continue
        doc_ids: list[int] = []
        if data_config.prepend_bos:
            bos_id = tokenizer.get_bos_token_id()
            if bos_id is None:
                raise ValueError("data.prepend_bos=true but tokenizer has no bos_token_id")
            doc_ids.append(int(bos_id))
        doc_ids.extend(int(i) for i in ids)
        if data_config.append_eos:
            eos_id = tokenizer.get_eos_token_id()
            if eos_id is None:
                raise ValueError("data.append_eos=true but tokenizer has no eos_token_id")
            if not doc_ids or doc_ids[-1] != eos_id:
                doc_ids.append(int(eos_id))
        out.append(doc_ids)
    return out


def _apply_train_mask_augmentation(
    data_config,
    loss_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    rng: random.Random,
) -> None:
    batch_size, seq_len = loss_mask.shape
    for row_idx in range(batch_size):
        if rng.random() < float(data_config.random_length_ratio):
            min_len = min(max(1, int(data_config.random_length_min)), seq_len)
            length = rng.randint(min_len, seq_len)
            attention_mask[row_idx, length:] = False
            loss_mask[row_idx, length:] = False
            continue
        if rng.random() < float(data_config.continuation_ratio):
            min_prefix = round(float(data_config.continuation_min_prefix_frac) * seq_len)
            max_prefix = round(float(data_config.continuation_max_prefix_frac) * seq_len)
            lo = min(seq_len, max(1, min_prefix))
            hi = min(seq_len, max(lo, max_prefix))
            prefix_len = rng.randint(lo, hi)
            loss_mask[row_idx, :prefix_len] = False


def _cap_document_prefix_with_eos(doc: list[int], max_tokens: int, eos_id: int | None) -> list[int]:
    max_tokens = int(max_tokens)
    if max_tokens <= 0:
        return []
    row = list(doc)
    if eos_id is not None:
        eos_id = int(eos_id)
        try:
            eos_pos = row.index(eos_id)
        except ValueError:
            eos_pos = -1
        if eos_pos >= 0 and eos_pos < max_tokens:
            return row[: eos_pos + 1]
    if len(row) <= max_tokens:
        return row
    row = row[:max_tokens]
    if eos_id is not None and row:
        row = row[: max_tokens - 1] + [int(eos_id)]
    return row[:max_tokens]


def _short_document_dataloader(
    data_config,
    tokenizer,
    batch_size: int,
    seq_len: int,
    split: str,
    device,
    augmentation_rng: random.Random,
    rank: int = 0,
    world_size: int = 1,
):
    """Yield one capped document or document prefix per padded row."""
    assert split in ("train", "val")
    source = ParquetTextSource(data_config)
    batches = source.document_batches(split, rank=rank, world_size=world_size)
    pin_memory = torch.device(device).type == "cuda"
    pad_id = tokenizer.get_pad_token_id()
    if pad_id is None:
        pad_id = 0
    eos_id = tokenizer.get_eos_token_id()
    max_tokens = min(int(data_config.short_document_max_tokens), int(seq_len))

    cpu_x_0 = torch.empty(batch_size, seq_len, dtype=torch.long)
    cpu_attention_mask = torch.empty(batch_size, seq_len, dtype=torch.bool)
    cpu_loss_mask = torch.empty(batch_size, seq_len, dtype=torch.bool)
    device_x_0 = torch.empty(batch_size, seq_len, dtype=torch.long, device=device)
    device_attention_mask = torch.empty(batch_size, seq_len, dtype=torch.bool, device=device)
    device_loss_mask = torch.empty(batch_size, seq_len, dtype=torch.bool, device=device)

    document_buffer: list[list[int]] = []
    epoch = 1

    def refill() -> None:
        nonlocal document_buffer, epoch
        while not document_buffer:
            doc_batch, epoch = next(batches)
            document_buffer = [
                _cap_document_prefix_with_eos(doc, max_tokens, eos_id)
                for doc in encode_document_batch(tokenizer, doc_batch, data_config)
            ]
            document_buffer = [doc for doc in document_buffer if doc]

    while True:
        rows: list[list[int]] = []
        while len(rows) < batch_size:
            refill()
            doc = document_buffer.pop(0)
            if doc:
                rows.append(doc)

        cpu_x_0.fill_(pad_id)
        cpu_attention_mask.fill_(False)
        cpu_loss_mask.fill_(False)

        for row_idx, row in enumerate(rows):
            row_len = min(len(row), seq_len)
            cpu_x_0[row_idx, :row_len] = torch.tensor(row[:row_len], dtype=torch.long)
            cpu_attention_mask[row_idx, :row_len] = True
            cpu_loss_mask[row_idx, :row_len] = True

        if not data_config.loss_on_bos:
            bos_id = tokenizer.get_bos_token_id()
            if bos_id is not None:
                cpu_loss_mask.logical_and_(cpu_x_0 != int(bos_id))
        if not data_config.loss_on_eos and eos_id is not None:
            cpu_loss_mask.logical_and_(cpu_x_0 != int(eos_id))

        if split == "train":
            _apply_train_mask_augmentation(data_config, cpu_loss_mask, cpu_attention_mask, augmentation_rng)

        device_x_0.copy_(cpu_x_0, non_blocking=pin_memory)
        device_attention_mask.copy_(cpu_attention_mask, non_blocking=pin_memory)
        device_loss_mask.copy_(cpu_loss_mask, non_blocking=pin_memory)
        yield DLMDataBatch(
            x_0=device_x_0,
            loss_mask=device_loss_mask,
            attention_mask=device_attention_mask,
            epoch=epoch,
        )


def _boundary_aware_dataloader(
    data_config,
    tokenizer,
    batch_size: int,
    seq_len: int,
    split: str,
    device,
    augmentation_rng: random.Random,
    rank: int = 0,
    world_size: int = 1,
):
    """Pack complete documents into rows without crossing document boundaries.

    Each row may contain one or more documents. Position IDs reset at the start
    of each document. The returned ``document_boundary_mask`` is a 3D causal
    attention mask that blocks future, cross-document, and PAD keys.
    """
    assert split in ("train", "val")
    source = ParquetTextSource(data_config)
    batches = source.document_batches(split, rank=rank, world_size=world_size)
    pin_memory = torch.device(device).type == "cuda"
    pad_id = tokenizer.get_pad_token_id()
    if pad_id is None:
        pad_id = 0

    cpu_x_0 = torch.empty(batch_size, seq_len, dtype=torch.long)
    cpu_attention_mask = torch.empty(batch_size, seq_len, dtype=torch.bool)
    cpu_loss_mask = torch.empty(batch_size, seq_len, dtype=torch.bool)
    cpu_position_ids = torch.empty(batch_size, seq_len, dtype=torch.long)
    cpu_boundary_mask = torch.empty(batch_size, seq_len, seq_len, dtype=torch.bool)
    cpu_causal_template = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

    device_x_0 = torch.empty(batch_size, seq_len, dtype=torch.long, device=device)
    device_attention_mask = torch.empty(batch_size, seq_len, dtype=torch.bool, device=device)
    device_loss_mask = torch.empty(batch_size, seq_len, dtype=torch.bool, device=device)
    device_position_ids = torch.empty(batch_size, seq_len, dtype=torch.long, device=device)
    device_boundary_mask = torch.empty(batch_size, seq_len, seq_len, dtype=torch.bool, device=device)

    document_buffer: list[list[int]] = []
    rows: list[list[int]] = []
    row_doc_lengths: list[list[int]] = []
    epoch = 1

    def finalize_batch():
        actual_batch_size = len(rows)
        if actual_batch_size == 0:
            return None

        cpu_x_0.fill_(pad_id)
        cpu_attention_mask.fill_(False)
        cpu_loss_mask.fill_(False)
        cpu_position_ids.fill_(0)
        cpu_boundary_mask.fill_(False)

        for row_idx, (row, doc_lengths) in enumerate(zip(rows, row_doc_lengths, strict=False)):
            row_len = len(row)
            cpu_x_0[row_idx, :row_len] = torch.tensor(row, dtype=torch.long)
            cpu_attention_mask[row_idx, :row_len] = True
            cpu_loss_mask[row_idx, :row_len] = True

            doc_starts = [0]
            for length in doc_lengths[:-1]:
                doc_starts.append(doc_starts[-1] + length)

            for start, length in zip(doc_starts, doc_lengths, strict=False):
                end = min(start + length, seq_len)
                cpu_position_ids[row_idx, start:end] = torch.arange(end - start, dtype=torch.long)
                doc_len = end - start
                cpu_boundary_mask[row_idx, start:end, start:end] = cpu_causal_template[:doc_len, :doc_len]

        if not data_config.loss_on_bos:
            bos_id = tokenizer.get_bos_token_id()
            if bos_id is not None:
                cpu_loss_mask.logical_and_(cpu_x_0 != int(bos_id))
        if not data_config.loss_on_eos:
            eos_id = tokenizer.get_eos_token_id()
            if eos_id is not None:
                cpu_loss_mask.logical_and_(cpu_x_0 != int(eos_id))

        if split == "train":
            _apply_train_mask_augmentation(data_config, cpu_loss_mask, cpu_attention_mask, augmentation_rng)

        device_x_0[:actual_batch_size].copy_(cpu_x_0[:actual_batch_size], non_blocking=pin_memory)
        device_attention_mask[:actual_batch_size].copy_(cpu_attention_mask[:actual_batch_size], non_blocking=pin_memory)
        device_loss_mask[:actual_batch_size].copy_(cpu_loss_mask[:actual_batch_size], non_blocking=pin_memory)
        device_position_ids[:actual_batch_size].copy_(cpu_position_ids[:actual_batch_size], non_blocking=pin_memory)
        device_boundary_mask[:actual_batch_size].copy_(cpu_boundary_mask[:actual_batch_size], non_blocking=pin_memory)

        return DLMDataBatch(
            x_0=device_x_0[:actual_batch_size],
            loss_mask=device_loss_mask[:actual_batch_size],
            attention_mask=device_attention_mask[:actual_batch_size],
            epoch=epoch,
            position_ids=device_position_ids[:actual_batch_size],
            document_boundary_mask=device_boundary_mask[:actual_batch_size],
        )

    while True:
        while len(rows) < batch_size:
            if not document_buffer:
                doc_batch, epoch = next(batches)
                document_buffer = encode_document_batch(tokenizer, doc_batch, data_config)

            while document_buffer and len(rows) < batch_size:
                doc = document_buffer.pop(0)
                if not doc:
                    continue

                if rows and len(rows[-1]) + len(doc) <= seq_len:
                    rows[-1].extend(doc)
                    row_doc_lengths[-1].append(len(doc))
                else:
                    if len(rows) == batch_size:
                        document_buffer.insert(0, doc)
                        break
                    if len(doc) > seq_len:
                        doc = doc[:seq_len]
                    rows.append(list(doc))
                    row_doc_lengths.append([len(doc)])

        if len(rows) == batch_size:
            batch = finalize_batch()
            if batch is not None:
                yield batch
            rows = []
            row_doc_lengths = []


def _content_budget_segmented_dataloader(
    data_config,
    tokenizer,
    batch_size: int,
    seq_len: int,
    split: str,
    device,
    augmentation_rng: random.Random,
    rank: int = 0,
    world_size: int = 1,
):
    # Preserve the historical entry point and data.py's tokenizer/source seams.
    from .data_segmented import _content_budget_segmented_dataloader as loader

    yield from loader(data_config, tokenizer, batch_size, seq_len, split, device, augmentation_rng, rank, world_size)


def build_synthetic_eos_overfit_canary_batch(
    data_config,
    tokenizer,
    batch_size: int,
    seq_len: int,
    device,
) -> DLMDataBatch:
    unique_non_eos_token_count = int(getattr(data_config, "synthetic_eos_overfit_unique_non_eos_token_count", 0) or 0)
    if unique_non_eos_token_count and getattr(data_config, "synthetic_eos_overfit_unique_non_eos_tokens", False):
        raise ValueError(
            "both synthetic_eos_overfit_unique_non_eos_token_count and "
            "synthetic_eos_overfit_unique_non_eos_tokens cannot be set simultaneously"
        )
    fixed_row_text = getattr(data_config, "synthetic_eos_overfit_fixed_row_text", "")
    docs_source = EOS_OVERFIT_CANARY_DOCUMENTS
    if fixed_row_text:
        docs_source = (fixed_row_text,)
    elif unique_non_eos_token_count:
        document_lengths = getattr(
            data_config,
            "synthetic_eos_overfit_document_lengths",
            _EOS_OVERFIT_CANARY_DOCUMENT_LENGTHS,
        )
        docs_source = _build_eos_overfit_canary_documents_for_non_eos_token_count(
            unique_non_eos_token_count, document_lengths=document_lengths
        )
    elif getattr(data_config, "synthetic_eos_overfit_unique_non_eos_tokens", False):
        docs_source = EOS_OVERFIT_CANARY_DOCUMENTS_UNIQUE_NON_EOS

    docs = encode_document_batch(
        tokenizer,
        list(docs_source),
        data_config,
    )
    if not docs:
        raise ValueError("EOS overfit canary documents encoded to an empty batch")
    if any(len(doc) > int(seq_len) for doc in docs):
        raise ValueError("EOS overfit canary max_seq_len is too small for the synthetic documents")

    pad_id = tokenizer.get_pad_token_id()
    if pad_id is None:
        pad_id = 0
    eos_id = tokenizer.get_eos_token_id()
    if eos_id is None:
        raise ValueError("EOS overfit canary requires a tokenizer with eos_token_id")
    batch_size = int(batch_size)
    seq_len = int(seq_len)
    x_0 = torch.full((batch_size, seq_len), int(pad_id), dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
    loss_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)

    for row_idx in range(batch_size):
        doc = docs[row_idx % len(docs)]
        row_len = len(doc)
        x_0[row_idx, :row_len] = torch.tensor(doc, dtype=torch.long, device=device)
        attention_mask[row_idx, :row_len] = True
        policy = getattr(data_config, "synthetic_eos_overfit_loss_mask_policy", "eos_only")
        if policy == "eos_only":
            loss_mask[row_idx, :row_len] = x_0[row_idx, :row_len] == int(eos_id)
        elif policy == "all_lossable_tokens":
            loss_mask[row_idx, :row_len] = True
        else:
            raise ValueError("data.synthetic_eos_overfit_loss_mask_policy must be 'eos_only' or 'all_lossable_tokens'")

    if not data_config.loss_on_bos:
        bos_id = tokenizer.get_bos_token_id()
        if bos_id is not None:
            loss_mask &= x_0 != int(bos_id)
    if not data_config.loss_on_eos and eos_id is not None:
        loss_mask &= x_0 != int(eos_id)

    return DLMDataBatch(x_0=x_0, loss_mask=loss_mask, attention_mask=attention_mask, epoch=1)


def _synthetic_eos_overfit_canary_dataloader(
    data_config,
    tokenizer,
    batch_size: int,
    seq_len: int,
    split: str,
    device,
    rank: int = 0,
    world_size: int = 1,
):
    del split, rank, world_size
    while True:
        yield build_synthetic_eos_overfit_canary_batch(data_config, tokenizer, batch_size, seq_len, device)


def build_mod97_task_batch(
    data_config,
    tokenizer,
    batch_size: int,
    seq_len: int,
    split: str,
    device,
    *,
    batch_index: int = 0,
    rank: int = 0,
    world_size: int = 1,
) -> DLMDataBatch:
    """Build one deterministic, prompt-conditioned Task-A batch."""
    from .mod97_task import EVAL_DEPTHS, SUPERVISION_BY_DATASET, evaluation_problem, train_problem, training_parts

    dataset = str(data_config.synthetic_dataset)
    if dataset not in SUPERVISION_BY_DATASET:
        raise ValueError("mod97 task batch requires an answer or trace dataset")
    if split not in {"train", "val"}:
        raise ValueError("mod97 task split must be train or val")
    if not data_config.prepend_bos or not data_config.append_eos or not data_config.loss_on_eos:
        raise ValueError("mod97 task rows require BOS context and supervised EOS")
    if batch_size <= 0 or seq_len <= 0 or batch_index < 0 or rank < 0 or world_size <= rank:
        raise ValueError("invalid mod97 batch coordinates")
    bos_id = tokenizer.get_bos_token_id()
    eos_id = tokenizer.get_eos_token_id()
    if bos_id is None or eos_id is None:
        raise ValueError("mod97 task rows require BOS and EOS token IDs")
    pad_id = tokenizer.get_pad_token_id()
    if pad_id is None:
        pad_id = 0
    x_0 = torch.full((batch_size, seq_len), int(pad_id), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
    loss_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
    data_seed = int(data_config.shuffle_seed)
    global_batch = batch_index * world_size + rank
    for row in range(batch_size):
        problem_index = global_batch * batch_size + row
        if split == "train":
            problem = train_problem(problem_index, data_seed)
        else:
            depth = EVAL_DEPTHS[problem_index % len(EVAL_DEPTHS)]
            stratum_index = problem_index // len(EVAL_DEPTHS)
            problem = evaluation_problem(depth, stratum_index, data_seed)
        prompt, completion = training_parts(problem, SUPERVISION_BY_DATASET[dataset])
        prompt_ids = [int(token) for token in tokenizer.encode(prompt)]
        completion_ids = [int(token) for token in tokenizer.encode(completion)]
        tokens = [int(bos_id), *prompt_ids, *completion_ids, int(eos_id)]
        if len(tokens) > seq_len:
            raise ValueError(
                f"mod97 task row exceeds the fixed {seq_len}-position capacity: "
                f"problem={problem.problem_id} tokens={len(tokens)}"
            )
        row_len = len(tokens)
        loss_start = 1 + len(prompt_ids)
        x_0[row, :row_len] = torch.tensor(tokens, dtype=torch.long)
        attention_mask[row, :row_len] = True
        loss_mask[row, loss_start:row_len] = True
        if not data_config.loss_on_bos:
            loss_mask[row] &= x_0[row] != int(bos_id)
    target_device = torch.device(device)
    return DLMDataBatch(
        x_0=x_0.to(target_device),
        loss_mask=loss_mask.to(target_device),
        attention_mask=attention_mask.to(target_device),
        epoch=1 + global_batch,
    )


def _synthetic_mod97_task_dataloader(
    data_config,
    tokenizer,
    batch_size: int,
    seq_len: int,
    split: str,
    device,
    rank: int = 0,
    world_size: int = 1,
):
    batch_index = 0
    while True:
        yield build_mod97_task_batch(
            data_config,
            tokenizer,
            batch_size,
            seq_len,
            split,
            device,
            batch_index=batch_index,
            rank=rank,
            world_size=world_size,
        )
        batch_index += 1


def make_dataloader(
    data_config,
    tokenizer,
    batch_size: int,
    seq_len: int,
    split: str,
    device,
    rank: int = 0,
    world_size: int = 1,
    augmentation_seed: int | None = None,
):
    if augmentation_seed is None:
        augmentation_seed = int(data_config.shuffle_seed) + int(rank)
    augmentation_rng = random.Random(int(augmentation_seed))

    from .mod97_task import is_mod97_dataset

    if is_mod97_dataset(str(getattr(data_config, "synthetic_dataset", ""))):
        yield from _synthetic_mod97_task_dataloader(
            data_config, tokenizer, batch_size, seq_len, split, device, rank=rank, world_size=world_size
        )
        return

    if getattr(data_config, "synthetic_dataset", "") == SYNTHETIC_DATASET_EOS_OVERFIT_CANARY:
        yield from _synthetic_eos_overfit_canary_dataloader(
            data_config, tokenizer, batch_size, seq_len, split, device, rank=rank, world_size=world_size
        )
        return

    if getattr(data_config, "short_document_curriculum", False):
        yield from _short_document_dataloader(
            data_config,
            tokenizer,
            batch_size,
            seq_len,
            split,
            device,
            augmentation_rng,
            rank=rank,
            world_size=world_size,
        )
        return

    if getattr(data_config, "boundary_aware_packing", False):
        yield from _boundary_aware_dataloader(
            data_config,
            tokenizer,
            batch_size,
            seq_len,
            split,
            device,
            augmentation_rng,
            rank=rank,
            world_size=world_size,
        )
        return

    if getattr(data_config, "content_budget_segmented_packing", False):
        if split == "train" and getattr(data_config, "bos_packing_mode", "none") != "none":
            from .data_bos_packing import bos_packing_dataloader

            yield from bos_packing_dataloader(
                data_config, tokenizer, batch_size, seq_len, device, augmentation_rng, rank, world_size
            )
            return
        if split == "train" and getattr(data_config, "eos_mask_tail_mode", "none") != "none":
            from .data_mask_tail import mask_tail_dataloader

            yield from mask_tail_dataloader(
                data_config,
                tokenizer,
                batch_size,
                seq_len,
                device,
                augmentation_rng,
                augmentation_seed,
                rank,
                world_size,
            )
            return
        yield from _content_budget_segmented_dataloader(
            data_config,
            tokenizer,
            batch_size,
            seq_len,
            split,
            device,
            augmentation_rng,
            rank=rank,
            world_size=world_size,
        )
        return

    assert split in ("train", "val")
    source = ParquetTextSource(data_config)
    batches = source.document_batches(split, rank=rank, world_size=world_size)
    token_buffer: list[int] = []
    epoch = 1
    tokens_per_batch = batch_size * seq_len

    def refill():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        for doc_ids in encode_document_batch(tokenizer, doc_batch, data_config):
            token_buffer.extend(doc_ids)

    row_buffer = torch.empty((batch_size, seq_len), dtype=torch.long)
    pin_memory = torch.device(device).type == "cuda"
    cpu_buffer = torch.empty(batch_size * seq_len, dtype=torch.long, pin_memory=pin_memory)
    device_buffer = torch.empty(batch_size * seq_len, dtype=torch.long, device=device)
    x_0 = device_buffer.view(batch_size, seq_len)
    cpu_loss_mask = torch.empty((batch_size, seq_len), dtype=torch.bool, pin_memory=pin_memory)
    cpu_attention_mask = torch.empty((batch_size, seq_len), dtype=torch.bool, pin_memory=pin_memory)
    device_loss_mask = torch.empty((batch_size, seq_len), dtype=torch.bool, device=device)
    device_attention_mask = torch.empty((batch_size, seq_len), dtype=torch.bool, device=device)

    while True:
        while len(token_buffer) < tokens_per_batch:
            refill()
        flat = token_buffer[:tokens_per_batch]
        del token_buffer[:tokens_per_batch]
        row_buffer.copy_(torch.tensor(flat, dtype=torch.long).view(batch_size, seq_len))

        cpu_attention_mask.fill_(True)
        cpu_loss_mask.fill_(True)
        if not data_config.loss_on_bos:
            bos_id = tokenizer.get_bos_token_id()
            if bos_id is not None:
                cpu_loss_mask &= row_buffer != int(bos_id)
        if not data_config.loss_on_eos:
            eos_id = tokenizer.get_eos_token_id()
            if eos_id is not None:
                cpu_loss_mask &= row_buffer != int(eos_id)
        if split == "train":
            _apply_train_mask_augmentation(data_config, cpu_loss_mask, cpu_attention_mask, augmentation_rng)

        cpu_buffer.copy_(row_buffer.view(-1))
        device_buffer.copy_(cpu_buffer, non_blocking=pin_memory)
        device_loss_mask.copy_(cpu_loss_mask, non_blocking=pin_memory)
        device_attention_mask.copy_(cpu_attention_mask, non_blocking=pin_memory)
        yield DLMDataBatch(
            x_0=x_0,
            loss_mask=device_loss_mask,
            attention_mask=device_attention_mask,
            epoch=epoch,
        )


def build_train_loader(config, tokenizer, rank: int, world_size: int, device):
    training_seed = int(config.runtime.training_seed if config.runtime.training_seed >= 0 else config.runtime.seed)
    return make_dataloader(
        config.data,
        tokenizer,
        int(config.train.device_batch_size),
        config.model.max_seq_len,
        "train",
        device=device,
        rank=rank,
        world_size=world_size,
        augmentation_seed=training_seed + int(rank),
    )


def build_val_loader(config, tokenizer, device):
    return make_dataloader(
        config.data,
        tokenizer,
        int(config.train.device_batch_size),
        config.model.max_seq_len,
        "val",
        device=device,
    )
