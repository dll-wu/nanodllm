"""Streaming document-isolated M2T rows with visible, non-target MASK tails.

The tail is chosen once per natural document, independently of remaining row
capacity. Original tokens are never dropped or relabeled. Boundary fragments
retain at least the last two original positions together with their entire tail.
"""

import random
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TailSegment:
    tokens: tuple[int, ...]
    tail_length: int
    document_start: bool


def tail_length(length: int, mode: str, size: int, rng: random.Random) -> int:
    if mode == "random":
        return rng.randint(0, size)
    if mode == "aligned":
        return (-length) % size
    raise ValueError("unknown EOS MASK tail mode")


def pack_tail_rows(documents, *, seq_len: int, mode: str, size: int, rng: random.Random):
    """Yield (segments, epoch); unused row slots remain hidden, not resampled."""
    if mode not in {"random", "aligned"} or not 0 < size <= seq_len - 2:
        raise ValueError("invalid EOS MASK tail packing dimensions")
    if mode == "aligned" and seq_len % size:
        raise ValueError("aligned row length must be a multiple of tail size")
    pending, offset, tail, epoch = [], 0, 0, 0
    while True:
        remaining, segments = seq_len, []
        while remaining:
            if not pending:
                document, epoch = next(documents)
                if not document:
                    continue
                pending, offset = list(document), 0
                tail = tail_length(len(pending), mode, size, rng)
            available = len(pending)
            if available + tail <= remaining:
                segments.append(TailSegment(tuple(pending), tail, offset == 0))
                remaining -= available + tail
                pending = []
            else:
                # Do not strand EOS (or its tail) alone across a row boundary.
                take = min(remaining, max(0, available - 2))
                if mode == "aligned":
                    take = take // size * size
                if not take:
                    break
                segments.append(TailSegment(tuple(pending[:take]), 0, offset == 0))
                pending = pending[take:]
                offset += take
                remaining -= take
        if not segments:
            raise ValueError("EOS MASK tail packer made no progress")
        yield segments, epoch


def mask_tail_dataloader(
    data_config, tokenizer, batch_size, seq_len, device, augmentation_rng, tail_seed, rank, world_size
):
    # data.py dispatches here only after its common types and source are loaded.
    from .data import DLMDataBatch, ParquetTextSource, _apply_train_mask_augmentation, encode_document_batch

    source = ParquetTextSource(data_config)
    mask_id, pad_id, eos_id = (
        tokenizer.get_mask_token_id(),
        tokenizer.get_pad_token_id(),
        tokenizer.get_eos_token_id(),
    )
    if None in (mask_id, pad_id, eos_id) or len({mask_id, pad_id, eos_id}) != 3:
        raise ValueError("EOS MASK tails require distinct MASK/PAD/EOS tokens")

    def documents():
        for texts, epoch in source.document_batches("train", rank=rank, world_size=world_size):
            for tokens in encode_document_batch(tokenizer, texts, data_config):
                if not tokens or tokens[-1] != eos_id or mask_id in tokens or pad_id in tokens:
                    raise ValueError("EOS-tail source must contain a natural EOS and no raw MASK/PAD tokens")
                yield tokens, epoch

    rows = pack_tail_rows(
        documents(),
        seq_len=seq_len,
        mode=data_config.eos_mask_tail_mode,
        size=data_config.eos_mask_tail_size,
        rng=random.Random(f"eos-mask-tail-v1:{tail_seed}"),
    )
    pin = torch.device(device).type == "cuda"
    shapes = {
        "x_0": ((batch_size, seq_len), torch.long),
        "loss_mask": ((batch_size, seq_len), torch.bool),
        "attention_mask": ((batch_size, seq_len), torch.bool),
        "position_ids": ((batch_size, seq_len), torch.long),
        "document_start_mask": ((batch_size, seq_len), torch.bool),
        "document_boundary_mask": ((batch_size, seq_len, seq_len), torch.bool),
    }
    cpu = {name: torch.empty(shape, dtype=dtype, pin_memory=pin) for name, (shape, dtype) in shapes.items()}
    destination = {name: torch.empty(shape, dtype=dtype, device=device) for name, (shape, dtype) in shapes.items()}
    bos_id = tokenizer.get_bos_token_id()
    while True:
        for tensor in cpu.values():
            tensor.zero_()
        cpu["x_0"].fill_(pad_id)
        for row in range(batch_size):
            segments, epoch = next(rows)
            start = 0
            for segment in segments:
                original_end = start + len(segment.tokens)
                end = original_end + segment.tail_length
                cpu["x_0"][row, start:original_end] = torch.tensor(segment.tokens)
                cpu["x_0"][row, original_end:end] = mask_id
                cpu["attention_mask"][row, start:end] = True
                cpu["loss_mask"][row, start:original_end] = True
                cpu["position_ids"][row, start:end] = torch.arange(end - start)
                cpu["document_boundary_mask"][row, start:end, start:end] = True
                cpu["document_start_mask"][row, start] = segment.document_start
                start = end
        if bos_id is not None:
            cpu["loss_mask"] &= cpu["x_0"] != bos_id
        # Preserve the baseline augmentation RNG and policy. Cropping may hide
        # an already allocated tail; it never turns tail tokens into targets.
        _apply_train_mask_augmentation(data_config, cpu["loss_mask"], cpu["attention_mask"], augmentation_rng)
        for name in cpu:
            destination[name].copy_(cpu[name], non_blocking=pin)
        yield DLMDataBatch(**destination, epoch=epoch)
