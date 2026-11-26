"""BOS anchors and best-fit cropping, independent of training objectives.

The original segmented loader remains the disabled path. New modes retain the
same source/tokenizer and natural EOS policy; they are not a native-recipe port.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Segment:
    tokens: tuple[int, ...]
    document_start: bool
    added_bos: bool = False


@dataclass(frozen=True)
class PackedRow:
    segments: tuple[Segment, ...]
    epoch: int
    cropped: tuple[int, ...] = ()
    pending_tokens: int = 0


def pack_bos_rows(documents, *, seq_len, mode, bos_id, buffer_size=1000, refill_size=128):
    """Yield full rows; only best-fit discards tails, never inventing an EOS.

    Continuation BOS is a context anchor, not a new true document boundary.
    Source order is deterministic; best-fit necessarily changes emission order.
    """
    if mode not in {"continuation_bos", "bestfit_segmented", "bestfit_cross_document"}:
        raise ValueError("unknown BOS packing mode")
    if seq_len < 2 or buffer_size < 1 or refill_size < 1:
        raise ValueError("invalid BOS packing dimensions")

    def next_doc():
        tokens, epoch = next(documents)
        if len(tokens) < 2 or tokens[0] != bos_id:
            raise ValueError("BOS packing expects nonempty BOS-prefixed documents")
        return tuple(tokens), epoch

    pending, offset, epoch, buffer = (), 0, 0, []
    while True:
        remaining, segments, cropped = seq_len, [], []
        while remaining:
            if mode == "continuation_bos":
                if not pending:
                    pending, epoch = next_doc()
                    offset = 0
                anchor = offset > 0
                take = min(remaining - int(anchor), len(pending))
                if take <= 0:
                    raise ValueError("continuation BOS failed to make progress")
                tokens = ((bos_id,) if anchor else ()) + pending[:take]
                segments.append(Segment(tokens, offset == 0, anchor))
                offset += take
                pending = pending[take:]
            else:
                # Batch refill matches the upstream best-fit selection algorithm.
                while len(buffer) < buffer_size:
                    for _ in range(refill_size):
                        doc, epoch = next_doc()
                        buffer.append(doc)
                best_idx = -1
                best_len = 0
                for index, doc in enumerate(buffer):
                    if best_len < len(doc) <= remaining:
                        best_idx, best_len = index, len(doc)
                if best_idx == -1:
                    best_idx = min(range(len(buffer)), key=lambda index: len(buffer[index]))
                doc = buffer.pop(best_idx)
                tokens = doc[:remaining]
                cropped.extend(doc[remaining:])
                segments.append(Segment(tokens, True))
            remaining -= len(tokens)
        pending_count = len(pending) if mode == "continuation_bos" else sum(map(len, buffer))
        yield PackedRow(tuple(segments), epoch, tuple(cropped), pending_count)


def bos_packing_dataloader(
    data_config, tokenizer, batch_size, seq_len, device, augmentation_rng, rank, world_size, *, observer=None
):
    # Keep data discovery and tokenization owned by data.py, including test seams.
    from .data import DLMDataBatch, ParquetTextSource, _apply_train_mask_augmentation, encode_document_batch

    source = ParquetTextSource(data_config)
    bos_id = tokenizer.get_bos_token_id()
    if bos_id is None:
        raise ValueError("BOS packing requires a BOS token")

    def documents():
        for texts, epoch in source.document_batches("train", rank=rank, world_size=world_size):
            for tokens in encode_document_batch(tokenizer, texts, data_config):
                yield tokens, epoch

    mode = data_config.bos_packing_mode
    rows = pack_bos_rows(
        documents(),
        seq_len=seq_len,
        mode=mode,
        bos_id=bos_id,
        buffer_size=data_config.bos_packing_buffer_size,
        refill_size=data_config.tokenizer_batch_size,
    )
    cross_document = mode == "bestfit_cross_document"
    pin = torch.device(device).type == "cuda"
    shapes = {
        "x_0": ((batch_size, seq_len), torch.long),
        "loss_mask": ((batch_size, seq_len), torch.bool),
        "attention_mask": ((batch_size, seq_len), torch.bool),
        "position_ids": ((batch_size, seq_len), torch.long),
        "document_start_mask": ((batch_size, seq_len), torch.bool),
        "document_boundary_mask": ((batch_size, seq_len, seq_len), torch.bool),
    }
    # Match the legacy segmented loader's host-buffer lifetime. Pinned reusable
    # buffers would require an explicit transfer-completion event before refill.
    cpu = {name: torch.empty(shape, dtype=dtype) for name, (shape, dtype) in shapes.items()}
    destination = {name: torch.empty(shape, dtype=dtype, device=device) for name, (shape, dtype) in shapes.items()}
    for_row = torch.arange(seq_len)
    while True:
        cpu["attention_mask"].fill_(True)
        cpu["loss_mask"].fill_(True)
        cpu["document_start_mask"].zero_()
        cpu["document_boundary_mask"].fill_(cross_document)
        for row in range(batch_size):
            packed = next(rows)
            if observer is not None:
                observer(packed)
            start = 0
            for segment in packed.segments:
                end = start + len(segment.tokens)
                cpu["x_0"][row, start:end] = torch.tensor(segment.tokens)
                cpu["position_ids"][row, start:end] = for_row[: end - start]
                cpu["document_start_mask"][row, start] = segment.document_start
                if not cross_document:
                    cpu["document_boundary_mask"][row, start:end, start:end] = True
                start = end
            if cross_document:
                cpu["position_ids"][row] = for_row
        cpu["loss_mask"] &= cpu["x_0"] != bos_id
        _apply_train_mask_augmentation(data_config, cpu["loss_mask"], cpu["attention_mask"], augmentation_rng)
        for name in cpu:
            destination[name].copy_(cpu[name], non_blocking=pin)
        yield DLMDataBatch(**destination, epoch=packed.epoch)
