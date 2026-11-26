"""Historical streaming segmentation with unchanged row, mask and RNG semantics."""

import random

import torch


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
    """Yield fully-filled rows with same-segment bidirectional attention masks.

    Documents keep their natural EOS token from ``encode_document_batch``. Long
    documents may continue in later row segments; no EOS is inserted at a split.
    """
    from .data import DLMDataBatch, ParquetTextSource, _apply_train_mask_augmentation, encode_document_batch

    assert split in ("train", "val")
    source = ParquetTextSource(data_config)
    batches = source.document_batches(split, rank=rank, world_size=world_size)
    pin_memory = torch.device(device).type == "cuda"

    cpu_x_0 = torch.empty(batch_size, seq_len, dtype=torch.long)
    cpu_attention_mask = torch.empty(batch_size, seq_len, dtype=torch.bool)
    cpu_loss_mask = torch.empty(batch_size, seq_len, dtype=torch.bool)
    cpu_position_ids = torch.empty(batch_size, seq_len, dtype=torch.long)
    cpu_boundary_mask = torch.empty(batch_size, seq_len, seq_len, dtype=torch.bool)
    cpu_document_start_mask = torch.empty(batch_size, seq_len, dtype=torch.bool)

    device_x_0 = torch.empty(batch_size, seq_len, dtype=torch.long, device=device)
    device_attention_mask = torch.empty(batch_size, seq_len, dtype=torch.bool, device=device)
    device_loss_mask = torch.empty(batch_size, seq_len, dtype=torch.bool, device=device)
    device_position_ids = torch.empty(batch_size, seq_len, dtype=torch.long, device=device)
    device_boundary_mask = torch.empty(batch_size, seq_len, seq_len, dtype=torch.bool, device=device)
    device_document_start_mask = torch.empty(batch_size, seq_len, dtype=torch.bool, device=device)

    document_buffer: list[list[int]] = []
    current_doc: list[int] = []
    current_doc_offset = 0
    epoch = 1

    def next_document() -> list[int]:
        nonlocal document_buffer, epoch
        while not document_buffer:
            doc_batch, epoch = next(batches)
            document_buffer = encode_document_batch(tokenizer, doc_batch, data_config)
            document_buffer = [doc for doc in document_buffer if doc]
        return document_buffer.pop(0)

    while True:
        cpu_attention_mask.fill_(True)
        cpu_loss_mask.fill_(True)
        cpu_position_ids.fill_(0)
        cpu_boundary_mask.fill_(False)
        cpu_document_start_mask.fill_(False)

        for row_idx in range(batch_size):
            row_tokens: list[int] = []
            row_segments: list[tuple[int, int, bool]] = []
            while len(row_tokens) < seq_len:
                if not current_doc:
                    current_doc = next_document()
                    current_doc_offset = 0
                remaining = seq_len - len(row_tokens)
                take = min(remaining, len(current_doc))
                start = len(row_tokens)
                end = start + take
                is_document_start = current_doc_offset == 0
                row_tokens.extend(current_doc[:take])
                row_segments.append((start, end, is_document_start))
                del current_doc[:take]
                current_doc_offset += take
                if not current_doc:
                    current_doc_offset = 0

            cpu_x_0[row_idx].copy_(torch.tensor(row_tokens, dtype=torch.long))
            for start, end, is_document_start in row_segments:
                segment_len = end - start
                if segment_len <= 0:
                    continue
                cpu_position_ids[row_idx, start:end] = torch.arange(segment_len, dtype=torch.long)
                cpu_boundary_mask[row_idx, start:end, start:end] = True
                if is_document_start:
                    cpu_document_start_mask[row_idx, start] = True

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

        device_x_0.copy_(cpu_x_0, non_blocking=pin_memory)
        device_attention_mask.copy_(cpu_attention_mask, non_blocking=pin_memory)
        device_loss_mask.copy_(cpu_loss_mask, non_blocking=pin_memory)
        device_position_ids.copy_(cpu_position_ids, non_blocking=pin_memory)
        device_boundary_mask.copy_(cpu_boundary_mask, non_blocking=pin_memory)
        device_document_start_mask.copy_(cpu_document_start_mask, non_blocking=pin_memory)
        yield DLMDataBatch(
            x_0=device_x_0,
            loss_mask=device_loss_mask,
            attention_mask=device_attention_mask,
            epoch=epoch,
            position_ids=device_position_ids,
            document_boundary_mask=device_boundary_mask,
            document_start_mask=device_document_start_mask,
        )
