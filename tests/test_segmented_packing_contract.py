from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from nanodllm.config import DataConfig
from nanodllm.data import ByteTokenizer, make_dataloader
from nanodllm.model import DiffusionConfig, DiffusionTransformer


def _write_parquet(path: Path, docs: list[str]) -> str:
    pq.write_table(pa.table({"text": docs}), path, row_group_size=1)
    return str(path)


def _data_config(path: str, **overrides) -> DataConfig:
    values = {
        "tokenizer": "byte",
        "train_data": [path],
        "val_data": [],
        "val_fraction": 0.0,
        "tokenizer_batch_size": 16,
        "shuffle_row_groups": False,
        "random_length_ratio": 0.0,
        "continuation_ratio": 0.0,
        "content_budget_segmented_packing": True,
    }
    values.update(overrides)
    return DataConfig(**values)


def _snapshot(batch) -> dict[str, torch.Tensor | int | None]:
    fields = (
        "x_0",
        "loss_mask",
        "attention_mask",
        "position_ids",
        "document_boundary_mask",
        "document_start_mask",
    )
    return {
        **{field: getattr(batch, field).clone() if getattr(batch, field) is not None else None for field in fields},
        "epoch": batch.epoch,
    }


def test_actual_model_logits_are_invariant_to_other_segment_tokens():
    torch.manual_seed(7)
    config = DiffusionConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_embd=16,
        mask_token_id=1,
        residual_output_projection_init="scaled_normal",
    )
    model = DiffusionTransformer(config, compute_dtype=torch.float32).eval()
    tokens_a = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.long)
    tokens_b = tokens_a.clone()
    tokens_b[:, 4:] = torch.tensor([[10, 11, 12, 13]])
    positions = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]], dtype=torch.long)
    boundary_mask = torch.zeros((1, 8, 8), dtype=torch.bool)
    boundary_mask[:, :4, :4] = True
    boundary_mask[:, 4:, 4:] = True

    with torch.inference_mode():
        unsegmented_a = model(tokens_a, position_ids=positions)
        unsegmented_b = model(tokens_b, position_ids=positions)
        segmented_a = model(tokens_a, attention_mask=boundary_mask, position_ids=positions)
        segmented_b = model(tokens_b, attention_mask=boundary_mask, position_ids=positions)

    assert not torch.allclose(unsegmented_a[:, :4], unsegmented_b[:, :4], atol=1e-6, rtol=1e-6)
    assert torch.allclose(segmented_a[:, :4], segmented_b[:, :4], atol=1e-6, rtol=1e-6)


def test_long_document_continues_without_synthetic_eos_at_row_splits(tmp_path):
    tokenizer = ByteTokenizer()
    path = _write_parquet(tmp_path / "data.parquet", ["abcdefghij", "z"])
    loader = make_dataloader(
        _data_config(path),
        tokenizer,
        batch_size=3,
        seq_len=4,
        split="train",
        device="cpu",
    )

    batch = next(loader)
    eos = tokenizer.get_eos_token_id()

    assert batch.x_0.tolist() == [
        [ord("a"), ord("b"), ord("c"), ord("d")],
        [ord("e"), ord("f"), ord("g"), ord("h")],
        [ord("i"), ord("j"), eos, ord("z")],
    ]
    assert eos not in batch.x_0[0].tolist()
    assert eos not in batch.x_0[1].tolist()
    assert batch.document_start_mask.tolist() == [
        [True, False, False, False],
        [False, False, False, False],
        [False, False, False, True],
    ]
    assert batch.position_ids.tolist() == [
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 0],
    ]
    assert bool(batch.document_boundary_mask[2, 0, 2])
    assert not bool(batch.document_boundary_mask[2, 2, 3])


def test_segmented_packing_replays_exactly_with_same_seed(tmp_path):
    tokenizer = ByteTokenizer()
    path = _write_parquet(tmp_path / "data.parquet", ["a", "bc", "def", "ghij", "klmno"])
    config = _data_config(
        path,
        shuffle_row_groups=True,
        shuffle_seed=29,
        random_length_ratio=0.4,
        continuation_ratio=0.4,
    )

    first_loader = make_dataloader(
        config, tokenizer, batch_size=3, seq_len=7, split="train", device="cpu", augmentation_seed=991
    )
    first = _snapshot(next(first_loader))
    second_loader = make_dataloader(
        config, tokenizer, batch_size=3, seq_len=7, split="train", device="cpu", augmentation_seed=991
    )
    second = _snapshot(next(second_loader))

    assert first["epoch"] == second["epoch"]
    for field in first.keys() - {"epoch"}:
        assert torch.equal(first[field], second[field])


def test_random_length_short_circuits_continuation_and_masks_effective_attention(tmp_path):
    tokenizer = ByteTokenizer()
    path = _write_parquet(tmp_path / "data.parquet", ["a", "bc", "defghij"])
    config = _data_config(path, random_length_ratio=0.5, continuation_ratio=0.5)
    loader = make_dataloader(
        config, tokenizer, batch_size=1, seq_len=6, split="train", device="cpu", augmentation_seed=9
    )

    batch = next(loader)

    assert batch.attention_mask.tolist() == [[True, True, True, False, False, False]]
    assert batch.loss_mask.tolist() == [[True, True, True, False, False, False]]
    valid = batch.attention_mask
    effective_mask = batch.document_boundary_mask & valid[:, :, None] & valid[:, None, :]
    assert not bool(effective_mask[:, 3:, :].any())
    assert not bool(effective_mask[:, :, 3:].any())


def test_continuation_masks_loss_prefix_without_crossing_segment_attention(tmp_path):
    tokenizer = ByteTokenizer()
    path = _write_parquet(tmp_path / "data.parquet", ["a", "bc", "defghij"])
    config = _data_config(path, random_length_ratio=0.5, continuation_ratio=0.5)
    loader = make_dataloader(
        config, tokenizer, batch_size=1, seq_len=6, split="train", device="cpu", augmentation_seed=26
    )

    batch = next(loader)

    assert batch.attention_mask.all()
    assert batch.loss_mask.tolist() == [[False, False, True, True, True, True]]
    assert batch.document_start_mask.tolist() == [[True, False, True, False, False, True]]
    assert bool(batch.document_boundary_mask[0, 0, 1])
    assert not bool(batch.document_boundary_mask[0, 1, 2])
    assert not bool(batch.document_boundary_mask[0, 4, 5])
