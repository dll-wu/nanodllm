import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from nanodllm.config import DataConfig
from nanodllm.data import ByteTokenizer, make_dataloader


def _write_parquet(path: Path, docs: list[str]) -> str:
    table = pa.table({"text": docs})
    pq.write_table(table, path, row_group_size=1)
    return str(path)


def _data_config(path: str, **overrides) -> DataConfig:
    values = {
        "tokenizer": "byte",
        "train_data": [path],
        "val_data": [],
        "val_fraction": 0.0,
        "tokenizer_batch_size": 8,
        "shuffle_row_groups": False,
        "random_length_ratio": 0.0,
        "continuation_ratio": 0.0,
    }
    values.update(overrides)
    return DataConfig(**values)


def test_stream_packing_preserves_document_tails(tmp_path):
    tok = ByteTokenizer()
    path = _write_parquet(tmp_path / "data.parquet", ["abc", "defgh"])
    loader = make_dataloader(_data_config(path), tok, batch_size=2, seq_len=5, split="train", device="cpu")

    batch = next(loader)
    expected = [
        [97, 98, 99, tok.get_eos_token_id(), 100],
        [101, 102, 103, 104, tok.get_eos_token_id()],
    ]
    assert batch.x_0.tolist() == expected
    assert batch.attention_mask.all()
    assert batch.loss_mask.all()


def test_eos_only_at_document_boundaries(tmp_path):
    tok = ByteTokenizer()
    path = _write_parquet(tmp_path / "data.parquet", ["abcdef"])
    loader = make_dataloader(_data_config(path), tok, batch_size=2, seq_len=4, split="train", device="cpu")

    batch = next(loader)
    assert batch.x_0[0].tolist() == [97, 98, 99, 100]
    assert tok.get_eos_token_id() not in batch.x_0[0].tolist()
    assert batch.x_0[1, 2].item() == tok.get_eos_token_id()


def test_loss_mask_protects_bos_but_not_eos(tmp_path):
    tok = ByteTokenizer()
    path = _write_parquet(tmp_path / "data.parquet", ["ab", "cd"])
    cfg = _data_config(path, prepend_bos=True, loss_on_bos=False, loss_on_eos=True)
    loader = make_dataloader(cfg, tok, batch_size=2, seq_len=4, split="train", device="cpu")

    batch = next(loader)
    bos_positions = batch.x_0 == tok.get_bos_token_id()
    eos_positions = batch.x_0 == tok.get_eos_token_id()
    assert not batch.loss_mask[bos_positions].any()
    assert batch.loss_mask[eos_positions].all()


def test_random_length_sets_suffix_invalid(tmp_path):
    tok = ByteTokenizer()
    path = _write_parquet(tmp_path / "data.parquet", ["abcdefghij"] * 4)
    cfg = _data_config(path, random_length_ratio=1.0, random_length_min=1)
    loader = make_dataloader(cfg, tok, batch_size=4, seq_len=8, split="train", device="cpu", augmentation_seed=0)
    batch = next(loader)
    assert (~batch.attention_mask).any()
    assert torch.equal(batch.loss_mask & ~batch.attention_mask, torch.zeros_like(batch.loss_mask))


def test_continuation_prefix_is_visible_not_lossed(tmp_path):
    tok = ByteTokenizer()
    path = _write_parquet(tmp_path / "data.parquet", ["abcdefghij"] * 4)
    cfg = _data_config(
        path,
        continuation_ratio=1.0,
        continuation_min_prefix_frac=0.25,
        continuation_max_prefix_frac=0.5,
    )
    loader = make_dataloader(cfg, tok, batch_size=2, seq_len=8, split="train", device="cpu", augmentation_seed=0)
    batch = next(loader)
    assert batch.attention_mask.all()
    for row in batch.loss_mask:
        first_loss = row.nonzero(as_tuple=True)[0][0].item()
        assert first_loss > 0
        assert not row[:first_loss].any()
        assert row[first_loss:].all()


def test_augmented_train_loader_replays_after_resume_fast_forward_without_global_rng_consumption(tmp_path):
    tok = ByteTokenizer()
    path = _write_parquet(tmp_path / "data.parquet", ["abcdefghij", "klmnopqrst", "uvwxyz"] * 4)
    cfg = _data_config(
        path,
        random_length_ratio=0.5,
        random_length_min=2,
        continuation_ratio=0.5,
        continuation_min_prefix_frac=0.25,
        continuation_max_prefix_frac=0.5,
    )
    loader_kwargs = {
        "batch_size": 4,
        "seq_len": 8,
        "split": "train",
        "device": "cpu",
        "augmentation_seed": 314159,
    }

    continuous_loader = make_dataloader(cfg, tok, **loader_kwargs)
    continuous_batches = []
    for _ in range(3):
        batch = next(continuous_loader)
        continuous_batches.append(
            {
                "x_0": batch.x_0.clone(),
                "loss_mask": batch.loss_mask.clone(),
                "attention_mask": batch.attention_mask.clone(),
                "epoch": batch.epoch,
            }
        )
    expected = continuous_batches[2]
    assert not expected["attention_mask"].all() or not expected["loss_mask"].all()

    random.seed(271828)
    global_rng_state = random.getstate()
    resumed_loader = make_dataloader(cfg, tok, **loader_kwargs)
    resumed_batch = next(resumed_loader)
    for _ in range(2):
        resumed_batch = next(resumed_loader)

    assert random.getstate() == global_rng_state
    assert resumed_batch.epoch == expected["epoch"]
    assert torch.equal(resumed_batch.x_0, expected["x_0"])
    assert torch.equal(resumed_batch.loss_mask, expected["loss_mask"])
    assert torch.equal(resumed_batch.attention_mask, expected["attention_mask"])


def test_boundary_aware_packing_resets_positions_and_blocks_cross_document_attention(tmp_path):
    tok = ByteTokenizer()
    path = _write_parquet(tmp_path / "data.parquet", ["a", "b", "c"])
    cfg = _data_config(path, prepend_bos=True, boundary_aware_packing=True)
    loader = make_dataloader(cfg, tok, batch_size=2, seq_len=8, split="train", device="cpu")

    batch = next(loader)
    bos = tok.get_bos_token_id()
    eos = tok.get_eos_token_id()
    pad = tok.get_pad_token_id()

    assert batch.x_0[0].tolist() == [bos, ord("a"), eos, bos, ord("b"), eos, pad, pad]
    assert batch.position_ids is not None
    assert batch.position_ids[0].tolist() == [0, 1, 2, 0, 1, 2, 0, 0]
    assert batch.document_boundary_mask is not None
    mask = batch.document_boundary_mask[0]
    assert bool(mask[1, 0])
    assert bool(mask[2, 2])
    assert not bool(mask[0, 1])
    assert not bool(mask[3, 2])
    assert not bool(mask[2, 3])
    assert not bool(mask[:, 6:].any())
    assert not bool(batch.loss_mask[0, 0])
    assert bool(batch.loss_mask[0, 2])


def test_content_budget_segmented_packing_fills_rows_and_blocks_cross_segment_attention(tmp_path):
    tok = ByteTokenizer()
    path = _write_parquet(tmp_path / "data.parquet", ["a", "bc", "defgh"])
    cfg = _data_config(path, content_budget_segmented_packing=True)
    loader = make_dataloader(cfg, tok, batch_size=2, seq_len=6, split="train", device="cpu")

    batch = next(loader)
    eos = tok.get_eos_token_id()

    assert batch.x_0[0].tolist() == [ord("a"), eos, ord("b"), ord("c"), eos, ord("d")]
    assert batch.attention_mask.all()
    assert batch.document_start_mask is not None
    assert batch.document_start_mask[0].tolist() == [True, False, True, False, False, True]
    assert batch.position_ids is not None
    assert batch.position_ids[0].tolist() == [0, 1, 0, 1, 2, 0]
    assert batch.document_boundary_mask is not None
    mask = batch.document_boundary_mask[0]
    assert bool(mask[0, 1])
    assert bool(mask[1, 0])
    assert bool(mask[2, 4])
    assert bool(mask[4, 2])
    assert not bool(mask[1, 2])
    assert not bool(mask[4, 5])
    assert batch.document_boundary_mask.dtype == torch.bool


def test_short_document_curriculum_caps_single_documents_and_keeps_eos(tmp_path):
    tok = ByteTokenizer()
    path = _write_parquet(tmp_path / "data.parquet", ["abc", "abcdef"])
    cfg = _data_config(
        path,
        prepend_bos=True,
        short_document_curriculum=True,
        short_document_max_tokens=5,
    )
    loader = make_dataloader(cfg, tok, batch_size=2, seq_len=8, split="train", device="cpu")

    batch = next(loader)
    bos = tok.get_bos_token_id()
    eos = tok.get_eos_token_id()
    pad = tok.get_pad_token_id()

    assert batch.x_0[0].tolist() == [bos, ord("a"), ord("b"), ord("c"), eos, pad, pad, pad]
    assert batch.x_0[1].tolist() == [bos, ord("a"), ord("b"), ord("c"), eos, pad, pad, pad]
    assert batch.attention_mask.tolist() == [[True] * 5 + [False] * 3, [True] * 5 + [False] * 3]
    assert not bool(batch.loss_mask[0, 0])
    assert not bool(batch.loss_mask[1, 0])
    assert bool(batch.loss_mask[0, 4])
    assert bool(batch.loss_mask[1, 4])
    assert batch.document_boundary_mask is None
