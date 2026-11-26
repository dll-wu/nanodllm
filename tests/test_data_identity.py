import hashlib
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from nanodllm.config import DataConfig, ExperimentConfig, ModelConfig, RuntimeConfig, TrainConfig, validate_config
from nanodllm.data import ParquetTextSource, PretrainedHFTokenizer, build_token_bytes
from nanodllm.data_identity import (
    assign_row_groups,
    build_data_identity,
    build_dataset_inventory,
    build_tokenizer_inventory,
)
from nanodllm.runtime.training_data_identity import _prepare_data_identity
from tokenizers import Tokenizer, models, pre_tokenizers


def _write_parquet(path: Path, rows: list[str], *, row_group_size: int = 1) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({"text": rows}),
        path,
        row_group_size=row_group_size,
        compression=None,
        use_dictionary=False,
        write_statistics=False,
    )
    return str(path)


def _config(
    train_path: str,
    *,
    tokenizer: str = "byte",
    tokenizer_path: str = "",
    val_fraction: float = 0.0,
    shuffle_row_groups: bool = False,
) -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(max_seq_len=16),
        data=DataConfig(
            tokenizer=tokenizer,
            tokenizer_path=tokenizer_path,
            train_data=[train_path],
            val_fraction=val_fraction,
            shuffle_row_groups=shuffle_row_groups,
            shuffle_seed=17,
            content_budget_segmented_packing=True,
        ),
        runtime=RuntimeConfig(seed=123, training_seed=-1),
        train=TrainConfig(device_batch_size=3),
    )


def _membership_keys(row_groups: list[dict]) -> list[tuple[str, int]]:
    return [(str(row_group["path"]), int(row_group["row_group"])) for row_group in row_groups]


def test_inventory_order_and_tail_split_membership_are_exact(tmp_path):
    a_path = Path(_write_parquet(tmp_path / "nested" / "a.parquet", ["a0", "a1", "a2"]))
    z_path = Path(_write_parquet(tmp_path / "z.parquet", ["z0", "z1"]))

    inventory = build_dataset_inventory(
        [str(z_path), str(tmp_path)],
        val_fraction=0.4,
    )

    canonical_a = str(a_path.resolve())
    canonical_z = str(z_path.resolve())
    assert [record["path"] for record in inventory["files"]] == [canonical_a, canonical_z]
    assert _membership_keys(inventory["membership"]["train"]["row_groups"]) == [
        (canonical_a, 0),
        (canonical_a, 1),
        (canonical_a, 2),
    ]
    assert _membership_keys(inventory["membership"]["validation"]["row_groups"]) == [
        (canonical_z, 0),
        (canonical_z, 1),
    ]
    assert inventory["membership"]["train"]["files"] == [canonical_a]
    assert inventory["membership"]["validation"]["files"] == [canonical_z]
    assert inventory["files"][0]["parquet_footer"]["num_row_groups"] == 3
    assert inventory["files"][0]["parquet_footer"]["row_group_num_rows"] == [1, 1, 1]
    assert inventory["metadata_inventory_sha256_is_content_hash"] is False
    assert "requested_train_exclude_patterns" not in inventory["selection"]
    assert "resolved_train_excluded_files" not in inventory["selection"]

    reordered = build_dataset_inventory([str(tmp_path), str(z_path)], val_fraction=0.4)
    assert reordered["metadata_inventory_sha256"] == inventory["metadata_inventory_sha256"]


def test_explicit_train_validation_overlap_is_rejected_after_canonicalization(tmp_path):
    data_path = Path(_write_parquet(tmp_path / "data.parquet", ["a", "b"]))
    alias_path = tmp_path / "alias.parquet"
    alias_path.symlink_to(data_path)

    with pytest.raises(ValueError, match="overlap after canonicalization"):
        build_dataset_inventory([str(data_path)], [str(alias_path)])

    data_config = DataConfig(
        tokenizer="byte",
        train_data=[str(data_path)],
        val_data=[str(alias_path)],
        val_fraction=0.0,
    )
    with pytest.raises(ValueError, match="overlap after canonicalization"):
        ParquetTextSource(data_config)


def test_explicit_train_exclusion_reserves_validation_with_loader_identity_parity(tmp_path):
    train_a = Path(_write_parquet(tmp_path / "shard_00000.parquet", ["a0", "a1"]))
    train_b = Path(_write_parquet(tmp_path / "shard_00001.parquet", ["b0"]))
    validation = Path(_write_parquet(tmp_path / "shard_00002.parquet", ["v0", "v1"]))
    data_config = DataConfig(
        tokenizer="byte",
        train_data=[str(tmp_path)],
        train_exclude_data=[str(validation)],
        val_data=[str(validation)],
        val_fraction=0.0,
        shuffle_row_groups=False,
    )

    source = ParquetTextSource(data_config)
    inventory = build_dataset_inventory(
        data_config.train_data,
        data_config.val_data,
        train_exclude_data=data_config.train_exclude_data,
        val_fraction=0.0,
    )

    expected_train = [str(train_a.resolve()), str(train_a.resolve()), str(train_b.resolve())]
    assert [row_group.path for row_group in source.train_row_groups] == expected_train
    assert [row_group.path for row_group in source.val_row_groups] == [
        str(validation.resolve()),
        str(validation.resolve()),
    ]
    assert _membership_keys(inventory["membership"]["train"]["row_groups"]) == [
        (str(train_a.resolve()), 0),
        (str(train_a.resolve()), 1),
        (str(train_b.resolve()), 0),
    ]
    assert inventory["selection"]["requested_train_exclude_patterns"] == [str(validation)]
    assert inventory["selection"]["resolved_train_excluded_files"] == [str(validation.resolve())]


def test_train_exclusions_fail_closed(tmp_path):
    included = Path(_write_parquet(tmp_path / "included" / "train.parquet", ["a"]))
    outside = Path(_write_parquet(tmp_path / "outside.parquet", ["b"]))

    with pytest.raises(FileNotFoundError, match="did not match"):
        build_dataset_inventory(
            [str(included)],
            train_exclude_data=[str(tmp_path / "missing.parquet")],
        )
    with pytest.raises(ValueError, match="outside the resolved train selection"):
        build_dataset_inventory([str(included)], train_exclude_data=[str(outside)])
    with pytest.raises(ValueError, match="removed every"):
        build_dataset_inventory([str(included)], train_exclude_data=[str(included)])

    alias = tmp_path / "included-alias.parquet"
    alias.symlink_to(included)
    inventory = build_dataset_inventory(
        [str(included), str(outside)],
        train_exclude_data=[str(alias)],
    )
    assert inventory["selection"]["resolved_train_excluded_files"] == [str(included.resolve())]
    assert inventory["membership"]["train"]["files"] == [str(outside.resolve())]


def test_config_rejects_train_exclusions_without_an_explicit_parquet_selection():
    missing_train = _config("unused.parquet")
    missing_train.data.train_data = []
    missing_train.data.train_exclude_data = ["heldout.parquet"]
    with pytest.raises(ValueError, match="requires data.train_data"):
        validate_config(missing_train)

    synthetic = _config("unused.parquet")
    synthetic.data.synthetic_dataset = "eos_overfit_canary"
    synthetic.data.train_exclude_data = ["heldout.parquet"]
    with pytest.raises(ValueError, match="cannot be combined"):
        validate_config(synthetic)


def test_training_identity_artifact_is_stable_and_blocks_changed_resume(tmp_path):
    data_path = Path(_write_parquet(tmp_path / "data.parquet", ["a", "b"]))
    config = _config(str(data_path))
    config.data.identity_mode = "metadata"
    config.train.run_dir = str(tmp_path / "run")
    Path(config.train.run_dir).mkdir()

    first = _prepare_data_identity(config, rank=0, world_size=1, is_ddp=False)
    second = _prepare_data_identity(config, rank=0, world_size=1, is_ddp=False)

    assert first == second
    assert first["enabled"] is True
    assert len(first["identity_sha256"]) == 64
    assert (Path(config.train.run_dir) / "data_identity.v1.json").exists()

    _write_parquet(data_path, ["a", "b", "c"])
    with pytest.raises(RuntimeError, match="changed data identity"):
        _prepare_data_identity(config, rank=0, world_size=1, is_ddp=False)


def test_full_hash_detects_same_metadata_content_mutation(tmp_path):
    data_path = Path(_write_parquet(tmp_path / "data.parquet", ["alpha1", "bravo2"], row_group_size=2))
    fixed_mtime_ns = 1_700_000_000_000_000_000
    os.utime(data_path, ns=(fixed_mtime_ns, fixed_mtime_ns))
    first = build_data_identity(_config(str(data_path)), full_sha256=True)
    first_size = data_path.stat().st_size

    _write_parquet(data_path, ["charl3", "delta4"], row_group_size=2)
    assert data_path.stat().st_size == first_size
    os.utime(data_path, ns=(fixed_mtime_ns, fixed_mtime_ns))
    second = build_data_identity(_config(str(data_path)), full_sha256=True)

    assert first["hashes"]["metadata_inventory_sha256"] == second["hashes"]["metadata_inventory_sha256"]
    assert first["dataset"]["files"][0]["full_sha256"] != second["dataset"]["files"][0]["full_sha256"]
    assert first["hashes"]["full_content_inventory_sha256"] != second["hashes"]["full_content_inventory_sha256"]
    assert first["hashes"]["identity_sha256"] != second["hashes"]["identity_sha256"]


def test_metadata_identity_changes_for_footer_membership_mutation(tmp_path):
    data_path = Path(_write_parquet(tmp_path / "data.parquet", ["a", "b"]))
    first = build_data_identity(_config(str(data_path)))

    _write_parquet(data_path, ["a", "b", "c"])
    second = build_data_identity(_config(str(data_path)))

    assert first["hashes"]["metadata_inventory_sha256"] != second["hashes"]["metadata_inventory_sha256"]
    assert first["dataset"]["membership"]["train"]["row_group_count"] == 2
    assert second["dataset"]["membership"]["train"]["row_group_count"] == 3


def test_tokenizer_files_and_token_bytes_are_content_hashed_in_stable_order(tmp_path):
    data_path = _write_parquet(tmp_path / "data.parquet", ["a", "b"])
    tokenizer_path = tmp_path / "tokenizer"
    (tokenizer_path / "nested").mkdir(parents=True)
    (tokenizer_path / "tokenizer.json").write_bytes(b'{"version": 1}')
    (tokenizer_path / "nested" / "added_tokens.json").write_bytes(b"{}")
    token_bytes = b"not-loaded-as-a-pickle"
    (tokenizer_path / "token_bytes.pt").write_bytes(token_bytes)

    identity = build_data_identity(
        _config(data_path, tokenizer="hf", tokenizer_path=str(tokenizer_path)),
    )
    tokenizer = identity["tokenizer"]

    assert [record["relative_path"] for record in tokenizer["files"]] == [
        "nested/added_tokens.json",
        "token_bytes.pt",
        "tokenizer.json",
    ]
    assert tokenizer["token_bytes_pt_sha256"] == hashlib.sha256(token_bytes).hexdigest()
    assert tokenizer["token_bytes_source"] == "token_bytes.pt"
    assert identity["hashes"]["identity_sha256_strength"] == "dataset_metadata_and_tokenizer_content"


def test_safe_json_token_byte_table_is_loaded_and_content_hashed(tmp_path):
    class FakeHFTokenizer:
        mask_token_id = 5
        eos_token_id = 4
        pad_token_id = 2
        bos_token_id = 3
        all_special_ids = [2, 3, 4, 5]

        def __len__(self):
            return 6

    tokenizer_path = tmp_path / "tokenizer"
    tokenizer_path.mkdir()
    token_bytes_path = tokenizer_path / "token_bytes.json"
    token_bytes_path.write_text(
        json.dumps(
            {
                "format": "nanodllm.token_bytes.v1",
                "dtype": "int32",
                "vocab_size": 6,
                "special_token_ids": [2, 3, 4, 5],
                "values": [1, 2, 0, 0, 0, 0],
            }
        ),
        encoding="utf-8",
    )
    wrapped = PretrainedHFTokenizer(FakeHFTokenizer(), str(tokenizer_path))

    assert build_token_bytes(wrapped).tolist() == [1, 2, 0, 0, 0, 0]
    inventory = build_tokenizer_inventory("hf", tokenizer_path)
    assert inventory["token_bytes_source"] == "token_bytes.json"
    assert inventory["token_bytes_json_sha256"] == hashlib.sha256(token_bytes_path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "declared_special_ids",
    [
        [2, 3, 4, 5, 5],
        [2, 3, 4, True],
        [2, 3, 4, 6],
        [2, 3, 4, {"id": 5}],
    ],
)
def test_safe_json_token_byte_table_rejects_malformed_special_ids(tmp_path, declared_special_ids):
    class FakeHFTokenizer:
        mask_token_id = 5
        eos_token_id = 4
        pad_token_id = 2
        bos_token_id = 3
        all_special_ids = [2, 3, 4, 5]

        def __len__(self):
            return 6

    tokenizer_path = tmp_path / "tokenizer"
    tokenizer_path.mkdir()
    (tokenizer_path / "token_bytes.json").write_text(
        json.dumps(
            {
                "format": "nanodllm.token_bytes.v1",
                "dtype": "int32",
                "vocab_size": 6,
                "special_token_ids": declared_special_ids,
                "values": [1, 2, 0, 0, 0, 0],
            }
        ),
        encoding="utf-8",
    )
    wrapped = PretrainedHFTokenizer(FakeHFTokenizer(), str(tokenizer_path))

    with pytest.raises(ValueError, match="special-token IDs"):
        build_token_bytes(wrapped)


def test_historical_hf_tokenizer_fallback_preserves_legacy_encode_calls(tmp_path):
    class FakeHFTokenizer:
        mask_token_id = 5
        eos_token_id = 4
        pad_token_id = 2
        bos_token_id = 3
        all_special_ids = [2, 3, 4, 5]

        def __init__(self):
            self.calls = []

        def encode(self, text, **kwargs):
            self.calls.append(("single", text, kwargs))
            return [1]

        def __call__(self, text, **kwargs):
            self.calls.append(("batch", text, kwargs))
            return {"input_ids": [[1] for _ in text]}

        def __len__(self):
            return 6

    backend = FakeHFTokenizer()
    wrapped = PretrainedHFTokenizer(backend, str(tmp_path))

    assert wrapped.encode("<|mask|>") == [1]
    assert wrapped.encode(["<|mask|>"]) == [[1]]
    assert backend.calls == [
        ("single", "<|mask|>", {"add_special_tokens": False}),
        (
            "batch",
            ["<|mask|>"],
            {"add_special_tokens": False, "return_attention_mask": False},
        ),
    ]


def test_ordinary_sidecar_requires_exact_base_vocab_mapping_and_handles_scalar_and_batch(tmp_path):
    class FakeHFTokenizer:
        mask_token_id = 6
        eos_token_id = 7
        pad_token_id = 8
        bos_token_id = 9
        all_special_ids = [6, 7, 8, 9]

        def __init__(self, backend_tokenizer):
            self.backend_tokenizer = backend_tokenizer

        def __len__(self):
            return 10

    def make_backend(vocab):
        backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
        backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        return backend

    base_vocab = {"[UNK]": 0, "literal": 1, "text": 2}
    matching = make_backend(base_vocab)
    matching.save(str(tmp_path / "tokenizer_ordinary.json"))
    wrapped = PretrainedHFTokenizer(FakeHFTokenizer(make_backend(base_vocab)), str(tmp_path))

    assert wrapped.encode("literal text") == [1, 2]
    assert wrapped.encode(["literal text", "<|mask|>"]) == [[1, 2], [0]]
    assert wrapped.get_mask_token_id() not in wrapped.encode("<|mask|>")

    mismatched_path = tmp_path / "mismatched"
    mismatched_path.mkdir()
    make_backend({"[UNK]": 0, "literal": 2, "text": 1}).save(str(mismatched_path / "tokenizer_ordinary.json"))
    with pytest.raises(ValueError, match="same base token-to-ID mapping"):
        PretrainedHFTokenizer(FakeHFTokenizer(make_backend(base_vocab)), str(mismatched_path))


def test_ddp_stride_membership_is_deterministic_disjoint_and_complete(tmp_path):
    data_path = _write_parquet(tmp_path / "data.parquet", [f"row-{index}" for index in range(11)])
    identity = build_data_identity(_config(data_path, shuffle_row_groups=True), world_size=3, assignment_epoch=4)
    membership = identity["dataset"]["membership"]["train"]["row_groups"]

    assignments = [
        assign_row_groups(membership, rank=rank, world_size=3, shuffle_seed=17, epoch=4) for rank in range(3)
    ]
    replay = [assign_row_groups(membership, rank=rank, world_size=3, shuffle_seed=17, epoch=4) for rank in range(3)]
    assignment_sets = [set(_membership_keys(rows)) for rows in assignments]

    assert assignments == replay
    assert assignment_sets[0].isdisjoint(assignment_sets[1])
    assert assignment_sets[0].isdisjoint(assignment_sets[2])
    assert assignment_sets[1].isdisjoint(assignment_sets[2])
    assert set.union(*assignment_sets) == set(_membership_keys(membership))
    assert [summary["train_row_group_count"] for summary in identity["distributed_membership"]["rank_summaries"]] == [
        4,
        4,
        3,
    ]


def test_packing_identity_records_dimensions_seed_and_segmented_settings(tmp_path):
    data_path = _write_parquet(tmp_path / "data.parquet", ["a", "b"])
    identity = build_data_identity(_config(data_path))

    packing = identity["packing"]
    assert packing["mode"] == "content_budget_segmented"
    assert packing["sequence_length"] == 16
    assert packing["device_batch_size"] == 3
    assert packing["resolved_training_seed"] == 123
    assert packing["mask_augmentation_rng"] == "loader-local random.Random"
    assert packing["mask_augmentation_seed_rule"] == "resolved_training_seed + rank"
    assert packing["data"]["append_eos"] is True
    assert packing["data"]["content_budget_segmented_packing"] is True
