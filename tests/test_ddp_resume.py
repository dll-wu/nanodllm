import copy
import json
import os
import random
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from nanodllm.config import (
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    ModelConfig,
    NoiseConfig,
    RuntimeConfig,
    TrainConfig,
    to_dict,
)
from nanodllm.data import ByteTokenizer, DLMDataBatch, build_token_bytes, build_train_loader
from nanodllm.training import (
    LOGIT_SCALE_HIST_BINS,
    PER_TOKEN_NOISE_LEVEL_HIST_BINS,
    TOP1_PROB_HIST_BINS,
    _batch_six_tensor_digests,
    _load_resume_contract,
    _new_eos_hazard_accumulator,
    _new_eos_supervision_counts,
    _new_length_head_accumulator,
    _new_position_decile_ce_accumulator,
    _new_token_edit_reconstruction_accumulator,
    _resume_trajectory_contract,
    _verify_replayed_next_batch,
)

REPO = Path(__file__).resolve().parents[1]


def _write_rank_distinct_parquet(path: Path) -> str:
    documents = [
        "a",
        "bbbbbbbbbbbbbbb",
        "cc",
        "dddddddddddddd",
        "eee",
        "fffffffffffff",
        "gggg",
        "hhhhhhhhhhhh",
    ]
    pq.write_table(pa.table({"text": documents}), path, row_group_size=1)
    return str(path)


def _config(data_path: str, run_dir: Path) -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(depth=1, n_embd=16, n_head=2, max_seq_len=8, pad_vocab_to=1),
        data=DataConfig(
            tokenizer="byte",
            train_data=[data_path],
            val_data=[],
            val_fraction=0.0,
            tokenizer_batch_size=1,
            shuffle_row_groups=False,
            random_length_ratio=0.0,
            continuation_ratio=0.0,
            identity_mode="metadata",
        ),
        noise=NoiseConfig(t_min=0.8, t_max=0.8, complement_masking=True),
        runtime=RuntimeConfig(
            device_type="cpu",
            dtype="float32",
            compile=False,
            seed=101,
            model_seed=202,
            training_seed=303,
        ),
        train=TrainConfig(
            run_dir=str(run_dir),
            num_steps=4,
            total_batch_size=32,
            device_batch_size=1,
            log_interval=1,
            tensorboard=False,
            save_every=2,
        ),
        eval=EvalConfig(enabled=False, gen_eval=False, fixed_ce_trace_enabled=False),
    )


def _launch(config_path: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(REPO),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    if sys.platform == "darwin":
        env["GLOO_SOCKET_IFNAME"] = "lo0"
    elif sys.platform.startswith("linux"):
        env["GLOO_SOCKET_IFNAME"] = "lo"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        master_port = listener.getsockname()[1]
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc-per-node=2",
            "--master-addr=127.0.0.1",
            f"--master-port={master_port}",
            str(REPO / "pretrain.py"),
            "--config",
            str(config_path),
        ],
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )


def _assert_nested_equal(left, right) -> None:
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, list | tuple):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def _trajectory(config: ExperimentConfig) -> dict[str, object]:
    return _resume_trajectory_contract(
        config,
        device=torch.device("cpu"),
        model_config={"depth": config.model.depth, "n_embd": config.model.n_embd},
        architecture_fingerprint={"fingerprint_sha256": "a" * 64},
        data_identity_summary={
            "enabled": True,
            "mode": "metadata",
            "identity_sha256": "b" * 64,
            "identity_sha256_strength": "metadata",
        },
        total_batch_size=32,
        grad_accum_steps=2,
        total_steps=int(config.train.num_steps),
        warmup_steps=4,
        lr_schedule={"name": "warmup_stable_warmdown", "warmup_steps": 4},
    )


def _valid_trainer_state() -> dict[str, object]:
    lineage = {
        "ar_boundary_counts": {
            "microbatches": 0,
            "microbatches_with_bos": 0,
            "valid_tokens": 0,
            "special_tokens": 0,
            "special_loss_tokens": 0,
            "bos_tokens": 0,
            "bos_loss_tokens": 0,
            "eos_tokens": 0,
            "eos_loss_tokens": 0,
        },
        "boundary_telemetry": {
            "documents_per_row_sum": 0.0,
            "documents_per_row_count": 0,
            "pad_tokens": 0,
            "total_row_tokens": 0,
            "blocked_pairs_sum": 0,
            "causal_pairs_sum": 0,
            "valid_pairs_sum": 0,
            "allowed_pairs_sum": 0,
            "segment_blocked_pairs_sum": 0,
        },
        "row_telemetry": {
            "rows": 0,
            "valid_tokens_sum": 0,
            "valid_tokens_min": 0,
            "valid_tokens_max": 0,
            "lossable_tokens_sum": 0,
            "lossable_tokens_min": 0,
            "lossable_tokens_max": 0,
            "pad_tokens": 0,
            "total_row_tokens": 0,
            "eos_rows": 0,
            "eos_tokens": 0,
            "eos_loss_tokens": 0,
        },
        "eos_supervision_train_counts": _new_eos_supervision_counts(),
        "block_causal_visible_prefix_frac_sum": 0.0,
        "block_causal_visible_prefix_count": 0,
        "train_m2t_ce_sum": 0.0,
        "train_m2t_loss_sum": 0.0,
        "train_m2t_loss_count": 0,
        "train_m2t_microbatches": 0,
        "train_m2t_active_tokens": 0,
        "train_t2t_ce_sum": 0.0,
        "train_t2t_loss_sum": 0.0,
        "train_t2t_loss_count": 0,
        "train_t2t_microbatches": 0,
        "train_t2t_active_tokens": 0,
        "train_length_head_accumulator": _new_length_head_accumulator(8),
        "train_token_edit_reconstruction_accumulator": _new_token_edit_reconstruction_accumulator(),
        "train_per_token_noise_level_hist_counts": [0 for _ in range(PER_TOKEN_NOISE_LEVEL_HIST_BINS)],
        "train_per_token_effective_masked_fraction_sum": 0.0,
        "train_per_token_effective_masked_fraction_count": 0,
        "train_per_token_effective_masked_fraction_min": float("inf"),
        "train_per_token_effective_masked_fraction_max": float("-inf"),
        "grad_clip_event_count": 0,
        "grad_clip_observed_steps": 0,
    }
    interval = {
        "smooth_loss": 0.0,
        "smooth_count": 0,
        "m2t_ce_sum": 0.0,
        "m2t_loss_sum": 0.0,
        "m2t_loss_count": 0,
        "m2t_microbatches": 0,
        "m2t_active_tokens": 0,
        "t2t_ce_sum": 0.0,
        "t2t_loss_sum": 0.0,
        "t2t_loss_count": 0,
        "t2t_microbatches": 0,
        "t2t_active_tokens": 0,
    }
    last = {
        "loss_value": 0.0,
        "main_ce": 0.0,
        "main_loss": 0.0,
        "confidence_penalty": 0.0,
        "confidence_penalty_loss": 0.0,
        "z_loss": 0.0,
        "z_loss_loss": 0.0,
        "eos_hazard_calibration_loss": 0.0,
        "eos_hazard_loss": 0.0,
        "eos_hazard_accumulator": _new_eos_hazard_accumulator(),
        "length_head_accumulator": _new_length_head_accumulator(8),
        "position_decile_ce_accumulator": _new_position_decile_ce_accumulator(),
        "token_edit_reconstruction_accumulator": _new_token_edit_reconstruction_accumulator(),
        "m2t_ce": 0.0,
        "m2t_loss": 0.0,
        "m2t_microbatches": 0,
        "m2t_active_tokens": 0,
        "t2t_ce": 0.0,
        "t2t_loss": 0.0,
        "t2t_microbatches": 0,
        "t2t_active_tokens": 0,
        "grad_norm": 0.0,
        "grad_clipped": False,
        "grad_clip_ratio": 1.0,
        "active_tokens": 0,
        "lossable_tokens": 0,
        "valid_tokens": 0,
        "content_tokens": 0,
        "content_bytes": 0,
        "class_ce": 0.0,
        "within_class_ce": 0.0,
        "ar_unsmoothed_ce": 0.0,
        "second_moment_norm": 0.0,
        "update_norm": 0.0,
        "update_grad_ratio": 0.0,
        "entropy_mean": 0.0,
        "top1_prob_mean": 0.0,
        "argmax_mode_frac": 0.0,
        "top1_prob_hist_counts": [0 for _ in range(TOP1_PROB_HIST_BINS)],
        "entropy_hist_counts": [0 for _ in range(LOGIT_SCALE_HIST_BINS)],
        "logsumexp_hist_counts": [0 for _ in range(LOGIT_SCALE_HIST_BINS)],
        "logit_norm_hist_counts": [0 for _ in range(LOGIT_SCALE_HIST_BINS)],
        "logit_norm_mean": 0.0,
        "logit_norm_min": 0.0,
        "logit_norm_max": 0.0,
        "logit_rms_mean": 0.0,
        "logsumexp_mean": 0.0,
        "logsumexp_min": 0.0,
        "logsumexp_max": 0.0,
        "soft_input_stats": {},
        "uniform_estimator_stats": {},
        "per_token_noise_level_hist_counts": [0 for _ in range(PER_TOKEN_NOISE_LEVEL_HIST_BINS)],
        "per_token_effective_masked_fraction_sum": 0.0,
        "per_token_effective_masked_fraction_count": 0,
        "per_token_effective_masked_fraction_min": float("inf"),
        "per_token_effective_masked_fraction_max": float("-inf"),
        "gradient_component_norms": {},
        "update_component_norms": {},
    }
    diagnostics = {
        "initial_parameter_norm_metrics": {},
        "initial_gradient_metrics": {},
        "initial_update_metrics": {},
        "initial_lm_head_row_norm_metrics": {},
        "early_lm_head_row_norm_metrics": {},
        "early_training_diagnostics": {},
        "early_residual_diagnostics": {},
        "fixed_ce_trace_records": 0,
        "last_fixed_ce_trace_record": {},
    }
    return {"version": 1, "lineage": lineage, "interval": interval, "last": last, "diagnostics": diagnostics}


def _valid_rank_state(rank: int = 0) -> dict[str, object]:
    x_0 = torch.zeros((1, 8), dtype=torch.long)
    digest = _batch_six_tensor_digests(
        DLMDataBatch(
            x_0=x_0,
            loss_mask=torch.ones_like(x_0, dtype=torch.bool),
            attention_mask=torch.ones_like(x_0, dtype=torch.bool),
            epoch=0,
        )
    )
    return {
        "rank": rank,
        "rng_state": {
            "version": 1,
            "torch_cpu": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
            "device_type": "cpu",
            "distributed_backend": "none",
            "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
            "local_world_size": int(os.environ.get("LOCAL_WORLD_SIZE", "1")),
            "group_rank": int(os.environ.get("GROUP_RANK", "0")),
            "cuda": None,
        },
        "next_batch_six_tensor_digests": digest,
        "trainer_state": _valid_trainer_state(),
    }


def _contract_checkpoint(
    trajectory: dict[str, object],
    *,
    step: int = 1,
    grad_accum_steps: int = 2,
    world_size: int = 1,
) -> dict[str, object]:
    consumed = step * grad_accum_steps
    return {
        "step": step,
        "resume_contract": {
            "version": 1,
            "world_size": world_size,
            "grad_accum_steps": grad_accum_steps,
            "consumed_microbatches": consumed,
            "consumed_microbatches_scope": "processed_microbatches_per_rank",
            "replay_advance_count_per_rank": consumed,
            "prefetched_next_batch_count_per_rank": 1,
            "loader_yields_per_rank": consumed + 1,
            "global_processed_microbatches": consumed * world_size,
            "checkpoint_boundary": "optimizer_step",
            "trajectory": trajectory,
            "rank_states": [_valid_rank_state(rank) for rank in range(world_size)],
        },
    }


def _load_test_contract(checkpoint: dict[str, object], config: ExperimentConfig, trajectory: dict[str, object]):
    return _load_resume_contract(
        checkpoint,
        rank=0,
        world_size=1,
        grad_accum_steps=2,
        checkpoint_step=int(checkpoint["step"]),
        trajectory_contract=trajectory,
        config=config,
        device=torch.device("cpu"),
    )


def _expected_data_counters(config: ExperimentConfig, *, world_size: int, microbatches: int) -> dict[str, int]:
    tokenizer = ByteTokenizer()
    token_bytes = build_token_bytes(tokenizer)
    counters = {"processed": 0, "valid": 0, "content": 0, "content_bytes": 0}
    for rank in range(world_size):
        loader = build_train_loader(config, tokenizer, rank, world_size, torch.device("cpu"))
        for _ in range(microbatches):
            batch = next(loader)
            byte_values = token_bytes[batch.x_0]
            counters["processed"] += int(batch.x_0.numel())
            counters["valid"] += int(batch.attention_mask.sum().item())
            counters["content"] += int(((byte_values > 0) & batch.attention_mask).sum().item())
            counters["content_bytes"] += int(byte_values[batch.attention_mask].sum().item())
    return counters


def _lineage_metrics(metrics: dict[str, object]) -> dict[str, object]:
    excluded_exact = {
        "start_step",
        "training_seconds",
        "peak_vram_mb",
        "resume_contract_version",
        "resume_contract_status",
        "resume_consumed_microbatches",
        "resume_next_batch_digest_verified",
    }
    return {
        key: value
        for key, value in metrics.items()
        if key not in excluded_exact
        and not key.startswith("session_")
        and not key.startswith("avg_")
        and not key.endswith("_path")
        and not key.endswith("evaluation_seconds")
    }


def test_six_tensor_batch_digest_covers_optional_loader_state():
    x_0 = torch.arange(8, dtype=torch.long).view(1, 8)
    batch = DLMDataBatch(
        x_0=x_0,
        loss_mask=torch.ones_like(x_0, dtype=torch.bool),
        attention_mask=torch.ones_like(x_0, dtype=torch.bool),
        epoch=3,
        position_ids=torch.arange(8, dtype=torch.long).view(1, 8),
        document_boundary_mask=torch.ones((1, 8, 8), dtype=torch.bool),
        document_start_mask=torch.tensor([[True, False, False, True, False, False, False, False]]),
    )

    digest = _batch_six_tensor_digests(batch)
    changed = DLMDataBatch(
        x_0=x_0,
        loss_mask=batch.loss_mask,
        attention_mask=batch.attention_mask,
        epoch=3,
        position_ids=batch.position_ids,
        document_boundary_mask=batch.document_boundary_mask.clone(),
        document_start_mask=batch.document_start_mask,
    )
    changed.document_boundary_mask[0, 0, 0] = False

    assert digest["tensor_fields"] == [
        "x_0",
        "loss_mask",
        "attention_mask",
        "position_ids",
        "document_boundary_mask",
        "document_start_mask",
    ]
    assert all(digest["tensors"][field]["present"] for field in digest["tensor_fields"])
    assert _batch_six_tensor_digests(changed)["combined_sha256"] != digest["combined_sha256"]


def test_multirank_resume_rejects_legacy_checkpoint():
    with pytest.raises(RuntimeError, match="predates the versioned resume contract"):
        _load_resume_contract(
            {"rng_state": {"torch": torch.get_rng_state()}},
            rank=0,
            world_size=2,
            grad_accum_steps=1,
            checkpoint_step=1,
        )


def test_trajectory_contract_binds_training_recipe_but_allows_operational_paths(tmp_path):
    config = _config(str(tmp_path / "train.parquet"), tmp_path / "first")
    baseline = _trajectory(config)

    operational = copy.deepcopy(config)
    operational.train.run_dir = str(tmp_path / "second")
    operational.train.resume_from = str(tmp_path / "checkpoint.pt")
    operational.train.save_every = 3
    assert _trajectory(operational)["sha256"] == baseline["sha256"]

    mutations = []
    for section, field, value in (
        ("optim", "lr", 9e-4),
        ("optim", "warmup_steps", 17),
        ("noise", "t_min", 0.2),
        ("data", "shuffle_seed", 999),
        ("data", "train_exclude_data", ["heldout.parquet"]),
        ("train", "num_steps", 5),
        ("train", "log_interval", 7),
    ):
        changed = copy.deepcopy(config)
        setattr(getattr(changed, section), field, value)
        mutations.append(_trajectory(changed)["sha256"])
    assert all(digest != baseline["sha256"] for digest in mutations)


def test_resume_contract_loader_yield_arithmetic_is_explicit_at_zero_and_nonzero_steps(tmp_path):
    config = _config(str(tmp_path / "train.parquet"), tmp_path / "run")
    trajectory = _trajectory(config)
    for step, expected_consumed, expected_yields in ((0, 0, 1), (3, 6, 7)):
        checkpoint = _contract_checkpoint(trajectory, step=step)
        state = _load_test_contract(checkpoint, config, trajectory)
        assert state["consumed_microbatches"] == expected_consumed
        assert checkpoint["resume_contract"]["loader_yields_per_rank"] == expected_yields


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda checkpoint: checkpoint["resume_contract"].update(world_size=2), "world_size mismatch"),
        (lambda checkpoint: checkpoint["resume_contract"].update(grad_accum_steps=3), "grad_accum_steps mismatch"),
        (
            lambda checkpoint: checkpoint["resume_contract"].update(checkpoint_boundary="microbatch"),
            "checkpoint boundary",
        ),
        (lambda checkpoint: checkpoint["resume_contract"].update(consumed_microbatches=1), "count disagree"),
        (lambda checkpoint: checkpoint["resume_contract"].update(loader_yields_per_rank=99), "loader-yield"),
        (lambda checkpoint: checkpoint["resume_contract"].update(consumed_microbatches=-1), "nonnegative"),
        (lambda checkpoint: checkpoint["resume_contract"].update(rank_states=[]), "one state for every rank"),
        (
            lambda checkpoint: checkpoint["resume_contract"]["rank_states"][0].update(rank=1),
            "not ordered by rank",
        ),
        (
            lambda checkpoint: checkpoint["resume_contract"]["rank_states"][0].update(rng_state={}),
            "no restorable RNG",
        ),
        (
            lambda checkpoint: checkpoint["resume_contract"]["rank_states"][0]["next_batch_six_tensor_digests"].update(
                version=99
            ),
            "unsupported next-batch digest",
        ),
    ],
)
def test_resume_contract_rejects_malformed_state(tmp_path, mutation, match):
    config = _config(str(tmp_path / "train.parquet"), tmp_path / "run")
    trajectory = _trajectory(config)
    checkpoint = _contract_checkpoint(trajectory)
    mutation(checkpoint)
    with pytest.raises(RuntimeError, match=match):
        _load_test_contract(checkpoint, config, trajectory)


def test_resume_contract_rejects_negative_checkpoint_step(tmp_path):
    config = _config(str(tmp_path / "train.parquet"), tmp_path / "run")
    trajectory = _trajectory(config)
    checkpoint = _contract_checkpoint(trajectory)
    checkpoint["step"] = -1
    with pytest.raises(RuntimeError, match="checkpoint step must be nonnegative"):
        _load_test_contract(checkpoint, config, trajectory)


def test_resume_contract_rejects_changed_trajectory(tmp_path):
    config = _config(str(tmp_path / "train.parquet"), tmp_path / "run")
    saved_trajectory = _trajectory(config)
    changed = copy.deepcopy(config)
    changed.optim.lr = 9e-4
    current_trajectory = _trajectory(changed)
    checkpoint = _contract_checkpoint(saved_trajectory)
    with pytest.raises(RuntimeError, match="resume trajectory mismatch"):
        _load_test_contract(checkpoint, changed, current_trajectory)


def test_replayed_next_batch_digest_mismatch_fails_closed():
    x_0 = torch.arange(8, dtype=torch.long).view(1, 8)
    batch = DLMDataBatch(
        x_0=x_0,
        loss_mask=torch.ones_like(x_0, dtype=torch.bool),
        attention_mask=torch.ones_like(x_0, dtype=torch.bool),
        epoch=0,
    )
    expected = _batch_six_tensor_digests(batch)
    changed = DLMDataBatch(
        x_0=x_0.clone(),
        loss_mask=batch.loss_mask,
        attention_mask=batch.attention_mask,
        epoch=0,
    )
    changed.x_0[0, 0] += 1
    with pytest.raises(RuntimeError, match="deterministic loader replay mismatch"):
        _verify_replayed_next_batch(
            {"expected_next_batch_six_tensor_digests": expected},
            changed,
            rank=0,
        )


def test_cpu_gloo_uninterrupted_and_resumed_training_are_exact(tmp_path):
    data_path = _write_rank_distinct_parquet(tmp_path / "train.parquet")
    continuous_dir = tmp_path / "continuous"
    continuous_config = _config(data_path, continuous_dir)
    continuous_config_path = tmp_path / "continuous.json"
    continuous_config_path.write_text(json.dumps(to_dict(continuous_config)), encoding="utf-8")

    continuous_result = _launch(continuous_config_path)
    assert continuous_result.returncode == 0, continuous_result.stdout + continuous_result.stderr

    intermediate_path = continuous_dir / "model_step2.pt"
    intermediate = torch.load(intermediate_path, map_location="cpu", weights_only=False)
    assert intermediate["resume_contract"]["version"] == 1
    assert intermediate["resume_contract"]["world_size"] == 2
    assert intermediate["resume_contract"]["grad_accum_steps"] == 2
    assert intermediate["resume_contract"]["consumed_microbatches"] == 4
    assert intermediate["resume_contract"]["replay_advance_count_per_rank"] == 4
    assert intermediate["resume_contract"]["loader_yields_per_rank"] == 5
    assert intermediate["resume_contract"]["global_processed_microbatches"] == 8
    assert intermediate["resume_contract"]["trajectory"]["sha256"]
    assert [state["rank"] for state in intermediate["resume_contract"]["rank_states"]] == [0, 1]
    assert not torch.equal(
        intermediate["resume_contract"]["rank_states"][0]["rng_state"]["torch_cpu"],
        intermediate["resume_contract"]["rank_states"][1]["rng_state"]["torch_cpu"],
    )

    resumed_dir = tmp_path / "resumed"
    resumed_config = _config(data_path, resumed_dir)
    resumed_config.train.resume_from = str(intermediate_path)
    resumed_config_path = tmp_path / "resumed.json"
    resumed_config_path.write_text(json.dumps(to_dict(resumed_config)), encoding="utf-8")

    resumed_result = _launch(resumed_config_path)
    assert resumed_result.returncode == 0, resumed_result.stdout + resumed_result.stderr

    continuous = torch.load(continuous_dir / "model_latest.pt", map_location="cpu", weights_only=False)
    resumed = torch.load(resumed_dir / "model_latest.pt", map_location="cpu", weights_only=False)
    _assert_nested_equal(continuous["model_state_dict"], resumed["model_state_dict"])
    _assert_nested_equal(continuous["optimizer_state_dict"], resumed["optimizer_state_dict"])
    _assert_nested_equal(continuous["scaler_state_dict"], resumed["scaler_state_dict"])
    _assert_nested_equal(
        continuous["resume_contract"]["rank_states"],
        resumed["resume_contract"]["rank_states"],
    )

    continuous_metrics = continuous["metrics"]
    resumed_metrics = resumed["metrics"]
    assert resumed_metrics["resume_contract_version"] == 1
    assert resumed_metrics["resume_contract_status"] == "exact_contract"
    assert resumed_metrics["resume_consumed_microbatches"] == 4
    assert resumed_metrics["resume_next_batch_digest_verified"] is True
    assert resumed_metrics["training_counter_reduction"] == "packed_int64_all_reduce"

    expected = _expected_data_counters(continuous_config, world_size=2, microbatches=8)
    assert continuous["train_tokens"] == resumed["train_tokens"] == expected["processed"] == 128
    assert continuous["train_valid_tokens"] == resumed["train_valid_tokens"] == expected["valid"]
    assert continuous_metrics["train_content_tokens"] == resumed_metrics["train_content_tokens"] == expected["content"]
    assert (
        continuous_metrics["train_content_bytes"] == resumed_metrics["train_content_bytes"] == expected["content_bytes"]
    )
    _assert_nested_equal(_lineage_metrics(continuous_metrics), _lineage_metrics(resumed_metrics))

    continuous_records = [
        json.loads(line) for line in (continuous_dir / "train_log.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    resumed_records = [
        json.loads(line) for line in (resumed_dir / "train_log.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["step"] for record in resumed_records] == [3, 4]
    for continuous_record, resumed_record in zip(continuous_records[2:], resumed_records, strict=True):
        assert resumed_record["raw_train_loss"] == continuous_record["raw_train_loss"]
        assert resumed_record["lr"] == continuous_record["lr"]
        assert resumed_record["train_tokens"] == continuous_record["train_tokens"]
        assert resumed_record["train_valid_tokens"] == continuous_record["train_valid_tokens"]
