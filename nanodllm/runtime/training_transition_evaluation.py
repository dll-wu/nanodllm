from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

import torch.distributed as dist
import torch.nn as nn

from .config import ExperimentConfig
from .data import build_frequency_token_classes, build_val_loader
from .eval_token_edit_transition import (
    TOKEN_EDIT_TRANSITION_RECORD_SCHEMA,
    TOKEN_EDIT_TRANSITION_RECORD_SCHEMA_VERSION,
    evaluate_token_edit_transitions,
)
from .training_artifacts import write_jsonl_atomic
from .training_rng import _restore_rng_state, _rng_state
from .training_runtime import _batch_model_attention_mask


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _jsonl_sha256(records: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_conflicting_artifact(path: str, expected_sha256: str) -> None:
    if os.path.exists(path) and _file_sha256(path) != expected_sha256:
        raise FileExistsError(f"refusing to overwrite conflicting transition artifact: {path}")


def _write_text_atomic(path: str, text: str) -> str:
    payload = text.encode("utf-8")
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    _reject_conflicting_artifact(path, payload_sha256)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = path + ".tmp"
    with open(temporary_path, "wb") as handle:
        handle.write(payload)
    os.replace(temporary_path, path)
    return payload_sha256


def run_token_edit_transition_evaluation(
    raw_model: nn.Module,
    config: ExperimentConfig,
    tokenizer: Any,
    autocast_ctx: Any,
    device: Any,
    step: int,
    model_logits_fn: Any,
) -> dict[str, object]:
    """Run and persist the bounded paired draft/correction panel."""

    if not config.train.eval_only or config.eval.source_checkpoint_identity_mode != "sha256":
        raise RuntimeError("token-edit transition evaluation requires eval-only replay with verified SHA-256 identity")
    if dist.is_initialized() and dist.get_world_size() != 1:
        raise RuntimeError("token-edit transition eval-only replay currently requires a single process")

    checkpoint_sha256 = str(config.eval.source_checkpoint_expected_sha256)
    checkpoint_size_bytes = int(config.eval.source_checkpoint_expected_size_bytes)
    checkpoint_binding = "verified_expected_sha256"
    panel_contract = {
        "schema": "nanodllm.token_edit_transition_panel_contract.v1",
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_size_bytes": checkpoint_size_bytes,
        "evaluation_step": int(step),
        "validation_batches": int(config.eval.token_edit_transition_batches),
        "frequency_training_batches": int(config.eval.token_edit_transition_frequency_batches),
        "mask_fractions": [float(value) for value in config.eval.token_edit_transition_mask_fractions],
        "seed": int(config.eval.token_edit_transition_seed),
        "log_samples_per_fraction": int(config.eval.token_edit_transition_log_samples),
    }
    panel_contract_sha256 = _canonical_json_sha256(panel_contract)
    record_context = {
        "evaluation_step": int(step),
        "checkpoint_binding": checkpoint_binding,
        "source_checkpoint_sha256": checkpoint_sha256,
        "source_checkpoint_size_bytes": checkpoint_size_bytes,
        "panel_contract": panel_contract,
        "panel_contract_sha256": panel_contract_sha256,
    }

    frequency_start = time.perf_counter()
    saved_state = _rng_state(config)
    try:
        token_frequency_class_ids, frequency_metadata = build_frequency_token_classes(
            config.data,
            tokenizer,
            int(config.train.device_batch_size),
            int(config.model.max_seq_len),
            device,
            int(config.eval.token_edit_transition_frequency_batches),
            num_deciles=7,
        )
        frequency_metadata = {key: value for key, value in frequency_metadata.items() if key != "output_factorization"}
        frequency_metadata.update(
            {
                "role": "token_edit_transition_frequency_stratification",
                "builder": "build_frequency_token_classes",
                "model_output_factorization": str(config.model.output_factorization),
            }
        )
    finally:
        _restore_rng_state(saved_state, config)
    frequency_seconds = time.perf_counter() - frequency_start

    val_loader = build_val_loader(config, tokenizer, device)
    clean_batches = [next(val_loader) for _ in range(int(config.eval.token_edit_transition_batches))]
    sample_lines: list[str] = []
    panel_start = time.perf_counter()
    with autocast_ctx:
        metrics, records = evaluate_token_edit_transitions(
            raw_model,
            tokenizer,
            clean_batches,
            mask_fractions=config.eval.token_edit_transition_mask_fractions,
            seed=int(config.eval.token_edit_transition_seed),
            device=device,
            model_logits_fn=model_logits_fn,
            attention_mask_fn=_batch_model_attention_mask,
            token_frequency_class_ids=token_frequency_class_ids,
            record_context=record_context,
            log_fn=sample_lines.append,
            log_samples=int(config.eval.token_edit_transition_log_samples),
        )
    panel_wall_seconds = time.perf_counter() - panel_start

    artifact_start = time.perf_counter()
    expected_artifact_sha256 = _jsonl_sha256(records)
    artifact_relpath = (
        f"telemetry/token_edit_transitions_checkpoint_{checkpoint_sha256}_content_{expected_artifact_sha256}_"
        f"step{int(step)}.v{TOKEN_EDIT_TRANSITION_RECORD_SCHEMA_VERSION}.jsonl"
    )
    artifact_path = os.path.join(config.train.run_dir, artifact_relpath)
    _reject_conflicting_artifact(artifact_path, expected_artifact_sha256)
    artifact_sha256 = write_jsonl_atomic(artifact_path, records)
    if artifact_sha256 != expected_artifact_sha256:
        raise RuntimeError("transition JSONL serialization changed between identity and atomic write")

    sample_text = (
        f"[token-edit-transition] step={int(step)} "
        f"batches={int(config.eval.token_edit_transition_batches)} "
        f"seed={int(config.eval.token_edit_transition_seed)} "
        f"checkpoint_binding={checkpoint_binding} "
        f"source_checkpoint_sha256={checkpoint_sha256} "
        f"panel_contract_sha256={panel_contract_sha256} "
        "mask_fractions="
        + ",".join(f"{float(value):.6g}" for value in config.eval.token_edit_transition_mask_fractions)
        + "\n"
        + "".join(line + "\n" for line in sample_lines)
    )
    expected_sample_log_sha256 = hashlib.sha256(sample_text.encode("utf-8")).hexdigest()
    sample_log_relpath = (
        f"telemetry/token_edit_transition_samples_checkpoint_{checkpoint_sha256}_"
        f"content_{expected_sample_log_sha256}_step{int(step)}."
        f"v{TOKEN_EDIT_TRANSITION_RECORD_SCHEMA_VERSION}.log"
    )
    sample_log_path = os.path.join(config.train.run_dir, sample_log_relpath)
    sample_log_sha256 = _write_text_atomic(sample_log_path, sample_text)
    artifact_write_seconds = time.perf_counter() - artifact_start
    sequence_exposures = int(metrics["token_edit_transition_sequence_count"]) * int(
        metrics["token_edit_transition_mask_fraction_count"]
    )
    nfe_sequence_exposures = sequence_exposures * int(metrics["token_edit_transition_total_nfe"])
    model_forward_seconds = float(metrics["token_edit_transition_model_forward_seconds"])
    metrics.update(
        {
            "token_edit_transition_evaluation_step": int(step),
            "token_edit_transition_records_artifact": artifact_relpath,
            "token_edit_transition_records_schema": TOKEN_EDIT_TRANSITION_RECORD_SCHEMA,
            "token_edit_transition_records_count": len(records),
            "token_edit_transition_records_sha256": artifact_sha256,
            "token_edit_transition_samples_artifact": sample_log_relpath,
            "token_edit_transition_samples_sha256": sample_log_sha256,
            "token_edit_transition_checkpoint_binding": checkpoint_binding,
            "token_edit_transition_source_checkpoint_sha256": checkpoint_sha256,
            "token_edit_transition_source_checkpoint_size_bytes": checkpoint_size_bytes,
            "token_edit_transition_panel_contract": panel_contract,
            "token_edit_transition_panel_contract_sha256": panel_contract_sha256,
            "token_edit_transition_frequency_metadata": frequency_metadata,
            "token_edit_transition_frequency_map_seconds": frequency_seconds,
            "token_edit_transition_panel_wall_seconds": panel_wall_seconds,
            "token_edit_transition_artifact_write_seconds": artifact_write_seconds,
            "token_edit_transition_sequence_exposures": sequence_exposures,
            "token_edit_transition_nfe_sequence_exposures": nfe_sequence_exposures,
            "token_edit_transition_panel_wall_seconds_per_sequence_exposure": panel_wall_seconds
            / max(1, sequence_exposures),
            "token_edit_transition_model_forward_seconds_per_nfe_sequence_exposure": model_forward_seconds
            / max(1, nfe_sequence_exposures),
            "token_edit_transition_latency_scope": (
                "synchronized_model_forward_only_excluding_panel_bookkeeping_frequency_scan_and_artifact_io"
            ),
            "token_edit_transition_panel_wall_scope": (
                "complete_evaluator_including_hashing_strata_and_sample_decoding_excluding_frequency_scan_and_artifact_io"
            ),
        }
    )
    return metrics
