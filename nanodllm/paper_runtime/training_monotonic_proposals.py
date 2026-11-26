"""Native auxiliary proposal training with a frozen causal teacher and exact resume."""

import os

import torch

from .core_eval_runtime import load_core_model
from .mdlm_auxiliary_views import legal_tokens
from .monotonic_proposals import (
    advance_prefixes,
    first_error_loss,
    legal_argmax,
    proposal_logits,
    soft_teacher_loss,
    training_prefixes,
)
from .telemetry import append_jsonl
from .training_objective import _corrupt_training_batch
from .training_rng import _restore_rng_state, _rng_state
from .training_state import _content_byte_counts, _record_ar_boundary_batch


def teacher_for_session(session):
    if getattr(session, "proposal_teacher", None) is None:
        aux = session.config.train.mdlm_auxiliary
        rng = _rng_state(session.config)
        try:
            loaded = load_core_model(
                aux.proposal_teacher_path,
                expected_checkpoint_sha256=aux.proposal_teacher_sha256,
                expected_tokenizer_inventory_sha256=aux.proposal_tokenizer_sha256,
                scorer="ar",
                device=session.device.type,
                dtype=session.config.runtime.dtype,
            )
            teacher = loaded.model.requires_grad_(False).eval()
            if {k: v.shape for k, v in teacher.state_dict().items()} != {
                k: v.shape for k, v in session.model.state_dict().items()
            }:
                raise ValueError("proposal teacher and student backbones differ")
            session.proposal_teacher = teacher
        finally:
            _restore_rng_state(rng, session.config)
    return session.proposal_teacher


def run_proposal_microbatch(session, *, step, micro):
    aux, clean = session.config.train.mdlm_auxiliary, session.x_0
    index = step * session.grad_accum_steps + micro
    _corrupt_training_batch(
        session.config, session.noise, clean, session.loss_mask, position_ids=session.batch.position_ids
    )
    teacher = teacher_for_session(session)
    prefixes = training_prefixes(
        clean,
        session.batch.position_ids,
        session.attention_mask,
        seed=aux.seed,
        index=index,
        maximum=aux.proposal_max_prefix,
        eos_id=session.eos_token_id,
    )
    support = legal_tokens(
        session.model_config.vocab_size,
        tuple(t for t in session.special_token_ids if t != session.eos_token_id),
        clean.device,
    )
    # Index by selected-slot ordinal, not the even/odd global index of a 50% mixture.
    count = (2, 4, 8)[(index // 2) % 3]
    record = {
        "schema": "nanodllm.proposal_microbatch.v1",
        "step": step + 1,
        "microbatch": index,
        "commit_tokens": count,
        "source_positions": clean.numel(),
        "teacher_forwards": 0,
        "student_forwards": 0,
        "targets": 0,
        "first_errors": 0,
        "teacher_agreements": 0,
        "teacher_eos_targets": 0,
        "loss": 0.0,
        "rounds": [],
        "first_error_weight": aux.proposal_first_error_weight,
    }
    if aux.proposal_teacher_targets == "soft":
        record["teacher_targets"] = "soft"
    empty = torch.zeros_like(session.loss_mask)
    _record_ar_boundary_batch(
        session,
        clean,
        empty,
        session.attention_mask,
        session.batch.position_ids,
        session.batch.document_boundary_mask,
        session.batch.document_start_mask,
    )
    content, byte_count = _content_byte_counts(session, clean, session.attention_mask)
    session.step_valid_tokens += int(session.attention_mask.sum())
    session.step_content_tokens += content
    session.step_content_bytes += byte_count
    for round_index in range(aux.proposal_rounds):
        if not prefixes:
            break
        kwargs = {
            "count": count,
            "tail": aux.proposal_canvas_tokens - round_index * count,
            "mask_id": session.model_config.mask_token_id,
            "device": clean.device,
        }
        with session.autocast_ctx:
            logits, student_work = proposal_logits(session.model, prefixes, **kwargs)
            proposals = legal_argmax(logits, support)
            with torch.no_grad():
                teacher_logits, teacher_work = proposal_logits(teacher, prefixes, proposals=proposals, **kwargs)
                targets = legal_argmax(teacher_logits, support)
            if aux.proposal_teacher_targets == "soft":
                loss, stats = soft_teacher_loss(
                    logits, teacher_logits, proposals, eos_id=session.eos_token_id, support=support
                )
            else:
                loss, stats = first_error_loss(
                    logits,
                    targets,
                    proposals,
                    eos_id=session.eos_token_id,
                    first_error_weight=aux.proposal_first_error_weight,
                )
            del teacher_logits
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite proposal distillation loss")
        # A missing round after EOS contributes zero; never reward an early stop by renormalizing.
        value = float(loss.detach()) / aux.proposal_rounds
        session.scaler.scale(loss / (session.grad_accum_steps * aux.proposal_rounds)).backward()
        record["loss"] += value
        for key in ("targets", "first_errors", "teacher_agreements", "teacher_eos_targets"):
            record[key] += stats[key]
        record["teacher_forwards"] += 1
        record["student_forwards"] += 1
        record["rounds"].append({**stats, "student_work": student_work, "teacher_work": teacher_work})
        prefixes = advance_prefixes(prefixes, proposals, session.eos_token_id)
    for name in ("step_lossable_tokens", "step_active_tokens", "step_m2t_lossable_tokens", "step_m2t_active_tokens"):
        setattr(session, name, getattr(session, name) + record["targets"])
    for name in ("step_loss_sum", "step_main_loss_sum", "step_m2t_loss_sum"):
        setattr(session, name, getattr(session, name) + record["loss"])
    ce = sum(r["target_mean_ce"] for r in record["rounds"]) / aux.proposal_rounds
    session.step_main_ce_sum += ce
    session.step_m2t_ce_sum += ce
    for name in ("step_loss_count", "step_m2t_loss_count", "step_m2t_microbatches"):
        setattr(session, name, getattr(session, name) + 1)
    session.step_loss_normalization_scale_sum += 1
    append_jsonl(os.path.join(session.config.train.run_dir, "proposal_microbatches.jsonl"), record)
