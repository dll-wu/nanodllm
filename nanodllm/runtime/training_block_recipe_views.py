"""Stopped-gradient draft/teacher views, separate from the ordinary block loss."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from .block_recipe_canvases import block_groups, confidence_margin, confidence_masked_canvas, generated_canvas
from .model import DiffusionTransformer
from .noise import MDLMNoise
from .training_block_conversion import build_block_conversion_logit_plan
from .training_initialization import load_model_state_initialization
from .training_objective import _forward_training_model
from .training_rng import _restore_rng_state, _rng_state


def setup_block_recipe(session) -> None:
    session.block_recipe_teacher = None
    if session.config.train.bd3lm_recipe_mechanism == "none":
        return
    if session.world_size != 1:
        raise ValueError("block recipe screens currently require exactly one training rank")
    name = session.config.train.bd3lm_recipe_mechanism
    if name not in {"anchor", "ar_teacher"}:
        return
    state = _rng_state(session.config)
    try:
        cfg = replace(
            session.model_config,
            causal=name == "ar_teacher",
            bd3lm_block_size=0 if name == "ar_teacher" else session.model_config.bd3lm_block_size,
        )
        teacher = DiffusionTransformer(
            cfg,
            rope_base=session.config.model.rope_base,
            compute_dtype=session.compute_dtype,
            pad_vocab_to=session.config.model.pad_vocab_to,
        ).to(session.device)
        train = session.config.train
        session.block_recipe_teacher_source = load_model_state_initialization(
            teacher,
            checkpoint_path=train.bd3lm_recipe_teacher_path,
            expected_sha256=train.bd3lm_recipe_teacher_sha256,
            expected_size_bytes=train.bd3lm_recipe_teacher_size_bytes,
            rank=0,
            world_size=1,
        )
        teacher.requires_grad_(False).eval()
        session.block_recipe_teacher = teacher
    finally:
        _restore_rng_state(state, session.config)


def forward_view(session, model, canvas, clean, active):
    plan = build_block_conversion_logit_plan(
        clean,
        session.loss_mask,
        session.attention_mask,
        active,
        include_ar=False,
        position_ids=getattr(session.batch, "position_ids", None),
    )
    return _forward_training_model(
        model,
        session.config,
        canvas,
        clean,
        session.forward_attention_mask,
        logits_indices=plan.logits_indices,
        return_aux_logits=False,
        noise_level=None,
        position_ids=getattr(session.batch, "position_ids", None),
        bd3lm_block_size=session.block_conversion_training_block_size,
    )


@torch.no_grad()
def sample_draft(logits: torch.Tensor, mask_token_id: int) -> torch.Tensor:
    samples = []
    for chunk in logits.split(256):
        values = chunk.detach().float().clone()
        values[:, mask_token_id] = -torch.inf
        samples.append(torch.multinomial(values.softmax(-1), 1).squeeze(-1))
    return torch.cat(samples)


def recipe_corruption(session, corruption):
    phase = getattr(session, "block_recipe_phase", None)
    name = phase.mechanism if phase is not None else "none"
    session.block_recipe_draft_passes = 0
    session.block_recipe_teacher_passes = 0
    session.block_recipe_auxiliary_loss = 0.0
    if name not in {"decode_mask", "staged_generated"} or (name == "staged_generated" and corruption.branch != "t2t"):
        return corruption
    clean, loss_mask = session.x_0, session.loss_mask
    mask_id = session.model_config.mask_token_id
    if name == "decode_mask":
        canvas = clean.masked_fill(loss_mask, mask_id)
        active = loss_mask
    else:
        canvas, weights = MDLMNoise(mask_token_id=mask_id).corrupt(clean, corruption.block_times, maskable=loss_mask)
        active = (weights > 0) & loss_mask
    if not bool(active.any()):
        if name == "staged_generated":
            return generated_canvas(
                clean, loss_mask, canvas, canvas, mask_token_id=mask_id, block_times=corruption.block_times
            )
        return corruption
    with torch.no_grad(), session.autocast_ctx:
        logits = forward_view(session, session.model, canvas, clean, active)
        session.block_recipe_draft_passes += 1
        if name == "decode_mask":
            scores = torch.zeros_like(clean, dtype=torch.float32)
            scores[active] = confidence_margin(logits)
            groups = block_groups(
                clean, getattr(session.batch, "position_ids", None), session.block_conversion_training_block_size
            )
            return confidence_masked_canvas(clean, loss_mask, corruption, scores, groups, mask_id)
        draft = canvas.clone()
        draft[active] = sample_draft(logits, mask_id)
        return generated_canvas(
            clean, loss_mask, canvas, draft, mask_token_id=mask_id, block_times=corruption.block_times
        )


def distribution_kl(student: torch.Tensor, teacher: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if student.shape != teacher.shape or student.ndim != 2 or student.shape[0] == 0:
        raise ValueError("KL requires matching nonempty [positions,vocabulary] logits")
    terms = []
    for start in range(0, student.shape[0], 256):
        gold = teacher[start : start + 256].detach().float().log_softmax(-1)
        value = F.kl_div(
            student[start : start + 256].float().log_softmax(-1), gold, log_target=True, reduction="none"
        ).sum(-1)
        if weights is not None:
            value = value * weights[start : start + 256]
        terms.append(value.sum())
    return torch.stack(terms).sum()


def anchor_loss(session):
    if getattr(getattr(session, "block_recipe_phase", None), "mechanism", "none") != "anchor":
        return session.loss.new_zeros(())
    with torch.no_grad(), session.autocast_ctx:
        teacher = forward_view(session, session.block_recipe_teacher, session.z_t, session.x_0, session.active)
    session.block_recipe_teacher_passes += 1
    value = (
        distribution_kl(session.logits, teacher, session.block_conversion_corruption.decoder_weights)
        / session.lossable_tokens
    )
    value = value * session.config.train.bd3lm_recipe_teacher_weight
    session.block_recipe_auxiliary_loss = float(value.detach())
    return value


def backward_ar_teacher(session):
    """KD on one-pass student-draft states, explicitly not full-trajectory OPD."""
    if getattr(getattr(session, "block_recipe_phase", None), "mechanism", "none") != "ar_teacher":
        return
    # The main M2T graph has already been consumed. Reuse its detached predictions.
    draft = session.z_t.clone()
    with torch.no_grad():
        draft[session.active] = sample_draft(session.logits.detach(), session.model_config.mask_token_id)
    valid = session.loss_mask.clone()
    valid[:, 0] = False
    positions = getattr(session.batch, "position_ids", None)
    if positions is not None:
        valid[:, 1:] &= positions[:, 1:] == positions[:, :-1] + 1
    rows, cols = valid.nonzero(as_tuple=True)
    if rows.numel() == 0:
        return
    with torch.no_grad(), session.autocast_ctx:
        teacher = session.block_recipe_teacher(
            draft,
            logits_indices=rows * draft.shape[1] + cols - 1,
            attention_mask=session.forward_attention_mask,
            position_ids=positions,
        )
    with session.autocast_ctx:
        student = forward_view(session, session.model, draft, draft, valid)
        value = distribution_kl(student, teacher) / rows.numel() * session.config.train.bd3lm_recipe_teacher_weight
    session.block_recipe_teacher_passes += 1
    session.block_recipe_auxiliary_loss = float(value.detach())
    if not math.isfinite(session.block_recipe_auxiliary_loss):
        raise RuntimeError("nonfinite AR-teacher auxiliary loss")
    session.step_loss_sum += session.block_recipe_auxiliary_loss
    session.scaler.scale(value / session.grad_accum_steps).backward()


def record_recipe_microbatch(session, step: int, micro: int) -> None:
    if session.config.train.bd3lm_recipe_mechanism == "none":
        return
    from .telemetry import append_jsonl

    phase = session.block_recipe_phase
    append_jsonl(
        Path(session.config.train.run_dir) / "block_recipe_microbatches.jsonl",
        {
            "step_index": step,
            "microbatch_index": micro,
            "mechanism": phase.mechanism,
            "block_size": phase.block_size,
            "branch": session.block_conversion_corruption.branch,
            "auxiliary_loss": session.block_recipe_auxiliary_loss,
            "draft_base_forward_passes": session.block_recipe_draft_passes,
            "teacher_base_forward_passes": session.block_recipe_teacher_passes,
            "extra_student_backward_passes": int(
                phase.mechanism == "ar_teacher" and session.block_recipe_teacher_passes > 0
            ),
            "self_conditioning_draft_forward_passes": int(
                phase.mechanism == "self_condition" and bool(session.z_t.eq(session.model_config.mask_token_id).any())
            ),
        },
    )
