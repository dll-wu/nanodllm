"""One optional auxiliary microbatch, sharing the native optimizer and resume clock."""

import math
import os

import torch

from .mdlm_auxiliary_config import selected
from .mdlm_auxiliary_views import correction_loss, fill_mask, generator, legal_tokens, make_view
from .telemetry import append_jsonl
from .training_objective import _corrupt_training_batch, _forward_training_model, _update_eos_supervision_batch_counts
from .training_runtime import _batch_model_attention_mask
from .training_state import _content_byte_counts, _record_ar_boundary_batch


def run_mdlm_auxiliary_microbatch(session, *, step: int, micro: int) -> bool:
    aux = session.config.train.mdlm_auxiliary
    if not selected(aux, step, micro, session.grad_accum_steps):
        return False
    if session.is_ddp:
        raise ValueError("the first MDLM auxiliary experiment is single-process only")
    if aux.mode == "proposal":
        from .training_monotonic_proposals import run_proposal_microbatch

        run_proposal_microbatch(session, step=step, micro=micro)
        return True
    index = step * session.grad_accum_steps + micro
    clean = session.x_0
    # Advance precisely the ordinary corruption stream for this slot, keeping
    # the remaining ordinary microbatches paired with the unchanged reference.
    _corrupt_training_batch(
        session.config, session.noise, clean, session.loss_mask, position_ids=session.batch.position_ids
    )
    view = make_view(
        clean,
        session.batch.position_ids,
        aux,
        microbatch=index,
        bos_id=session.bos_token_id,
        eos_id=session.eos_token_id,
        mask_id=session.model_config.mask_token_id,
        special_ids=tuple(session.special_token_ids),
        attention_mask=session.attention_mask,
        loss_mask=session.loss_mask,
    )
    attention = _batch_model_attention_mask(session.batch, view.visible)
    active = view.targets.clone()
    canvas = view.canvas.clone()
    times = torch.ones((clean.shape[0], 1), device=clean.device)
    if aux.mode == "boundary":
        rng = generator(clean.device, aux.seed, index, "boundary-corruption")
        times = torch.rand((clean.shape[0], 1), generator=rng, device=clean.device)
        times = session.config.noise.t_min + times * (session.config.noise.t_max - session.config.noise.t_min)
        active &= torch.rand(clean.shape, generator=rng, device=clean.device) < times
        canvas.masked_fill_(active, session.model_config.mask_token_id)
    target_indices = active.flatten().nonzero(as_tuple=True)[0]
    target_count, denominator = int(target_indices.numel()), int(view.targets.sum().item())
    branch = "t2t" if aux.mode in {"uniform", "generated"} else "m2t"
    record = {
        "schema": "nanodllm.mdlm_auxiliary_microbatch.v1",
        "step": step + 1,
        "microbatch": index,
        "mode": aux.mode,
        "source_positions": clean.numel(),
        "visible_positions": int(view.visible.sum().item()),
        "lossable_targets": denominator,
        "active_targets": target_count,
        "excluded_segments": view.excluded_segments,
        "segments": view.segments,
        "draft_forwards": 0,
        "filled_targets": 0,
        "filled_wrong": 0,
        "filled_correct": 0,
        "empty": target_count == 0,
    }
    if aux.mode == "prefix_frontier":
        record.update(
            objective="full_mask_suffix_target_mean_ce",
            loss_denominator=denominator,
            gold_suffix_tokens=0,
            eos_targets=int((active & clean.eq(session.eos_token_id)).sum().item()),
        )
    _record_ar_boundary_batch(
        session,
        clean,
        view.targets,
        view.visible,
        session.batch.position_ids,
        session.batch.document_boundary_mask,
        session.batch.document_start_mask,
    )
    content_tokens, content_bytes = _content_byte_counts(session, clean, view.visible)
    session.step_valid_tokens += record["visible_positions"]
    session.step_content_tokens += content_tokens
    session.step_content_bytes += content_bytes
    session.step_lossable_tokens += denominator
    session.step_active_tokens += target_count
    name = f"step_{branch}_lossable_tokens"
    setattr(session, name, getattr(session, name) + denominator)
    if branch == "t2t":
        session.step_t2t_eligible_tokens += denominator
    if session.eos_supervision_audit_enabled:
        _update_eos_supervision_batch_counts(
            session.progress.lineage.eos_supervision_train_counts,
            clean,
            view.targets,
            view.visible,
            session.special_token_ids,
            session.eos_token_id,
            active,
        )
    if target_count:

        def forward(inputs):
            return _forward_training_model(
                session.model,
                session.config,
                inputs,
                clean,
                attention,
                logits_indices=target_indices,
                return_aux_logits=False,
                noise_level=None,
                position_ids=session.batch.position_ids,
            )

        fill = fill_mask(clean.shape, active, aux, index)
        if branch == "t2t":
            forbidden = tuple(i for i in session.special_token_ids if i != session.eos_token_id)
            support = legal_tokens(session.model_config.vocab_size, forbidden, clean.device)
            rng = generator(clean.device, aux.seed, index, "draft-tokens")
            if aux.mode == "uniform":
                draws = support[torch.randint(support.numel(), (target_count,), generator=rng, device=clean.device)]
            else:
                with torch.no_grad(), session.autocast_ctx:
                    draft_logits = forward(view.canvas)
                # Chunk the FP32 softmax; do not retain the draft graph or a
                # second full-vocabulary activation for the correction pass.
                draws = []
                for chunk in draft_logits.split(256):
                    probabilities = chunk.detach().float().index_select(1, support).softmax(-1)
                    draws.append(support[torch.multinomial(probabilities, 1, generator=rng).squeeze(1)])
                draws = torch.cat(draws)
                del draft_logits, probabilities
                record["draft_forwards"] = 1
            canvas[fill] = draws[fill[active]]
            record.update(
                filled_targets=int(fill.sum().item()),
                filled_wrong=int((fill & canvas.ne(clean)).sum().item()),
                filled_correct=int((fill & canvas.eq(clean)).sum().item()),
            )
        with session.autocast_ctx:
            logits = forward(canvas.detach())
            weights = (
                (1 / times.expand_as(clean)[active])
                if aux.mode == "boundary"
                else torch.ones(target_count, device=clean.device)
            )
            loss, raw_ce = correction_loss(logits, clean[active], weights, denominator)
        loss_value = float(loss.detach().item())
        if branch == "t2t":
            prediction = logits.detach().argmax(-1)
            initial_correct = canvas[active].eq(clean[active])
            final_correct = prediction.eq(clean[active])
            visible_drafts = fill[active]
            record.update(
                corrected_wrong_visible=int((visible_drafts & ~initial_correct & final_correct).sum()),
                damaged_correct_visible=int((visible_drafts & initial_correct & ~final_correct).sum()),
                final_correct_targets=int(final_correct.sum()),
            )
        if not math.isfinite(loss_value):
            raise RuntimeError(f"nonfinite MDLM auxiliary loss at step {step}, micro {micro}")
        session.scaler.scale(loss / session.grad_accum_steps).backward()
        session.step_loss_sum += loss_value
        session.step_main_loss_sum += loss_value
        session.step_main_ce_sum += float(raw_ce.detach().item())
        session.step_loss_normalization_scale_sum += target_count / denominator
        session.step_loss_count += 1
        for suffix, amount in (
            ("ce_sum", float(raw_ce.detach().item())),
            ("loss_sum", loss_value),
            ("loss_count", 1),
            ("microbatches", 1),
            ("active_tokens", target_count),
        ):
            name = f"step_{branch}_{suffix}"
            setattr(session, name, getattr(session, name) + amount)
        record.update(loss=loss_value, target_mean_ce=float(raw_ce.detach().item()))
        if aux.mode == "boundary":
            session.step_t_sum += float(times.sum().item())
            session.step_t_count += times.numel()
            session.step_t_min = min(session.step_t_min, float(times.min().item()))
            session.step_t_max = max(session.step_t_max, float(times.max().item()))
            session.step_schedule_multiplier_sum += float((1 / times).sum().item())
    append_jsonl(os.path.join(session.config.train.run_dir, "mdlm_auxiliary_microbatches.jsonl"), record)
    return True
