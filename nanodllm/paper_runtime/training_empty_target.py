"""DDP graph participation for microbatches with no local loss targets."""

from __future__ import annotations

import torch

from .training_features import _uses_length_head, _uses_uniform_low_variance_estimator
from .training_objective import _forward_training_model, _target_lengths_for_conditioning
from .training_state import TrainingSession


def _graph_connected_zero(tensors: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Return an exact zero whose autograd graph reaches every input tensor."""

    if not tensors:
        raise ValueError("graph-connected zero requires at least one tensor")
    zero = tensors[0].reshape(-1)[:0].sum()
    for tensor in tensors[1:]:
        zero = zero + tensor.reshape(-1)[:0].sum()
    return zero


def _empty_target_model_output_zero(
    model_out,
    *,
    use_hierarchical_output: bool,
    return_aux_logits: bool,
    has_length_output: bool,
    include_length_loss: bool,
) -> torch.Tensor:
    """Mirror the configured loss heads while assigning them zero gradient mass."""

    length_logits = None
    if use_hierarchical_output:
        if return_aux_logits and has_length_output:
            class_logits, token_logits, aux_logits, length_logits = model_out
        elif return_aux_logits:
            class_logits, token_logits, aux_logits = model_out
        elif has_length_output:
            class_logits, token_logits, length_logits = model_out
            aux_logits = []
        else:
            class_logits, token_logits = model_out
            aux_logits = []
        loss_tensors = [class_logits, token_logits, *aux_logits]
    else:
        if return_aux_logits and has_length_output:
            logits, aux_logits, length_logits = model_out
        elif return_aux_logits:
            logits, aux_logits = model_out
        elif has_length_output:
            logits, length_logits = model_out
            aux_logits = []
        else:
            logits, aux_logits = model_out, []
        loss_tensors = [logits, *aux_logits]
    if include_length_loss:
        if length_logits is None:
            raise RuntimeError("length-head loss is enabled but the empty-target forward returned no length logits")
        loss_tensors.append(length_logits)
    return _graph_connected_zero(tuple(loss_tensors))


def backward_empty_ddp_microbatch(session: TrainingSession) -> None:
    """Keep an empty rank in the same DDP forward/backward sequence as its peers."""

    if not session.is_ddp:
        raise RuntimeError("empty-target DDP backward requires an initialized distributed session")
    with session.sync_context():
        with session.autocast_ctx:
            if _uses_uniform_low_variance_estimator(session.config):
                if isinstance(session.t_val, torch.Tensor) or session.t_val is None:
                    raise ValueError("two-sample uniform estimator requires scalar uniform t")
                if session.use_hierarchical_output or session.return_aux_logits:
                    raise ValueError("two-sample uniform estimator requires flat logits without auxiliary heads")
                session.z_second, _ = session.noise.corrupt(
                    session.x_0,
                    float(session.t_val),
                    maskable=session.loss_mask,
                )
                zero_terms = []
                for z_sample in (session.z_t, session.z_second):
                    model_out = _forward_training_model(
                        session.model,
                        session.config,
                        z_sample,
                        session.x_0,
                        session.forward_attention_mask,
                        logits_indices=session.target_indices,
                        return_aux_logits=False,
                        noise_level=session.noise_level,
                        return_hierarchical_logits=False,
                        input_embeddings=None,
                    )
                    zero_terms.append(_graph_connected_zero((model_out,)))
                zero_loss = sum(zero_terms[1:], zero_terms[0])
            else:
                target_lengths = _target_lengths_for_conditioning(
                    session.config,
                    session.x_0,
                    session.attention_mask,
                    session.eos_token_id,
                )
                model_out = _forward_training_model(
                    session.model,
                    session.config,
                    session.z_t,
                    session.x_0,
                    session.forward_attention_mask,
                    logits_indices=session.logits_indices,
                    return_aux_logits=session.return_aux_logits,
                    noise_level=session.noise_level,
                    return_hierarchical_logits=session.use_hierarchical_output,
                    return_length_logits=_uses_length_head(session.config),
                    input_embeddings=session.soft_input_embeddings,
                    position_ids=getattr(session.batch, "position_ids", None),
                    target_lengths=target_lengths,
                    bd3lm_block_size=getattr(session, "block_conversion_training_block_size", None),
                )
                zero_loss = _empty_target_model_output_zero(
                    model_out,
                    use_hierarchical_output=session.use_hierarchical_output,
                    return_aux_logits=session.return_aux_logits,
                    has_length_output=_uses_length_head(session.config),
                    include_length_loss=session.config.train.length_head_loss_weight > 0,
                )
        session.scaler.scale(zero_loss / session.grad_accum_steps).backward()
