"""Explicit opt-in architecture bundle and new parameter-role policies."""

import math

PROFILE = "nanochat_block_v1"
SCOPE = "single_rank_nanochat_backbone_d4"
ROLES = ("value_embedding", "value_gate", "residual_scale", "input_mix", "smear_gate", "smear_scale", "backout_scale")


def validate_experiment(config):
    m, p = config.model, config.parameterization
    if (
        p.mode != "ordinary"
        or p.execution_scope != "ordinary"
        or p.hidden_init_std_multiplier != 1
        or config.optim.type not in {"adamw", "nanochat_muon"}
        or m.rope_base != 100000
        or m.attention_scale != -1
        or config.noise.time_conditioning
        or m.block_causal_pilot_size
    ):
        raise ValueError("nanochat_block_v1 requires ordinary, unconditioned mechanics and RoPE100000/default scale")
    block = config.noise.type == "bd3lm" and m.bd3lm_clean_stream_mode == "causal" and not m.causal
    full_sequence = config.noise.type == "mdlm" and not m.bd3lm_block_size
    if not (full_sequence or (block and m.bd3lm_block_size)):
        raise ValueError("nanochat_block_v1 admits causal AR, full bidirectional MDLM or causal-clean block MDLM only")
    if (
        full_sequence
        and not m.causal
        and (
            config.train.token_edit_corruption_batch_fraction
            or config.train.token_edit_corruption_ratio
            or config.train.mdlm_auxiliary.mode != "none"
        )
    ):
        raise ValueError("nanochat_block_v1 full MDLM excludes token-edit/T2T mixtures and auxiliary views")
    if config.train.bd3lm_uniform_t2t_batch_fraction or config.train.bd3lm_recipe_mechanism != "none":
        raise ValueError("nanochat_block_v1 excludes T2T and recipe extensions")


def validate_runtime(config):
    if (
        config.parameterization_mode != "ordinary"
        or config.parameterization_mechanics_sha256
        or config.parameterization_residual_multiplier != 1
        or config.parameterization_output_multiplier != 1
        or config.time_conditioning
        or config.attention_scale != -1
        or config.n_embd < 24
        or config.n_embd % config.n_head
        or (config.n_embd // config.n_head) % 2
    ):
        raise ValueError("nanochat_block_v1 requires ordinary unconditioned mechanics and even head width")


def initialization(role, width):
    if role in {"token_embedding", "output_head"}:
        return {
            "rule": "normal",
            "resolved_scale": 0.8 if role == "token_embedding" else 0.001,
            "scale_multiplier": 1.0,
        }
    if role in {"attention_output", "mlp_output", "smear_scale"}:
        return {"rule": "zeros", "resolved_scale": 0.0, "scale_multiplier": 1.0}
    if role in {"smear_gate", "value_gate"}:
        return {"rule": "uniform_positive", "lower": 0.0, "upper": 0.02}
    if role in {"residual_scale", "input_mix"}:
        return {
            "rule": "linear_by_layer",
            "first": 1.15 if role == "residual_scale" else 0.20,
            "last": 1.05 if role == "residual_scale" else 0.05,
        }
    if role == "backout_scale":
        return {"rule": "constant", "resolved_scale": 0.2, "scale_multiplier": 1.0}
    scale = 0.4 if role == "mlp_up" else 1.0
    return {
        "rule": "uniform",
        "resolved_scale": scale / math.sqrt(width),
        "uniform_bound": scale * math.sqrt(3 / width),
        "scale_multiplier": scale,
    }


def auxiliary_optimizer(role, optim):
    """Pinned upstream grouping; existing tuned embedding/head roots stay explicit.

    Scalar root0.5 is scaled by sqrt(EB/524288)=sqrt(1/8) for this EB64K
    screen. Smear's0.2 root is deliberately fixed, as in pinned upstream.
    """
    scalar = 0.5 * math.sqrt(1 / 8)
    table = {
        "value_embedding": (optim.nanochat_embedding_lr * 0.5, 0.01, [0.8, 0.995]),
        "residual_scale": (scalar * 0.01, 0.05, [0.8, 0.95]),
        "input_mix": (scalar, 0.0, [0.96, 0.95]),
        "smear_gate": (0.2, 0.0, [0.8, 0.95]),
        "smear_scale": (0.2, 0.0, [0.8, 0.95]),
        "backout_scale": (0.2, 0.0, [0.8, 0.95]),
    }
    lr, wd, betas = table[role]
    return "adamw." + role, "adamw", lr, wd, 1e-10, {"betas": betas, "eps": 1e-10}
