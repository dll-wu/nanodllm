"""Isolated D8 residual-depth ablation, not a full CompleteP/Muon recipe."""

import hashlib
import json

SCOPE = "d8_muon_residual_depth"
RULE = "nanodllm.muon_residual_depth.d8.v1"


def active(config):
    return config.parameterization.execution_scope == SCOPE


def validate_config(config, *, optimizer=True):
    p, m, o = config.parameterization, config.model, config.optim
    if (
        p.mode != "ordinary"
        or p.contract_binding != "required"
        or (p.base_depth, p.base_width) != (4, 256)
        or p.hidden_init_std_multiplier != 1
        or m.backbone_profile != "nanollada_v1"
        or (m.depth, m.n_embd, m.n_head, m.mlp_hidden_dim) != (8, 512, 8, 2048)
        or m.attention_scale != -1
        or config.noise.type != "mdlm"
        or config.noise.time_conditioning
        or m.bd3lm_block_size
        or m.block_causal_pilot_size
        or (optimizer and o.type != "nanochat_muon")
        or o.nanochat_matrix_optimizer != "muon"
        or o.nanochat_execution_scope != "ddp_replicated_d4_d8"
    ):
        raise ValueError("D8 Muon residual-depth scope requires its plain bound D8 Muon configuration")


def mechanics():
    material = {
        "schema": "nanodllm.parameterization_mechanics.v1",
        "mode": "ordinary",
        "rule_set": RULE,
        "identity": False,
        "base_coordinates": {"depth": 4, "width": 256},
        "target_coordinates": {"depth": 8, "width": 512},
        "initialization": "unchanged nanollada_v1: embedding std1, head std.001, uniform hidden, zero residual outputs",
        "forward": {
            "attention_residual_multiplier": 0.5,
            "mlp_residual_multiplier": 0.5,
            "unembedding_multiplier": 1.0,
            "attention_scale": "unchanged 1/sqrt(head_dim)",
        },
        "optimizer_multipliers": "unchanged Muon shape factor and explicit role roots",
        "claim_boundary": "residual-only 1/(depth/base_depth) ablation, not CompleteP",
    }
    digest = hashlib.sha256(json.dumps(material, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {**material, "mechanics_sha256": digest}


def model_kwargs():
    return {
        "parameterization_rule_set": RULE,
        "parameterization_base_depth": 4,
        "parameterization_base_width": 256,
        "parameterization_mechanics_sha256": mechanics()["mechanics_sha256"],
        "parameterization_residual_multiplier": 0.5,
    }


def validate_runtime(config):
    expected = {
        **model_kwargs(),
        "parameterization_mode": "ordinary",
        "backbone_profile": "nanollada_v1",
        "n_layer": 8,
        "n_embd": 512,
        "n_head": 8,
        "mlp_hidden_dim": 2048,
        "parameterization_output_multiplier": 1.0,
        "parameterization_hidden_init_std": -1.0,
        "parameterization_embedding_head_init_std": -1.0,
        "parameterization_attention_scale": -1.0,
        "parameterization_attention_scale_rule": "",
        "attention_scale": -1.0,
        "time_conditioning": False,
        "bd3lm_block_size": 0,
    }
    if any(getattr(config, key) != value for key, value in expected.items()):
        raise ValueError("D8 Muon residual-depth runtime mechanics differ")


def finish_contract(contract):
    contract.update(
        rule_set=RULE,
        execution_scope=SCOPE,
        adaptation_status="implemented_residual_only_screen_not_completep",
        mechanics=mechanics(),
    )
    contract["optimizer"]["width_depth_transfer"] = "empirical_d8_residual_only_screen_not_completep"
    for records in (contract["parameter_records"], contract["parameter_groups"]):
        for record in records:
            if record["role"] in {"attention_output", "mlp_output"}:
                record["forward_multiplier"] = 0.5


def validate_model(model):
    validate_runtime(model.config)
    for block in model.blocks:
        if block.residual_multiplier != 0.5 or block.attn.attention_scale is not None:
            raise ValueError("instantiated residual multiplier or default attention scale differs")
