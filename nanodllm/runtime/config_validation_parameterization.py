"""Scaling-contract admission and frozen mechanics projection validation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from typing import TYPE_CHECKING

from . import muon_residual_depth, nanollada_completep
from .completep_hidden_init_transfer import (
    COMPLETEP_HIDDEN_INIT_TRANSFER_EXECUTION_SCOPE,
    is_completep_hidden_init_transfer,
    validate_completep_hidden_init_transfer_config,
)
from .config_validation_optim import strip_disabled_nanochat_muon
from .d4_hidden_init_confirmation import (
    D4_HIDDEN_INIT_CONFIRMATION_EXECUTION_SCOPE,
    is_d4_hidden_init_confirmation,
    validate_d4_hidden_init_confirmation_config,
)
from .d8_batch_objective_config import SCOPE as D8_BATCH_OBJECTIVE_SCOPE
from .d8_batch_objective_config import validate_config as validate_d8_batch_objective_config
from .d20_parameterization_transfer import (
    D20_COMPLETEP_TPP20_EXECUTION_SCOPE,
    is_d20_completep_tpp20,
    validate_d20_completep_tpp20_config,
)
from .data_bos_packing_config import strip_disabled_bos_packing
from .data_mask_tail_config import strip_disabled_mask_tail
from .data_snapshot import strip_disabled_snapshot
from .lr_schedule import strip_legacy_lr_schedule
from .mdlm_auxiliary_config import strip_disabled_auxiliary
from .model_mtp import strip_disabled_prediction_extensions
from .parameterization_calibration import (
    D4_ROOT_CALIBRATION_EXECUTION_SCOPE,
    is_d4_root_calibration,
    validate_d4_root_calibration_config,
)
from .parameterization_transfer import (
    D8_CENTER_TRANSFER_EXECUTION_SCOPE,
    D8_RESPONSE_NEIGHBORHOOD_EXECUTION_SCOPE,
    D12_TRANSFER_EXECUTION_SCOPE,
    is_d8_center_transfer,
    is_d8_response_neighborhood,
    is_d12_transfer,
    validate_d8_center_transfer_config,
    validate_d8_response_neighborhood_config,
    validate_d12_transfer_config,
)

if TYPE_CHECKING:
    from .config import ExperimentConfig


_D4_D6_MECHANICS_QUALIFICATION_PROJECTION_SHA256 = "ff19c670814113c22fe378616d811906335ed75e8831902bdfd28a0308aa34e0"


def _validate_d4_d6_mechanics_qualification_projection(config: ExperimentConfig) -> None:
    """Admit only the frozen canary config, apart from explicit cell/output bindings."""

    from .telemetry_config import strip_disabled_telemetry

    value = asdict(config)
    if config.recipe.parent_requirement is None:
        value["recipe"].pop("parent_requirement")
    strip_disabled_telemetry(value)
    strip_disabled_snapshot(value)
    strip_disabled_prediction_extensions(value)
    strip_disabled_auxiliary(value)
    strip_disabled_mask_tail(value)
    strip_disabled_bos_packing(value)
    strip_legacy_lr_schedule(value)
    strip_disabled_nanochat_muon(value)
    if value["noise"].pop("mdlm_time_sampling") != "legacy_auto":
        raise ValueError("historical mechanics qualification requires legacy MDLM time sampling")
    if value["noise"].pop("mdlm_unweighted_loss_scale") != 1.0:
        raise ValueError("historical mechanics qualification requires default loss scale")
    if value["model"].pop("backbone_profile") != "nanodllm":
        raise ValueError("the D4/D6 mechanics qualification requires the historical backbone")
    if value["train"].pop("token_edit_backbone_gradient_mode") != "shared":
        raise ValueError("the D4/D6 mechanics qualification requires shared token-edit backbone gradients")
    if value["train"].pop("token_edit_batch_layout") != "whole_microbatch_cadence":
        raise ValueError("the D4/D6 mechanics qualification requires the historical token-edit batch layout")
    for field in ("token_edit_adapter_bottleneck_dim", "token_edit_adapter_top_layers"):
        if value["model"].pop(field) != 0:
            raise ValueError("the D4/D6 mechanics qualification requires token-edit adapters to be disabled")
    if value["model"].pop("token_edit_provenance_embedding") is not False:
        raise ValueError("the D4/D6 mechanics qualification requires token provenance to be disabled")
    if value["model"].pop("token_edit_provenance_confidence_bins") != 0:
        raise ValueError("the D4/D6 mechanics qualification requires confidence provenance to be disabled")
    if value["train"].pop("model_init_allow_token_edit_extensions") is not False:
        raise ValueError("the D4/D6 mechanics qualification forbids extended model initialization")
    if value["train"].pop("token_edit_canvas_mode") != "full_draft":
        raise ValueError("the D4/D6 mechanics qualification requires the historical full-draft canvas")
    if value["train"].pop("token_edit_mixed_canvas_draft_fraction") != 1.0:
        raise ValueError("the D4/D6 mechanics qualification requires the historical full-draft fraction")
    if value["train"].pop("token_edit_trajectory_completion_fraction") != 0.5:
        raise ValueError("the D4/D6 mechanics qualification requires the default trajectory fraction")
    if value["train"].pop("token_edit_trajectory_steps") != 4:
        raise ValueError("the D4/D6 mechanics qualification requires the default trajectory steps")
    if value["train"].pop("token_edit_correction_objective") != "cross_entropy":
        raise ValueError("the D4/D6 mechanics qualification requires the historical correction objective")
    if value["train"].pop("token_edit_correction_aux_weight") != 0.0:
        raise ValueError("the D4/D6 mechanics qualification requires no correction auxiliary")
    if value["train"].pop("token_edit_expected_accuracy_power") != 1.0:
        raise ValueError("the D4/D6 mechanics qualification requires the historical expected-accuracy power")
    if value["train"].pop("token_edit_expected_accuracy_power_schedule") != "constant":
        raise ValueError("the D4/D6 mechanics qualification requires constant expected-accuracy power")
    if value["train"].pop("token_edit_expected_accuracy_corruption_threshold") != 0.8:
        raise ValueError("the D4/D6 mechanics qualification requires the default expected-accuracy threshold")
    if value["train"].pop("token_edit_cross_view_nonregression_weight") != 0.0:
        raise ValueError("the D4/D6 mechanics qualification requires cross-view preservation to be disabled")
    if value["train"].pop("token_edit_draft_m2t_loss_weight") != 0.0:
        raise ValueError("the D4/D6 mechanics qualification requires no joint draft objective")
    frozen_draft_defaults = {
        "token_edit_draft_model_source": "live_student",
        "token_edit_draft_checkpoint": "",
        "token_edit_draft_checkpoint_expected_sha256": "",
        "token_edit_draft_checkpoint_expected_size_bytes": -1,
    }
    for field_name, expected in frozen_draft_defaults.items():
        if value["train"].pop(field_name) != expected:
            raise ValueError(f"the D4/D6 mechanics qualification requires train.{field_name}={expected!r}")
    credit_defaults = {
        "token_edit_draft_m2t_credit_mode": "uniform",
        "token_edit_draft_m2t_credit_strength": 1.0,
        "token_edit_draft_m2t_credit_clip_min": 0.5,
        "token_edit_draft_m2t_credit_clip_max": 1.5,
    }
    for field_name, expected in credit_defaults.items():
        if value["train"].pop(field_name) != expected:
            raise ValueError(f"the D4/D6 mechanics qualification requires train.{field_name}={expected!r}")
    if value["train"].pop("token_edit_full_draft_provenance") is not False:
        raise ValueError("the D4/D6 mechanics qualification requires full-draft provenance to be disabled")
    prefix_defaults = {
        "token_edit_prefix_enabled": False,
        "token_edit_prefix_canvas_mode": "fallible_draft",
        "token_edit_prefix_loss_scope": "all_active",
        "token_edit_prefix_min_fraction": 0.125,
        "token_edit_prefix_max_fraction": 0.75,
        "token_edit_prefix_boundary_seed": 271828,
    }
    for field_name, expected in prefix_defaults.items():
        if value["train"].pop(field_name) != expected:
            raise ValueError(f"the D4/D6 mechanics qualification requires train.{field_name}={expected!r}")
    from .config_validation_block_recipe import project_disabled_block_recipe_defaults

    project_disabled_block_recipe_defaults(value)
    block_conversion_defaults = {
        ("model", "bd3lm_clean_stream_mode"): "block_causal",
        ("train", "bd3lm_uniform_t2t_batch_fraction"): 0.0,
        ("train", "bd3lm_causal_ar_loss_weight"): 0.0,
        ("train", "bd3lm_block_size_schedule"): "fixed",
        ("train", "bd3lm_block_size_schedule_start_step_indices"): [],
        ("train", "bd3lm_block_size_schedule_values"): [],
    }
    for (section, field_name), expected in block_conversion_defaults.items():
        if value[section].pop(field_name) != expected:
            raise ValueError(f"the D4/D6 mechanics qualification requires {section}.{field_name}={expected!r}")
    for section, field in (
        ("model", "depth"),
        ("model", "n_embd"),
        ("model", "n_head"),
        ("model", "mlp_hidden_dim"),
        ("parameterization", "mode"),
        ("parameterization", "contract_sha256"),
    ):
        value[section][field] = "<cell-bound>"
    value["train"]["run_dir"] = "<attempt-bound>"
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != _D4_D6_MECHANICS_QUALIFICATION_PROJECTION_SHA256:
        raise ValueError("the D4/D6 mechanics qualification config differs from its exact frozen field projection")


def validate_parameterization_config(config: ExperimentConfig) -> None:
    from . import ar_width_mup

    if ar_width_mup.active(config):
        ar_width_mup.validate_config(config)
        digest = config.parameterization.contract_sha256
        if not isinstance(digest, str) or (
            digest and (len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest))
        ):
            raise ValueError("AR width muP contract digest must be empty or lowercase SHA-256")
        return
    if config.parameterization.mode not in {"ordinary", "mup", "completep"}:
        raise ValueError("parameterization.mode must be 'ordinary', 'mup', or 'completep'")
    if config.parameterization.execution_scope not in {
        muon_residual_depth.SCOPE,
        nanollada_completep.AR_SCOPE,
        nanollada_completep.MDLM_SCOPE,
        "ordinary",
        "d4_d6_mechanics_qualification",
        D4_ROOT_CALIBRATION_EXECUTION_SCOPE,
        D4_HIDDEN_INIT_CONFIRMATION_EXECUTION_SCOPE,
        D8_CENTER_TRANSFER_EXECUTION_SCOPE,
        D8_RESPONSE_NEIGHBORHOOD_EXECUTION_SCOPE,
        D12_TRANSFER_EXECUTION_SCOPE,
        D20_COMPLETEP_TPP20_EXECUTION_SCOPE,
        COMPLETEP_HIDDEN_INIT_TRANSFER_EXECUTION_SCOPE,
        D8_BATCH_OBJECTIVE_SCOPE,
    }:
        raise ValueError(
            "parameterization.execution_scope must be 'ordinary', 'ar_completep_transfer', "
            "'mdlm_completep_transfer', 'd4_d6_mechanics_qualification', "
            "'d4_mdlm_parameterization_root_calibration', 'd4_mdlm_hidden_init_confirmation', "
            "'d8_mdlm_parameterization_center_transfer_screen', "
            "'d8_mdlm_parameterization_response_neighborhood', or "
            "'d12_mdlm_parameterization_transfer_confirmation', "
            "'d20_mdlm_completep_tpp20', or 'd8_d12_mdlm_completep_hidden_init_0p5_transfer'"
        )
    for name in ("base_depth", "base_width"):
        value = getattr(config.parameterization, name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"parameterization.{name} must be a positive integer")
    if (config.parameterization.base_depth, config.parameterization.base_width) != (4, 256):
        raise ValueError("the frozen ratio-64 parameterization contract requires base_depth=4 and base_width=256")
    hidden_init_multiplier = config.parameterization.hidden_init_std_multiplier
    if (
        isinstance(hidden_init_multiplier, bool)
        or not isinstance(hidden_init_multiplier, (int, float))
        or not math.isfinite(float(hidden_init_multiplier))
        or float(hidden_init_multiplier) <= 0
    ):
        raise ValueError("parameterization.hidden_init_std_multiplier must be finite and positive")
    if (
        config.parameterization.mode == "ordinary"
        and not is_d4_root_calibration(config)
        and not is_d4_hidden_init_confirmation(config)
        and float(hidden_init_multiplier) != 1.0
    ):
        raise ValueError("ordinary parameterization requires hidden_init_std_multiplier=1.0")
    if config.parameterization.contract_binding not in {"legacy_unbound", "required"}:
        raise ValueError("parameterization.contract_binding must be 'legacy_unbound' or 'required'")
    parameterization_sha256 = config.parameterization.contract_sha256
    if not isinstance(parameterization_sha256, str) or (
        parameterization_sha256
        and (
            len(parameterization_sha256) != 64
            or any(character not in "0123456789abcdef" for character in parameterization_sha256)
        )
    ):
        raise ValueError("parameterization.contract_sha256 must be empty or a lowercase 64-hex digest")
    if config.parameterization.contract_binding == "legacy_unbound" and parameterization_sha256:
        raise ValueError("legacy_unbound parameterization.contract_binding requires an empty contract_sha256")
    if config.parameterization.execution_scope == "d4_d6_mechanics_qualification":
        if config.recipe.name != "d4-d6-parameterization-mechanics-canary-v1" or config.recipe.status != "locked":
            raise ValueError("D4/D6 mechanics qualification requires its locked canary recipe identity")
        if config.parameterization.contract_binding != "required":
            raise ValueError("D4/D6 mechanics qualification requires parameterization.contract_binding='required'")
    if nanollada_completep.is_completep(config):
        nanollada_completep.validate_config(config)
    elif muon_residual_depth.active(config):
        muon_residual_depth.validate_config(config)
    elif config.parameterization.mode != "ordinary":
        if config.parameterization.execution_scope not in {
            "d4_d6_mechanics_qualification",
            D4_ROOT_CALIBRATION_EXECUTION_SCOPE,
            D8_CENTER_TRANSFER_EXECUTION_SCOPE,
            D8_RESPONSE_NEIGHBORHOOD_EXECUTION_SCOPE,
            D12_TRANSFER_EXECUTION_SCOPE,
            D20_COMPLETEP_TPP20_EXECUTION_SCOPE,
            COMPLETEP_HIDDEN_INIT_TRANSFER_EXECUTION_SCOPE,
            D8_BATCH_OBJECTIVE_SCOPE,
        }:
            raise ValueError(
                "mup and completep execution requires execution_scope='d4_d6_mechanics_qualification' "
                "or a reviewed D4 calibration, D8 transfer, D12 confirmation, D20 CompleteP, "
                "or D8/D12 CompleteP hidden-init scope"
            )
        if config.parameterization.contract_binding != "required":
            raise ValueError("mup and completep require parameterization.contract_binding='required'")
        if config.optim.type != "adamw":
            raise ValueError("mup and completep support only optim.type='adamw'")
        if config.model.depth * 64 != config.model.n_embd:
            raise ValueError("mup and completep are frozen only for the ratio-64 depth/width ladder")
        if config.model.n_embd // config.model.n_head != 128:
            raise ValueError("mup and completep require fixed head_dim=128")
        if config.model.mlp_hidden_dim not in {-1, 4 * config.model.n_embd}:
            raise ValueError("mup and completep require the 4 * n_embd SwiGLU hidden width")
        if config.model.attention_scale != -1.0:
            raise ValueError("mup and completep derive attention scale; model.attention_scale must remain -1")
        if config.model.local_attention_lower_layers or config.model.local_attention_window:
            raise ValueError("mup and completep do not yet support local attention")
        if config.model.auxiliary_head_layers:
            raise ValueError("mup and completep do not yet support auxiliary output heads")
        if config.model.output_factorization != "flat":
            raise ValueError("mup and completep require model.output_factorization='flat'")
        if config.model.causal:
            raise ValueError("the frozen NanoDLLM mup and completep adaptation requires non-causal MDLM")
        if config.model.length_head or config.model.target_length_conditioning or config.model.mask_position_embedding:
            raise ValueError("mup and completep do not yet support optional conditioning or length roles")
        if config.noise.type != "mdlm" or config.noise.time_conditioning:
            raise ValueError("the frozen NanoDLLM mup and completep adaptation requires unconditioned MDLM")
    if config.parameterization.execution_scope == "d4_d6_mechanics_qualification":
        expected_shape = {
            4: (256, 2, 1024),
            6: (384, 3, 1536),
        }.get(int(config.model.depth))
        observed_shape = (int(config.model.n_embd), int(config.model.n_head), int(config.model.mlp_hidden_dim))
        if expected_shape is None or observed_shape != expected_shape:
            raise ValueError("d4_d6_mechanics_qualification permits only the frozen D4/D6 ratio-64 model shapes")
        if int(config.train.num_steps) != 8:
            raise ValueError("the D4/D6 mechanics qualification requires exactly 8 optimizer steps")
        if (
            int(config.model.max_seq_len) != 2048
            or int(config.train.target_tokens) != 131072
            or int(config.train.total_batch_size) != 16384
            or int(config.train.device_batch_size) != 8
            or int(config.train.log_interval) != 1
            or int(config.train.save_every) != -1
            or bool(config.train.eval_only)
            or bool(config.eval.enabled)
        ):
            raise ValueError("the D4/D6 mechanics qualification requires the frozen bounded training budget")
        if (
            config.runtime.device_type != "cuda"
            or config.runtime.dtype != "bfloat16"
            or not config.runtime.compile
            or not config.runtime.compile_dynamic
        ):
            raise ValueError("the D4/D6 mechanics qualification requires compiled CUDA BF16 execution")
        if config.diagnostics.parameter_state_hash_steps != [1, 8]:
            raise ValueError("the D4/D6 mechanics qualification requires parameter-state receipts at steps 1 and 8")
        _validate_d4_d6_mechanics_qualification_projection(config)
    if is_d4_root_calibration(config):
        validate_d4_root_calibration_config(config)
    if is_d4_hidden_init_confirmation(config):
        validate_d4_hidden_init_confirmation_config(config)
    if is_d8_center_transfer(config):
        validate_d8_center_transfer_config(config)
    if is_d8_response_neighborhood(config):
        validate_d8_response_neighborhood_config(config)
    if is_d12_transfer(config):
        validate_d12_transfer_config(config)
    if is_d20_completep_tpp20(config):
        validate_d20_completep_tpp20_config(config)
    if is_completep_hidden_init_transfer(config):
        validate_completep_hidden_init_transfer_config(config)
    if config.parameterization.execution_scope == D8_BATCH_OBJECTIVE_SCOPE:
        validate_d8_batch_objective_config(config)
