"""Token-edit objectives, canvas selection, and backbone compatibility."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config_validation_token_edit_draft import validate_draft_credit, validate_draft_inputs

if TYPE_CHECKING:
    from .config import ExperimentConfig


def validate_token_edit(config: ExperimentConfig) -> None:
    validate_draft_inputs(config)
    correction_objectives = {
        "cross_entropy",
        "expected_accuracy",
        "cross_entropy_plus_expected_accuracy",
        "cross_entropy_plus_draft_preference",
    }
    if config.train.token_edit_correction_objective not in correction_objectives:
        raise ValueError(
            "train.token_edit_correction_objective must be 'cross_entropy', 'expected_accuracy', "
            "'cross_entropy_plus_expected_accuracy', or 'cross_entropy_plus_draft_preference'"
        )
    correction_aux_weight = float(config.train.token_edit_correction_aux_weight)
    expected_accuracy_power = float(config.train.token_edit_expected_accuracy_power)
    expected_accuracy_power_schedule = config.train.token_edit_expected_accuracy_power_schedule
    expected_accuracy_corruption_threshold = float(config.train.token_edit_expected_accuracy_corruption_threshold)
    cross_view_nonregression_weight = float(config.train.token_edit_cross_view_nonregression_weight)
    if not 0.0 < expected_accuracy_power <= 1.0:
        raise ValueError("train.token_edit_expected_accuracy_power must be in (0, 1]")
    if expected_accuracy_power_schedule not in ("constant", "corruption_threshold"):
        raise ValueError(
            "train.token_edit_expected_accuracy_power_schedule must be 'constant' or 'corruption_threshold'"
        )
    if not 0.0 < expected_accuracy_corruption_threshold < 1.0:
        raise ValueError("train.token_edit_expected_accuracy_corruption_threshold must be in (0, 1)")
    if expected_accuracy_power_schedule == "constant":
        if expected_accuracy_corruption_threshold != 0.8:
            raise ValueError("constant expected-accuracy power requires the default corruption threshold 0.8")
    elif expected_accuracy_power >= 1.0:
        raise ValueError("corruption-threshold expected-accuracy power requires a configured power below 1")
    if cross_view_nonregression_weight < 0.0:
        raise ValueError("train.token_edit_cross_view_nonregression_weight must be non-negative")
    correction_objective_is_default = config.train.token_edit_correction_objective == "cross_entropy"
    correction_objective_is_pure_reward = config.train.token_edit_correction_objective == "expected_accuracy"
    if correction_objective_is_default or correction_objective_is_pure_reward:
        if correction_aux_weight != 0.0:
            raise ValueError(
                f"train.token_edit_correction_objective={config.train.token_edit_correction_objective!r} "
                "requires token_edit_correction_aux_weight=0"
            )
    elif correction_aux_weight <= 0.0:
        raise ValueError("combined token-edit correction objectives require token_edit_correction_aux_weight>0")
    if config.train.token_edit_backbone_gradient_mode not in ("shared", "output_head_only", "residual_adapter"):
        raise ValueError(
            "train.token_edit_backbone_gradient_mode must be 'shared', 'output_head_only', or 'residual_adapter'"
        )
    trajectory_canvas_modes = {"trajectory_gold_context", "trajectory_model_context"}
    if config.train.token_edit_canvas_mode not in {"full_draft", "mixed_mask_draft", *trajectory_canvas_modes}:
        raise ValueError(
            "train.token_edit_canvas_mode must be 'full_draft', 'mixed_mask_draft', "
            "'trajectory_gold_context', or 'trajectory_model_context'"
        )
    if not isinstance(config.train.token_edit_full_draft_provenance, bool):
        raise ValueError("train.token_edit_full_draft_provenance must be a boolean")
    if not isinstance(config.train.token_edit_prefix_enabled, bool):
        raise ValueError("train.token_edit_prefix_enabled must be a boolean")
    if config.train.token_edit_prefix_canvas_mode not in ("fallible_draft", "clean"):
        raise ValueError("train.token_edit_prefix_canvas_mode must be 'fallible_draft' or 'clean'")
    if config.train.token_edit_prefix_loss_scope not in ("all_active", "suffix_active"):
        raise ValueError("train.token_edit_prefix_loss_scope must be 'all_active' or 'suffix_active'")
    prefix_min_fraction = float(config.train.token_edit_prefix_min_fraction)
    prefix_max_fraction = float(config.train.token_edit_prefix_max_fraction)
    if not 0.0 < prefix_min_fraction <= prefix_max_fraction < 1.0:
        raise ValueError("train token-edit prefix fractions must satisfy 0 < min <= max < 1")
    if (
        isinstance(config.train.token_edit_prefix_boundary_seed, bool)
        or not isinstance(config.train.token_edit_prefix_boundary_seed, int)
        or config.train.token_edit_prefix_boundary_seed < 0
    ):
        raise ValueError("train.token_edit_prefix_boundary_seed must be a non-negative integer")
    mixed_canvas = config.train.token_edit_canvas_mode == "mixed_mask_draft"
    trajectory_canvas = config.train.token_edit_canvas_mode in trajectory_canvas_modes
    token_edit_enabled = (
        config.train.token_edit_corruption_batch_fraction > 0
        or config.train.token_edit_corruption_ratio > 0
        or config.train.token_edit_corruption_schedule != "legacy_fixed_ratio"
        or config.train.token_edit_input_source != "random_replacement"
        or config.train.token_edit_batch_layout != "whole_microbatch_cadence"
        or config.train.token_edit_prefix_enabled
        or config.train.token_edit_draft_model_source != "live_student"
    )
    if config.train.token_edit_draft_model_source == "frozen_checkpoint":
        if config.train.token_edit_input_source != "self_generated_argmax":
            raise ValueError("frozen token-edit draft checkpoints require self-generated argmax drafts")
        if config.train.token_edit_canvas_mode != "full_draft":
            raise ValueError("frozen token-edit draft checkpoints currently require the full-draft canvas")
        if config.train.token_edit_prefix_enabled:
            raise ValueError("frozen token-edit draft checkpoints do not support token-edit prefix training")
        if float(config.train.token_edit_draft_m2t_loss_weight) != 0.0:
            raise ValueError("frozen token-edit draft checkpoints require a stopped-gradient draft pass")
        if config.train.token_edit_backbone_gradient_mode != "shared":
            raise ValueError("frozen token-edit draft checkpoints currently require shared backbone gradients")
    if config.train.token_edit_prefix_enabled:
        if config.train.token_edit_input_source != "self_generated_argmax":
            raise ValueError("token-edit prefix training requires self-generated token editing")
        if config.train.token_edit_batch_layout != "whole_microbatch_cadence":
            raise ValueError("token-edit prefix training requires whole_microbatch_cadence")
        if config.train.token_edit_canvas_mode != "full_draft":
            raise ValueError("token-edit prefix training requires token_edit_canvas_mode='full_draft'")
        if config.train.token_edit_prefix_loss_scope != "suffix_active":
            raise ValueError("enabled token-edit prefix training requires suffix_active loss")
    elif (
        config.train.token_edit_prefix_canvas_mode != "fallible_draft"
        or config.train.token_edit_prefix_loss_scope != "all_active"
        or prefix_min_fraction != 0.125
        or prefix_max_fraction != 0.75
        or config.train.token_edit_prefix_boundary_seed != 271828
    ):
        raise ValueError("disabled token-edit prefix controls must retain their defaults")
    fixed_row_layout = config.train.token_edit_batch_layout == "fixed_12_m2t_4_t2t_rows"
    if fixed_row_layout:
        if config.train.device_batch_size != 16:
            raise ValueError("fixed_12_m2t_4_t2t_rows requires train.device_batch_size=16")
        if config.train.total_batch_size != 16 * config.model.max_seq_len:
            raise ValueError(
                "fixed_12_m2t_4_t2t_rows requires one DB16 microbatch per optimizer update "
                "(train.total_batch_size=16*model.max_seq_len)"
            )
        if config.train.allow_batch_size_round_up:
            raise ValueError("fixed_12_m2t_4_t2t_rows requires train.allow_batch_size_round_up=false")
        if float(config.train.token_edit_corruption_batch_fraction) != 0.25:
            raise ValueError("fixed_12_m2t_4_t2t_rows requires token_edit_corruption_batch_fraction=0.25")
        if config.train.token_edit_input_source != "self_generated_argmax":
            raise ValueError("fixed_12_m2t_4_t2t_rows requires self-generated token editing")
        if config.train.token_edit_corruption_schedule != "match_mdlm_per_row":
            raise ValueError("fixed_12_m2t_4_t2t_rows requires token_edit_corruption_schedule='match_mdlm_per_row'")
        if config.train.token_edit_backbone_gradient_mode != "shared":
            raise ValueError("fixed_12_m2t_4_t2t_rows requires shared token-edit backbone gradients")
        if config.train.token_edit_canvas_mode != "full_draft":
            raise ValueError("fixed_12_m2t_4_t2t_rows requires the full-draft correction canvas")
        if config.train.token_edit_correction_objective != "expected_accuracy":
            raise ValueError("fixed_12_m2t_4_t2t_rows requires the expected-accuracy correction objective")
    draft_fraction = float(config.train.token_edit_mixed_canvas_draft_fraction)
    if mixed_canvas:
        if not 0.0 < draft_fraction < 1.0:
            raise ValueError("mixed-mask/draft canvases require token_edit_mixed_canvas_draft_fraction in (0, 1)")
    elif draft_fraction != 1.0:
        raise ValueError("full-draft canvases require token_edit_mixed_canvas_draft_fraction=1")
    trajectory_fraction = float(config.train.token_edit_trajectory_completion_fraction)
    trajectory_steps = config.train.token_edit_trajectory_steps
    if isinstance(trajectory_steps, bool) or not isinstance(trajectory_steps, int):
        raise ValueError("train.token_edit_trajectory_steps must be an integer")
    if trajectory_canvas:
        if not 0.0 < trajectory_fraction < 1.0:
            raise ValueError("trajectory canvases require token_edit_trajectory_completion_fraction in (0, 1)")
        if trajectory_steps < 2:
            raise ValueError("trajectory canvases require token_edit_trajectory_steps>=2")
        if config.train.token_edit_input_source != "self_generated_argmax":
            raise ValueError("trajectory canvases require self-generated token editing")
        if config.train.token_edit_batch_layout != "whole_microbatch_cadence":
            raise ValueError("trajectory canvases require whole_microbatch_cadence")
        if config.train.token_edit_prefix_enabled:
            raise ValueError("trajectory canvases and token-edit prefix training must be screened independently")
        if config.train.token_edit_backbone_gradient_mode != "shared":
            raise ValueError("trajectory canvases require shared backbone gradients")
        if float(config.train.token_edit_draft_m2t_loss_weight) != 0.0:
            raise ValueError("trajectory canvases require stopped-gradient rollout passes")
        if config.model.token_edit_provenance_embedding or config.train.token_edit_full_draft_provenance:
            raise ValueError("trajectory canvases must be screened without provenance extensions")
    elif trajectory_fraction != 0.5 or trajectory_steps != 4:
        raise ValueError("non-trajectory canvases require default trajectory controls 0.5/4")
    if config.train.token_edit_input_source == "random_replacement":
        if config.train.token_edit_correct_draft_loss_weight != 0:
            raise ValueError("random-replacement token editing requires token_edit_correct_draft_loss_weight=0")
        if config.train.token_edit_wrong_draft_loss_weight != 1:
            raise ValueError("random-replacement token editing requires token_edit_wrong_draft_loss_weight=1")
    if not correction_objective_is_default:
        if not token_edit_enabled or config.train.token_edit_input_source != "self_generated_argmax":
            raise ValueError("non-default correction objectives require enabled self-generated token editing")
        if (
            float(config.train.token_edit_correct_draft_loss_weight) != 1.0
            or float(config.train.token_edit_wrong_draft_loss_weight) != 1.0
        ):
            raise ValueError("non-default correction objectives require correct/wrong draft weights 1/1")
    corruption_conditioned_expected_accuracy = expected_accuracy_power_schedule == "corruption_threshold"
    generalized_expected_accuracy = expected_accuracy_power != 1.0
    cross_view_nonregression = cross_view_nonregression_weight > 0.0
    if generalized_expected_accuracy and cross_view_nonregression:
        raise ValueError("generalized expected accuracy and cross-view non-regression must be screened independently")
    if generalized_expected_accuracy or cross_view_nonregression:
        mechanism_name = (
            "corruption-conditioned expected accuracy"
            if corruption_conditioned_expected_accuracy
            else "generalized expected accuracy"
            if generalized_expected_accuracy
            else "cross-view non-regression"
        )
        if not token_edit_enabled or config.train.token_edit_input_source != "self_generated_argmax":
            raise ValueError(f"{mechanism_name} requires enabled self-generated token editing")
        if config.train.token_edit_correction_objective != "expected_accuracy":
            raise ValueError(f"{mechanism_name} requires the expected_accuracy correction objective")
        if config.train.token_edit_correction_aux_weight != 0.0:
            raise ValueError(f"{mechanism_name} requires token_edit_correction_aux_weight=0")
        if config.train.token_edit_batch_layout != "whole_microbatch_cadence":
            raise ValueError(f"{mechanism_name} requires whole_microbatch_cadence")
        if config.train.token_edit_canvas_mode != "full_draft":
            raise ValueError(f"{mechanism_name} requires token_edit_canvas_mode='full_draft'")
        if config.train.token_edit_prefix_enabled:
            raise ValueError(f"{mechanism_name} does not support token-edit prefix training")
        if config.train.token_edit_backbone_gradient_mode != "shared":
            raise ValueError(f"{mechanism_name} requires shared backbone gradients")
        if float(config.train.token_edit_draft_m2t_loss_weight) != 0.0:
            raise ValueError(f"{mechanism_name} requires a stopped-gradient draft pass")
        if config.model.token_edit_provenance_embedding or config.train.token_edit_full_draft_provenance:
            raise ValueError(f"{mechanism_name} must be screened without provenance extensions")
    isolated_gradient_mode = config.train.token_edit_backbone_gradient_mode in {
        "output_head_only",
        "residual_adapter",
    }
    if isolated_gradient_mode and (
        not token_edit_enabled or config.train.token_edit_input_source != "self_generated_argmax"
    ):
        raise ValueError("isolated token-edit gradients require enabled self-generated token editing")
    if isolated_gradient_mode and config.runtime.init_single_rank_process_group:
        raise ValueError("isolated token-edit gradients currently require single-process training")
    if config.train.token_edit_backbone_gradient_mode == "residual_adapter":
        if config.model.token_edit_adapter_bottleneck_dim <= 0:
            raise ValueError("residual-adapter gradients require model.token_edit_adapter_bottleneck_dim>0")
        if config.model.token_edit_adapter_top_layers <= 0:
            raise ValueError("residual-adapter gradients require model.token_edit_adapter_top_layers>0")
        if mixed_canvas:
            raise ValueError("residual-adapter and mixed-canvas mechanisms must be screened independently")
    elif config.model.token_edit_adapter_bottleneck_dim != 0 or config.model.token_edit_adapter_top_layers != 0:
        raise ValueError("token-edit adapter parameters require token_edit_backbone_gradient_mode='residual_adapter'")
    full_draft_provenance = config.train.token_edit_full_draft_provenance
    if mixed_canvas:
        if config.train.token_edit_input_source != "self_generated_argmax":
            raise ValueError("mixed-mask/draft canvases require self-generated token editing")
        if not config.model.token_edit_provenance_embedding:
            raise ValueError("mixed-mask/draft canvases require model.token_edit_provenance_embedding=true")
        if full_draft_provenance:
            raise ValueError("mixed-mask/draft canvases cannot enable full-draft provenance")
        if config.model.token_edit_provenance_confidence_bins > 0:
            raise ValueError("confidence-binned provenance is restricted to full-draft canvases")
    elif full_draft_provenance:
        if config.train.token_edit_input_source != "self_generated_argmax":
            raise ValueError("full-draft provenance requires self-generated token editing")
        if not config.model.token_edit_provenance_embedding:
            raise ValueError("full-draft provenance requires model.token_edit_provenance_embedding=true")
        if config.train.token_edit_prefix_enabled:
            raise ValueError("full-draft provenance does not support token-edit prefix training")
    elif config.model.token_edit_provenance_embedding and token_edit_enabled:
        raise ValueError("token-edit provenance embeddings require a provenance-enabled token-edit canvas")
    validate_draft_credit(config)
    if token_edit_enabled:
        if config.train.token_edit_corruption_batch_fraction <= 0:
            raise ValueError("token-edit corruption requires a positive batch fraction")
        if config.train.token_edit_input_source == "self_generated_argmax":
            if config.train.token_edit_corruption_schedule != "match_mdlm_per_row":
                raise ValueError(
                    "self-generated token editing requires token_edit_corruption_schedule='match_mdlm_per_row'"
                )
            if config.train.token_edit_corruption_ratio != 0:
                raise ValueError("self-generated token editing requires token_edit_corruption_ratio=0")
            if config.train.token_edit_correct_draft_loss_weight <= 0:
                raise ValueError("self-generated token editing requires token_edit_correct_draft_loss_weight>0")
            if config.train.token_edit_wrong_draft_loss_weight <= 0:
                raise ValueError("self-generated token editing requires token_edit_wrong_draft_loss_weight>0")
            if config.data.loss_on_bos:
                raise ValueError("self-generated token editing requires data.loss_on_bos=false")
        elif config.train.token_edit_corruption_schedule == "legacy_fixed_ratio":
            if config.train.token_edit_corruption_ratio <= 0:
                raise ValueError("legacy fixed-ratio token-edit corruption requires a positive token ratio")
        elif config.train.token_edit_corruption_ratio != 0:
            raise ValueError("matched-MDLM token-edit corruption requires token_edit_corruption_ratio=0")
        if config.model.causal:
            raise ValueError("token-edit corruption is only supported for non-causal MDLM training")
        if config.noise.type != "mdlm":
            raise ValueError("token-edit corruption requires noise.type='mdlm'")
        if not config.noise.schedule_weight:
            raise ValueError("token-edit corruption preserves strict MDLM and requires schedule_weight=true")
        if config.noise.complement_masking:
            raise ValueError("token-edit corruption preserves strict MDLM and requires complement_masking=false")
        if config.noise.suffix_masking:
            raise ValueError(
                "token-edit corruption preserves independent MDLM masking and requires suffix_masking=false"
            )
        if config.noise.time_conditioning:
            raise ValueError("token-edit corruption preserves no time conditioning")
        if config.noise.per_token_noise_levels:
            raise ValueError(
                "token-edit corruption preserves per-row MDLM times and requires per_token_noise_levels=false"
            )
        if config.model.bd3lm_block_size != 0:
            raise ValueError("token-edit corruption cannot be coupled with BD3LM")
        if config.model.block_causal_pilot_size != 0:
            raise ValueError("token-edit corruption cannot be coupled with block-causal pilot training")
        if config.model.output_factorization != "flat":
            raise ValueError("token-edit corruption requires model.output_factorization='flat'")
        if config.model.auxiliary_head_layers:
            raise ValueError("token-edit corruption requires model.auxiliary_head_layers to be empty")
        if config.model.length_head:
            raise ValueError("token-edit corruption cannot be coupled with length-head training")
        if config.model.target_length_conditioning:
            raise ValueError("token-edit corruption cannot be coupled with target-length conditioning")
        if config.train.auxiliary_loss_weight != 0:
            raise ValueError("token-edit corruption cannot be coupled with auxiliary loss")
        if config.train.entropy_bonus_beta != 0:
            raise ValueError("token-edit corruption cannot be coupled with entropy bonus")
        if config.train.confidence_penalty_weight != 0:
            raise ValueError("token-edit corruption cannot be coupled with confidence penalty")
        if config.train.z_loss_weight != 0:
            raise ValueError("token-edit corruption cannot be coupled with z-loss")
        if config.train.eos_hazard_loss_weight != 0:
            raise ValueError("token-edit corruption cannot be coupled with EOS hazard loss")
        if config.train.length_head_loss_weight != 0:
            raise ValueError("token-edit corruption cannot be coupled with length-head loss")
        if config.train.position_decile_balanced_loss:
            raise ValueError("token-edit corruption cannot be coupled with position decile balancing")
        if config.train.uniform_loss_estimator != "single_sample":
            raise ValueError("token-edit corruption cannot be coupled with uniform loss estimators")
