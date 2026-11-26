"""Compatibility checks for the three EOS canary protocols."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import ExperimentConfig


def validate_learned_eos_sampler(config: ExperimentConfig) -> None:
    if config.train.strict_mdlm_learned_eos_sampler_canary_only:
        if config.train.eval_only:
            raise ValueError("learned EOS sampler canary requires train.eval_only=false")
        if config.data.synthetic_dataset != "eos_overfit_canary":
            raise ValueError("learned EOS sampler canary requires data.synthetic_dataset='eos_overfit_canary'")
        if config.data.synthetic_eos_overfit_loss_mask_policy != "eos_only":
            raise ValueError(
                "learned EOS sampler canary requires data.synthetic_eos_overfit_loss_mask_policy='eos_only'"
            )
        if config.data.train_data:
            raise ValueError("learned EOS sampler canary uses only the synthetic in-memory dataset")
        if config.data.val_data:
            raise ValueError("learned EOS sampler canary uses the synthetic in-memory validation batch")
        if config.data.tokenizer != "byte":
            raise ValueError("learned EOS sampler canary requires data.tokenizer='byte'")
        if not config.data.append_eos:
            raise ValueError("learned EOS sampler canary requires data.append_eos=true")
        if not config.data.loss_on_eos:
            raise ValueError("learned EOS sampler canary requires data.loss_on_eos=true")
        if config.data.random_length_ratio != 0.0:
            raise ValueError("learned EOS sampler canary requires data.random_length_ratio=0")
        if config.data.continuation_ratio != 0.0:
            raise ValueError("learned EOS sampler canary requires data.continuation_ratio=0")
        if config.data.boundary_aware_packing:
            raise ValueError("learned EOS sampler canary requires data.boundary_aware_packing=false")
        if config.data.content_budget_segmented_packing:
            raise ValueError("learned EOS sampler canary requires data.content_budget_segmented_packing=false")
        if config.data.short_document_curriculum:
            raise ValueError("learned EOS sampler canary requires data.short_document_curriculum=false")
        if config.model.causal:
            raise ValueError("learned EOS sampler canary requires model.causal=false")
        if config.model.output_factorization != "flat":
            raise ValueError("learned EOS sampler canary requires model.output_factorization='flat'")
        if config.model.auxiliary_head_layers:
            raise ValueError("learned EOS sampler canary cannot be coupled with auxiliary heads")
        if config.model.bd3lm_block_size != 0:
            raise ValueError("learned EOS sampler canary cannot be coupled with BD3LM")
        if config.model.block_causal_pilot_size != 0:
            raise ValueError("learned EOS sampler canary cannot be coupled with block-causal pilot training")
        if config.model.length_head:
            raise ValueError("learned EOS sampler canary cannot be coupled with length-head training")
        if config.model.target_length_conditioning:
            raise ValueError("learned EOS sampler canary cannot be coupled with target-length conditioning")
        if config.noise.type != "mdlm":
            raise ValueError("learned EOS sampler canary requires noise.type='mdlm'")
        if not config.noise.schedule_weight:
            raise ValueError("learned EOS sampler canary requires noise.schedule_weight=true")
        if config.noise.complement_masking:
            raise ValueError("learned EOS sampler canary requires noise.complement_masking=false")
        if config.noise.suffix_masking:
            raise ValueError("learned EOS sampler canary requires noise.suffix_masking=false")
        if config.noise.time_conditioning:
            raise ValueError("learned EOS sampler canary preserves no time conditioning")
        if config.noise.per_token_noise_levels:
            raise ValueError("learned EOS sampler canary requires noise.per_token_noise_levels=false")
        if config.train.auxiliary_loss_weight != 0:
            raise ValueError("learned EOS sampler canary requires auxiliary_loss_weight=0")
        if config.train.entropy_bonus_beta != 0:
            raise ValueError("learned EOS sampler canary requires entropy_bonus_beta=0")
        if config.train.confidence_penalty_weight != 0:
            raise ValueError("learned EOS sampler canary requires confidence_penalty_weight=0")
        if config.train.z_loss_weight != 0:
            raise ValueError("learned EOS sampler canary requires z_loss_weight=0")
        if config.train.eos_hazard_loss_weight != 0:
            raise ValueError("learned EOS sampler canary requires eos_hazard_loss_weight=0")
        if config.train.length_head_loss_weight != 0:
            raise ValueError("learned EOS sampler canary requires length_head_loss_weight=0")
        if config.train.position_decile_balanced_loss:
            raise ValueError("learned EOS sampler canary requires position_decile_balanced_loss=false")
        if config.train.uniform_loss_estimator != "single_sample":
            raise ValueError("learned EOS sampler canary requires uniform_loss_estimator='single_sample'")
        if config.train.token_edit_corruption_batch_fraction != 0:
            raise ValueError("learned EOS sampler canary requires token_edit_corruption_batch_fraction=0")
        if config.train.token_edit_corruption_ratio != 0:
            raise ValueError("learned EOS sampler canary requires token_edit_corruption_ratio=0")
        if config.eval.sampler != "fhs":
            raise ValueError("learned EOS sampler canary requires eval.sampler='fhs'")
        if config.eval.sample_token_selection != "argmax":
            raise ValueError("learned EOS sampler canary requires eval.sample_token_selection='argmax'")
        if not config.eval.sample_trace:
            raise ValueError("learned EOS sampler canary requires eval.sample_trace=true")
        if config.eval.length_policy != "none":
            raise ValueError("learned EOS sampler canary requires eval.length_policy='none'")
        if config.eval.fhs_tau >= 1.0:
            raise ValueError("learned EOS sampler canary requires eval.fhs_tau < 1")
        if config.eval.gen_len < 4:
            raise ValueError("learned EOS sampler canary requires eval.gen_len >= 4")


def validate_eos_reachability(config: ExperimentConfig) -> None:
    if config.train.strict_mdlm_sampler_eos_reachability_canary_only:
        if config.train.eval_only:
            raise ValueError("sampler EOS reachability canary requires train.eval_only=false")
        if config.data.tokenizer != "byte":
            raise ValueError("sampler EOS reachability canary requires data.tokenizer='byte'")
        if config.data.train_data:
            raise ValueError("sampler EOS reachability canary uses controlled logits and requires data.train_data=[]")
        if config.data.val_data:
            raise ValueError("sampler EOS reachability canary uses controlled logits and requires data.val_data=[]")
        if config.data.synthetic_dataset:
            raise ValueError("sampler EOS reachability canary uses controlled logits and requires no synthetic dataset")
        if not config.data.append_eos:
            raise ValueError("sampler EOS reachability canary requires data.append_eos=true")
        if config.data.boundary_aware_packing:
            raise ValueError("sampler EOS reachability canary requires data.boundary_aware_packing=false")
        if config.data.content_budget_segmented_packing:
            raise ValueError("sampler EOS reachability canary requires data.content_budget_segmented_packing=false")
        if config.data.short_document_curriculum:
            raise ValueError("sampler EOS reachability canary requires data.short_document_curriculum=false")
        if config.model.causal:
            raise ValueError("sampler EOS reachability canary requires model.causal=false")
        if config.model.output_factorization != "flat":
            raise ValueError("sampler EOS reachability canary requires model.output_factorization='flat'")
        if config.model.auxiliary_head_layers:
            raise ValueError("sampler EOS reachability canary cannot be coupled with auxiliary heads")
        if config.model.bd3lm_block_size != 0:
            raise ValueError("sampler EOS reachability canary cannot be coupled with BD3LM")
        if config.model.block_causal_pilot_size != 0:
            raise ValueError("sampler EOS reachability canary cannot be coupled with block-causal pilot training")
        if config.model.length_head:
            raise ValueError("sampler EOS reachability canary cannot be coupled with length-head training")
        if config.model.target_length_conditioning:
            raise ValueError("sampler EOS reachability canary cannot be coupled with target-length conditioning")
        if config.noise.type != "mdlm":
            raise ValueError("sampler EOS reachability canary requires noise.type='mdlm'")
        if not config.noise.schedule_weight:
            raise ValueError("sampler EOS reachability canary requires noise.schedule_weight=true")
        if config.noise.complement_masking:
            raise ValueError("sampler EOS reachability canary requires noise.complement_masking=false")
        if config.noise.suffix_masking:
            raise ValueError("sampler EOS reachability canary requires noise.suffix_masking=false")
        if config.noise.time_conditioning:
            raise ValueError("sampler EOS reachability canary preserves no time conditioning")
        if config.noise.per_token_noise_levels:
            raise ValueError("sampler EOS reachability canary requires noise.per_token_noise_levels=false")
        if config.train.auxiliary_loss_weight != 0:
            raise ValueError("sampler EOS reachability canary requires auxiliary_loss_weight=0")
        if config.train.entropy_bonus_beta != 0:
            raise ValueError("sampler EOS reachability canary requires entropy_bonus_beta=0")
        if config.train.confidence_penalty_weight != 0:
            raise ValueError("sampler EOS reachability canary requires confidence_penalty_weight=0")
        if config.train.z_loss_weight != 0:
            raise ValueError("sampler EOS reachability canary requires z_loss_weight=0")
        if config.train.eos_hazard_loss_weight != 0:
            raise ValueError("sampler EOS reachability canary requires eos_hazard_loss_weight=0")
        if config.train.length_head_loss_weight != 0:
            raise ValueError("sampler EOS reachability canary requires length_head_loss_weight=0")
        if config.train.position_decile_balanced_loss:
            raise ValueError("sampler EOS reachability canary requires position_decile_balanced_loss=false")
        if config.train.uniform_loss_estimator != "single_sample":
            raise ValueError("sampler EOS reachability canary requires uniform_loss_estimator='single_sample'")
        if config.train.token_edit_corruption_batch_fraction != 0:
            raise ValueError("sampler EOS reachability canary requires token_edit_corruption_batch_fraction=0")
        if config.train.token_edit_corruption_ratio != 0:
            raise ValueError("sampler EOS reachability canary requires token_edit_corruption_ratio=0")
        if config.eval.sampler != "fhs":
            raise ValueError("sampler EOS reachability canary requires eval.sampler='fhs'")
        if config.eval.sample_token_selection != "argmax":
            raise ValueError("sampler EOS reachability canary requires eval.sample_token_selection='argmax'")
        if not config.eval.sample_trace:
            raise ValueError("sampler EOS reachability canary requires eval.sample_trace=true")
        if config.eval.length_policy != "none":
            raise ValueError("sampler EOS reachability canary requires eval.length_policy='none'")
        if config.eval.fhs_tau >= 1.0:
            raise ValueError("sampler EOS reachability canary requires eval.fhs_tau < 1")
        if config.eval.gen_len < 4:
            raise ValueError("sampler EOS reachability canary requires eval.gen_len >= 4")


def validate_eos_overfit(config: ExperimentConfig) -> None:
    if config.train.strict_mdlm_eos_overfit_canary_only:
        if config.train.eval_only:
            raise ValueError("EOS overfit canary requires train.eval_only=false")
        if config.data.synthetic_dataset != "eos_overfit_canary":
            raise ValueError("EOS overfit canary requires data.synthetic_dataset='eos_overfit_canary'")
        if config.data.train_data:
            raise ValueError("EOS overfit canary uses only the synthetic in-memory dataset")
        if config.data.val_data:
            raise ValueError("EOS overfit canary uses the synthetic in-memory validation batch")
        if config.data.tokenizer != "byte":
            raise ValueError("EOS overfit canary requires data.tokenizer='byte'")
        if not config.data.append_eos:
            raise ValueError("EOS overfit canary requires data.append_eos=true")
        if not config.data.loss_on_eos:
            raise ValueError("EOS overfit canary requires data.loss_on_eos=true")
        if config.data.random_length_ratio != 0.0:
            raise ValueError("EOS overfit canary requires data.random_length_ratio=0")
        if config.data.continuation_ratio != 0.0:
            raise ValueError("EOS overfit canary requires data.continuation_ratio=0")
        if config.data.boundary_aware_packing:
            raise ValueError("EOS overfit canary requires data.boundary_aware_packing=false")
        if config.data.content_budget_segmented_packing:
            raise ValueError("EOS overfit canary requires data.content_budget_segmented_packing=false")
        if config.data.short_document_curriculum:
            raise ValueError("EOS overfit canary requires data.short_document_curriculum=false")
        if config.model.causal:
            raise ValueError("EOS overfit canary requires model.causal=false")
        if config.model.output_factorization != "flat":
            raise ValueError("EOS overfit canary requires model.output_factorization='flat'")
        if config.model.auxiliary_head_layers:
            raise ValueError("EOS overfit canary cannot be coupled with auxiliary heads")
        if config.model.bd3lm_block_size != 0:
            raise ValueError("EOS overfit canary cannot be coupled with BD3LM")
        if config.model.block_causal_pilot_size != 0:
            raise ValueError("EOS overfit canary cannot be coupled with block-causal pilot training")
        if config.model.length_head:
            raise ValueError("EOS overfit canary cannot be coupled with length-head training")
        if config.model.target_length_conditioning:
            raise ValueError("EOS overfit canary cannot be coupled with target-length conditioning")
        if config.noise.type != "mdlm":
            raise ValueError("EOS overfit canary requires noise.type='mdlm'")
        if not config.noise.schedule_weight and not config.train.strict_mdlm_uniform_schedule_weight:
            raise ValueError("EOS overfit canary requires noise.schedule_weight=true")
        if config.train.strict_mdlm_uniform_schedule_weight and config.noise.schedule_weight:
            raise ValueError("strict_mdlm_uniform_schedule_weight requires noise.schedule_weight=false")
        if config.noise.complement_masking:
            raise ValueError("EOS overfit canary requires noise.complement_masking=false")
        if config.noise.suffix_masking:
            raise ValueError("EOS overfit canary requires noise.suffix_masking=false")
        if config.noise.time_conditioning:
            raise ValueError("EOS overfit canary preserves no time conditioning")
        if config.noise.per_token_noise_levels:
            raise ValueError("EOS overfit canary requires noise.per_token_noise_levels=false")
        if config.train.auxiliary_loss_weight != 0:
            raise ValueError("EOS overfit canary requires auxiliary_loss_weight=0")
        if config.train.entropy_bonus_beta != 0:
            raise ValueError("EOS overfit canary requires entropy_bonus_beta=0")
        if config.train.confidence_penalty_weight != 0:
            raise ValueError("EOS overfit canary requires confidence_penalty_weight=0")
        if config.train.z_loss_weight != 0:
            raise ValueError("EOS overfit canary requires z_loss_weight=0")
        if config.train.eos_hazard_loss_weight != 0:
            raise ValueError("EOS overfit canary requires eos_hazard_loss_weight=0")
        if config.train.length_head_loss_weight != 0:
            raise ValueError("EOS overfit canary requires length_head_loss_weight=0")
        if config.train.position_decile_balanced_loss:
            raise ValueError("EOS overfit canary requires position_decile_balanced_loss=false")
        if config.train.uniform_loss_estimator != "single_sample":
            raise ValueError("EOS overfit canary requires uniform_loss_estimator='single_sample'")
        if config.train.token_edit_corruption_batch_fraction != 0:
            raise ValueError("EOS overfit canary requires token_edit_corruption_batch_fraction=0")
        if config.train.token_edit_corruption_ratio != 0:
            raise ValueError("EOS overfit canary requires token_edit_corruption_ratio=0")
        if (
            config.train.strict_mdlm_all_token_diagnostic_telemetry
            and config.data.synthetic_eos_overfit_loss_mask_policy != "all_lossable_tokens"
        ):
            raise ValueError(
                "all-token diagnostic telemetry requires "
                "data.synthetic_eos_overfit_loss_mask_policy='all_lossable_tokens'"
            )
        if config.train.strict_mdlm_teacher_forced_replay_diagnostic:
            if config.data.synthetic_eos_overfit_loss_mask_policy != "all_lossable_tokens":
                raise ValueError(
                    "teacher-forced replay diagnostic requires "
                    "data.synthetic_eos_overfit_loss_mask_policy='all_lossable_tokens'"
                )
            if not config.train.strict_mdlm_all_token_diagnostic_telemetry:
                raise ValueError(
                    "teacher-forced replay diagnostic requires strict_mdlm_all_token_diagnostic_telemetry=true"
                )
            if abs(float(config.noise.t_min) - 0.95) > 1e-8 or abs(float(config.noise.t_max) - 0.95) > 1e-8:
                raise ValueError("teacher-forced replay diagnostic requires fixed train noise t_min=t_max=0.95")
            replay_t_values = [float(value) for value in config.train.strict_mdlm_all_token_diagnostic_t_values]
            if replay_t_values and (len(replay_t_values) != 1 or abs(replay_t_values[0] - 0.95) > 1e-8):
                raise ValueError(
                    "teacher-forced replay diagnostic requires strict_mdlm_all_token_diagnostic_t_values=[0.95]"
                )
            if config.train.strict_mdlm_t_curriculum_enabled:
                raise ValueError("teacher-forced replay diagnostic cannot be coupled with strict MDLM t curriculum")
            if config.train.strict_mdlm_eos_loss_mass_balance:
                raise ValueError("teacher-forced replay diagnostic cannot be coupled with EOS loss-mass balancing")
            if config.train.comparison_run_ids != ["r-0014", "r-0015", "r-0017", "r-0020"]:
                raise ValueError(
                    "teacher-forced replay diagnostic requires "
                    "comparison_run_ids=['r-0014', 'r-0015', 'r-0017', 'r-0020']"
                )
            if config.eval.sampler != "default":
                raise ValueError("teacher-forced replay diagnostic requires eval.sampler='default'")
            if config.eval.sample_token_selection != "argmax":
                raise ValueError("teacher-forced replay diagnostic requires eval.sample_token_selection='argmax'")
            if config.eval.sample_trace:
                raise ValueError("teacher-forced replay diagnostic requires eval.sample_trace=false")
            if config.eval.length_policy != "none":
                raise ValueError("teacher-forced replay diagnostic requires eval.length_policy='none'")
    elif (
        config.train.strict_mdlm_all_token_diagnostic_telemetry
        and not config.train.strict_mdlm_all_token_bayes_audit_only
    ):
        raise ValueError(
            "all-token diagnostic telemetry requires strict_mdlm_eos_overfit_canary_only=true "
            "or strict_mdlm_all_token_bayes_audit_only=true"
        )
    elif config.train.strict_mdlm_teacher_forced_replay_diagnostic:
        raise ValueError("teacher-forced replay diagnostic requires strict_mdlm_eos_overfit_canary_only=true")
