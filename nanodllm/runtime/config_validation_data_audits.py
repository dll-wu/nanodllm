"""Checks for diagnostics that construct, corrupt, and score held-out batches."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import ExperimentConfig


def validate_collator_audit(config: ExperimentConfig) -> None:
    if config.train.strict_mdlm_real_collator_micro_overfit_canary_only:
        if config.train.eval_only:
            raise ValueError("real-collator micro-overfit canary requires train.eval_only=false")
        if config.data.synthetic_dataset:
            raise ValueError("real-collator micro-overfit canary requires real train_data, not a synthetic dataset")
        if not config.data.train_data:
            raise ValueError("real-collator micro-overfit canary requires data.train_data")
        if not config.data.append_eos:
            raise ValueError("real-collator micro-overfit canary requires data.append_eos=true")
        if not config.data.loss_on_eos:
            raise ValueError("real-collator micro-overfit canary requires data.loss_on_eos=true")
        if config.data.boundary_aware_packing:
            raise ValueError(
                "real-collator micro-overfit canary supports the default stream packer or "
                "content-budget segmented packer, not boundary-aware packing"
            )
        if config.data.short_document_curriculum:
            raise ValueError(
                "real-collator micro-overfit canary supports the default stream packer or "
                "content-budget segmented packer, not the short-document curriculum"
            )
        if config.train.strict_mdlm_real_collator_component_probe:
            if not config.data.content_budget_segmented_packing:
                raise ValueError("real-collator component probe requires data.content_budget_segmented_packing=true")
            if config.train.device_batch_size != 1:
                raise ValueError("real-collator component probe requires train.device_batch_size=1")
        if config.model.causal:
            raise ValueError("real-collator micro-overfit canary requires model.causal=false")
        if config.model.output_factorization != "flat":
            raise ValueError("real-collator micro-overfit canary requires model.output_factorization='flat'")
        if config.model.auxiliary_head_layers:
            raise ValueError("real-collator micro-overfit canary cannot be coupled with auxiliary heads")
        if config.model.bd3lm_block_size != 0:
            raise ValueError("real-collator micro-overfit canary cannot be coupled with BD3LM")
        if config.model.block_causal_pilot_size != 0:
            raise ValueError("real-collator micro-overfit canary cannot be coupled with block-causal pilot training")
        if config.model.length_head:
            raise ValueError("real-collator micro-overfit canary cannot be coupled with length-head training")
        if config.model.target_length_conditioning:
            raise ValueError("real-collator micro-overfit canary cannot be coupled with target-length conditioning")
        if config.noise.type != "mdlm":
            raise ValueError("real-collator micro-overfit canary requires noise.type='mdlm'")
        if not config.noise.schedule_weight:
            raise ValueError("real-collator micro-overfit canary requires noise.schedule_weight=true")
        if config.noise.complement_masking:
            raise ValueError("real-collator micro-overfit canary requires noise.complement_masking=false")
        if config.noise.suffix_masking:
            raise ValueError("real-collator micro-overfit canary requires noise.suffix_masking=false")
        if config.noise.time_conditioning:
            raise ValueError("real-collator micro-overfit canary preserves no time conditioning")
        if config.noise.per_token_noise_levels:
            raise ValueError("real-collator micro-overfit canary requires noise.per_token_noise_levels=false")
        if config.train.auxiliary_loss_weight != 0:
            raise ValueError("real-collator micro-overfit canary requires auxiliary_loss_weight=0")
        if config.train.entropy_bonus_beta != 0:
            raise ValueError("real-collator micro-overfit canary requires entropy_bonus_beta=0")
        if config.train.confidence_penalty_weight != 0:
            raise ValueError("real-collator micro-overfit canary requires confidence_penalty_weight=0")
        if config.train.z_loss_weight != 0:
            raise ValueError("real-collator micro-overfit canary requires z_loss_weight=0")
        if config.train.eos_hazard_loss_weight != 0:
            raise ValueError("real-collator micro-overfit canary requires eos_hazard_loss_weight=0")
        if config.train.length_head_loss_weight != 0:
            raise ValueError("real-collator micro-overfit canary requires length_head_loss_weight=0")
        if config.train.position_decile_balanced_loss:
            raise ValueError("real-collator micro-overfit canary requires position_decile_balanced_loss=false")
        if config.train.uniform_loss_estimator != "single_sample":
            raise ValueError("real-collator micro-overfit canary requires uniform_loss_estimator='single_sample'")
        if config.train.token_edit_corruption_batch_fraction != 0:
            raise ValueError("real-collator micro-overfit canary requires token_edit_corruption_batch_fraction=0")
        if config.train.token_edit_corruption_ratio != 0:
            raise ValueError("real-collator micro-overfit canary requires token_edit_corruption_ratio=0")


def validate_packing_audit(config: ExperimentConfig) -> None:
    if config.train.real_packing_audit_only:
        if config.train.eval_only:
            raise ValueError("real packing audit is a train-loader smoke and requires train.eval_only=false")
        if config.model.causal:
            raise ValueError("real packing audit requires model.causal=false")
        if config.model.output_factorization != "flat":
            raise ValueError("real packing audit requires model.output_factorization='flat'")
        if config.model.auxiliary_head_layers:
            raise ValueError("real packing audit cannot be coupled with auxiliary heads")
        if config.model.bd3lm_block_size != 0:
            raise ValueError("real packing audit cannot be coupled with BD3LM")
        if config.model.block_causal_pilot_size != 0:
            raise ValueError("real packing audit cannot be coupled with block-causal pilot training")
        if config.model.length_head:
            raise ValueError("real packing audit cannot be coupled with length-head training")
        if config.model.target_length_conditioning:
            raise ValueError("real packing audit cannot be coupled with target-length conditioning")
        if config.data.boundary_aware_packing:
            raise ValueError(
                "real packing audit targets the default stream packer and requires boundary_aware_packing=false"
            )
        if config.data.content_budget_segmented_packing:
            raise ValueError(
                "real packing audit targets the default stream packer and requires content_budget_segmented_packing=false"
            )
        if config.data.short_document_curriculum:
            raise ValueError(
                "real packing audit targets the default stream packer and requires short_document_curriculum=false"
            )
        if not config.data.append_eos:
            raise ValueError("real packing audit requires data.append_eos=true")
        if not config.data.loss_on_eos:
            raise ValueError("real packing audit requires data.loss_on_eos=true")
        if config.noise.type != "mdlm":
            raise ValueError("real packing audit requires noise.type='mdlm'")
        if config.noise.complement_masking:
            raise ValueError("real packing audit requires complement_masking=false")
        if config.noise.suffix_masking:
            raise ValueError("real packing audit requires suffix_masking=false")
        if config.noise.time_conditioning:
            raise ValueError("real packing audit preserves no time conditioning")
        if config.noise.per_token_noise_levels:
            raise ValueError("real packing audit requires per_token_noise_levels=false")
        if config.train.auxiliary_loss_weight != 0:
            raise ValueError("real packing audit cannot be coupled with auxiliary loss")
        if config.train.entropy_bonus_beta != 0:
            raise ValueError("real packing audit cannot be coupled with entropy bonus")
        if config.train.confidence_penalty_weight != 0:
            raise ValueError("real packing audit cannot be coupled with confidence penalty")
        if config.train.z_loss_weight != 0:
            raise ValueError("real packing audit cannot be coupled with z-loss")
        if config.train.eos_hazard_loss_weight != 0:
            raise ValueError("real packing audit cannot be coupled with EOS hazard loss")
        if config.train.length_head_loss_weight != 0:
            raise ValueError("real packing audit cannot be coupled with length-head loss")
        if config.train.position_decile_balanced_loss:
            raise ValueError("real packing audit cannot be coupled with position decile balancing")
        if config.train.uniform_loss_estimator != "single_sample":
            raise ValueError("real packing audit cannot be coupled with uniform loss estimators")
        if config.train.token_edit_corruption_batch_fraction != 0 or config.train.token_edit_corruption_ratio != 0:
            raise ValueError("real packing audit cannot be coupled with token-edit corruption")


def validate_corruption_audit(config: ExperimentConfig) -> None:
    if config.train.strict_mdlm_corruption_audit_only:
        if config.train.eval_only:
            raise ValueError("strict MDLM corruption audit is a train-mode smoke and requires train.eval_only=false")
        if config.model.causal:
            raise ValueError("strict MDLM corruption audit requires model.causal=false")
        if config.model.output_factorization != "flat":
            raise ValueError("strict MDLM corruption audit requires model.output_factorization='flat'")
        if config.model.auxiliary_head_layers:
            raise ValueError("strict MDLM corruption audit cannot be coupled with auxiliary heads")
        if config.model.bd3lm_block_size != 0:
            raise ValueError("strict MDLM corruption audit cannot be coupled with BD3LM")
        if config.model.block_causal_pilot_size != 0:
            raise ValueError("strict MDLM corruption audit cannot be coupled with block-causal pilot training")
        if config.model.length_head:
            raise ValueError("strict MDLM corruption audit cannot be coupled with length-head training")
        if config.model.target_length_conditioning:
            raise ValueError("strict MDLM corruption audit cannot be coupled with target-length conditioning")
        if config.noise.type != "mdlm":
            raise ValueError("strict MDLM corruption audit requires noise.type='mdlm'")
        if config.noise.complement_masking:
            raise ValueError("strict MDLM corruption audit requires complement_masking=false")
        if config.noise.suffix_masking:
            raise ValueError("strict MDLM corruption audit requires suffix_masking=false")
        if config.noise.time_conditioning:
            raise ValueError("strict MDLM corruption audit preserves no time conditioning")
        if config.noise.per_token_noise_levels:
            raise ValueError("strict MDLM corruption audit requires per_token_noise_levels=false")
        if not config.noise.schedule_weight:
            raise ValueError("strict MDLM corruption audit requires schedule_weight=true")
        if config.train.auxiliary_loss_weight != 0:
            raise ValueError("strict MDLM corruption audit cannot be coupled with auxiliary loss")
        if config.train.entropy_bonus_beta != 0:
            raise ValueError("strict MDLM corruption audit cannot be coupled with entropy bonus")
        if config.train.confidence_penalty_weight != 0:
            raise ValueError("strict MDLM corruption audit cannot be coupled with confidence penalty")
        if config.train.z_loss_weight != 0:
            raise ValueError("strict MDLM corruption audit cannot be coupled with z-loss")
        if config.train.eos_hazard_loss_weight != 0:
            raise ValueError("strict MDLM corruption audit cannot be coupled with EOS hazard loss")
        if config.train.length_head_loss_weight != 0:
            raise ValueError("strict MDLM corruption audit cannot be coupled with length-head loss")
        if config.train.position_decile_balanced_loss:
            raise ValueError("strict MDLM corruption audit cannot be coupled with position decile balancing")
        if config.train.uniform_loss_estimator != "single_sample":
            raise ValueError("strict MDLM corruption audit cannot be coupled with uniform loss estimators")
        if config.train.token_edit_corruption_batch_fraction != 0 or config.train.token_edit_corruption_ratio != 0:
            raise ValueError("strict MDLM corruption audit cannot be coupled with token-edit corruption")


def validate_target_ce_audit(config: ExperimentConfig) -> None:
    if config.train.strict_mdlm_target_ce_audit_only:
        if config.train.eval_only:
            raise ValueError("strict MDLM target/CE audit is a train-mode smoke and requires train.eval_only=false")
        if config.model.causal:
            raise ValueError("strict MDLM target/CE audit requires model.causal=false")
        if config.model.output_factorization != "flat":
            raise ValueError("strict MDLM target/CE audit requires model.output_factorization='flat'")
        if config.model.auxiliary_head_layers:
            raise ValueError("strict MDLM target/CE audit cannot be coupled with auxiliary heads")
        if config.model.bd3lm_block_size != 0:
            raise ValueError("strict MDLM target/CE audit cannot be coupled with BD3LM")
        if config.model.block_causal_pilot_size != 0:
            raise ValueError("strict MDLM target/CE audit cannot be coupled with block-causal pilot training")
        if config.model.length_head:
            raise ValueError("strict MDLM target/CE audit cannot be coupled with length-head training")
        if config.model.target_length_conditioning:
            raise ValueError("strict MDLM target/CE audit cannot be coupled with target-length conditioning")
        if not config.data.append_eos:
            raise ValueError("strict MDLM target/CE audit requires data.append_eos=true")
        if not config.data.loss_on_eos:
            raise ValueError("strict MDLM target/CE audit requires data.loss_on_eos=true")
        if config.noise.type != "mdlm":
            raise ValueError("strict MDLM target/CE audit requires noise.type='mdlm'")
        if config.noise.complement_masking:
            raise ValueError("strict MDLM target/CE audit requires complement_masking=false")
        if config.noise.suffix_masking:
            raise ValueError("strict MDLM target/CE audit requires suffix_masking=false")
        if config.noise.time_conditioning:
            raise ValueError("strict MDLM target/CE audit preserves no time conditioning")
        if config.noise.per_token_noise_levels:
            raise ValueError("strict MDLM target/CE audit requires per_token_noise_levels=false")
        if not config.noise.schedule_weight:
            raise ValueError("strict MDLM target/CE audit requires schedule_weight=true")
        if config.train.auxiliary_loss_weight != 0:
            raise ValueError("strict MDLM target/CE audit cannot be coupled with auxiliary loss")
        if config.train.entropy_bonus_beta != 0:
            raise ValueError("strict MDLM target/CE audit cannot be coupled with entropy bonus")
        if config.train.confidence_penalty_weight != 0:
            raise ValueError("strict MDLM target/CE audit cannot be coupled with confidence penalty")
        if config.train.z_loss_weight != 0:
            raise ValueError("strict MDLM target/CE audit cannot be coupled with z-loss")
        if config.train.eos_hazard_loss_weight != 0:
            raise ValueError("strict MDLM target/CE audit cannot be coupled with EOS hazard loss")
        if config.train.length_head_loss_weight != 0:
            raise ValueError("strict MDLM target/CE audit cannot be coupled with length-head loss")
        if config.train.position_decile_balanced_loss:
            raise ValueError("strict MDLM target/CE audit cannot be coupled with position decile balancing")
        if config.train.uniform_loss_estimator != "single_sample":
            raise ValueError("strict MDLM target/CE audit cannot be coupled with uniform loss estimators")
        if config.train.token_edit_corruption_batch_fraction != 0 or config.train.token_edit_corruption_ratio != 0:
            raise ValueError("strict MDLM target/CE audit cannot be coupled with token-edit corruption")


def validate_right_context_audit(config: ExperimentConfig) -> None:
    if config.train.strict_mdlm_right_context_canary_only:
        if config.train.eval_only:
            raise ValueError(
                "strict MDLM right-context canary is a train-mode smoke and requires train.eval_only=false"
            )
        if config.model.causal:
            raise ValueError("strict MDLM right-context canary requires model.causal=false")
        if config.model.local_attention_lower_layers != 0 or config.model.local_attention_window != 0:
            raise ValueError("strict MDLM right-context canary requires global attention in all layers")
        if config.model.output_factorization != "flat":
            raise ValueError("strict MDLM right-context canary requires model.output_factorization='flat'")
        if config.model.auxiliary_head_layers:
            raise ValueError("strict MDLM right-context canary cannot be coupled with auxiliary heads")
        if config.model.bd3lm_block_size != 0:
            raise ValueError("strict MDLM right-context canary cannot be coupled with BD3LM")
        if config.model.block_causal_pilot_size != 0:
            raise ValueError("strict MDLM right-context canary cannot be coupled with block-causal pilot training")
        if config.model.length_head:
            raise ValueError("strict MDLM right-context canary cannot be coupled with length-head training")
        if config.model.target_length_conditioning:
            raise ValueError("strict MDLM right-context canary cannot be coupled with target-length conditioning")
        if config.noise.type != "mdlm":
            raise ValueError("strict MDLM right-context canary requires noise.type='mdlm'")
        if config.noise.complement_masking:
            raise ValueError("strict MDLM right-context canary requires noise.complement_masking=false")
        if config.noise.suffix_masking:
            raise ValueError("strict MDLM right-context canary requires noise.suffix_masking=false")
        if config.noise.time_conditioning:
            raise ValueError("strict MDLM right-context canary preserves no time conditioning")
        if config.noise.per_token_noise_levels:
            raise ValueError("strict MDLM right-context canary requires per_token_noise_levels=false")
        if not config.noise.schedule_weight:
            raise ValueError("strict MDLM right-context canary requires schedule_weight=true")
        if config.train.auxiliary_loss_weight != 0:
            raise ValueError("strict MDLM right-context canary cannot be coupled with auxiliary loss")
        if config.train.entropy_bonus_beta != 0:
            raise ValueError("strict MDLM right-context canary cannot be coupled with entropy bonus")
        if config.train.confidence_penalty_weight != 0:
            raise ValueError("strict MDLM right-context canary cannot be coupled with confidence penalty")
        if config.train.z_loss_weight != 0:
            raise ValueError("strict MDLM right-context canary cannot be coupled with z-loss")
        if config.train.eos_hazard_loss_weight != 0:
            raise ValueError("strict MDLM right-context canary cannot be coupled with EOS hazard loss")
        if config.train.length_head_loss_weight != 0:
            raise ValueError("strict MDLM right-context canary cannot be coupled with length-head loss")
        if config.train.position_decile_balanced_loss:
            raise ValueError("strict MDLM right-context canary cannot be coupled with position decile balancing")
        if config.train.uniform_loss_estimator != "single_sample":
            raise ValueError("strict MDLM right-context canary cannot be coupled with uniform loss estimators")
        if config.train.token_edit_corruption_batch_fraction != 0 or config.train.token_edit_corruption_ratio != 0:
            raise ValueError("strict MDLM right-context canary cannot be coupled with token-edit corruption")
