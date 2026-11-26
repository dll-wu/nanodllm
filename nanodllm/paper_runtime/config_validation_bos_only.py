"""Bounded EOS-free data admission; historical EOS-based routes stay unchanged."""


def validate_bos_only(config) -> None:
    if config.data.append_eos:
        return
    if not config.data.prepend_bos or config.data.loss_on_bos or config.data.loss_on_eos:
        raise ValueError("BOS-only MDLM requires prepend_bos=true, loss_on_bos=false and loss_on_eos=false")
    if config.data.synthetic_dataset or config.data.boundary_aware_packing or config.data.short_document_curriculum:
        raise ValueError("BOS-only MDLM requires ordinary stream or content-budget segmented packing")
    if (
        config.model.causal
        or config.noise.type != "mdlm"
        or config.noise.time_conditioning
        or config.noise.suffix_masking
        or config.noise.soft_input_curriculum
        or config.model.output_factorization != "flat"
        or config.model.block_causal_pilot_size
        or config.model.bd3lm_block_size
        or config.model.auxiliary_head_layers
        or config.model.length_head
        or config.model.target_length_conditioning
        or config.train.token_edit_corruption_batch_fraction
        or config.train.mdlm_auxiliary.mode != "none"
        or any(
            (
                config.train.auxiliary_loss_weight,
                config.train.entropy_bonus_beta,
                config.train.label_smoothing_epsilon,
                config.train.z_loss_weight,
                config.train.confidence_penalty_weight,
                config.train.eos_hazard_loss_weight,
                config.train.length_head_loss_weight,
            )
        )
    ):
        raise ValueError("BOS-only data currently supports plain, flat, time-agnostic MDLM without auxiliary losses")
