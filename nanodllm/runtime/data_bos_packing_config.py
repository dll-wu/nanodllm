"""Admission for train-only BOS/packing attribution; legacy defaults are inert."""

DEFAULTS = {"bos_packing_mode": "none", "bos_packing_buffer_size": 1000}
MODES = ("none", "continuation_bos", "bestfit_segmented", "bestfit_cross_document")


def validation_data_policy(data: dict) -> dict:
    """Compare frozen-row semantics, not train packing or inventory transport.

    Older frozen input metadata lacks the opt-in defaults. Never mutate that
    evidence or ignore tokenizer, source, loss, or validation-layout changes.
    Snapshot pins govern data loading and are checked there; they do not alter
    already-frozen validation tensors, whose own hashes remain mandatory.
    """
    from .data_mask_tail_config import with_mask_tail_defaults

    result = {**with_mask_tail_defaults(data), **DEFAULTS}
    result.pop("dataset_snapshot", None)
    result.pop("dataset_snapshot_sha256", None)
    return result


def strip_disabled_bos_packing(value: dict) -> None:
    for key, default in DEFAULTS.items():
        if value["data"].pop(key, default) != default:
            raise ValueError("historical configuration requires disabled BOS packing")


def validate_bos_packing(config) -> None:
    data, model, noise, train = config.data, config.model, config.noise, config.train
    if data.bos_packing_mode not in MODES:
        raise ValueError("unsupported data.bos_packing_mode")
    size = data.bos_packing_buffer_size
    if type(size) is not int or size < 1:
        raise ValueError("data.bos_packing_buffer_size must be a positive integer")
    if data.bos_packing_mode == "none":
        if size != DEFAULTS["bos_packing_buffer_size"]:
            raise ValueError("disabled BOS packing requires the default buffer size")
        return
    # The causal-clean block objective consumes the same segment-local
    # positions/visibility as full MDLM. Admit isolated best-fit only; allowing
    # cross-document attention here would change the block conditioning task.
    block = (
        data.bos_packing_mode == "bestfit_segmented"
        and noise.type == "bd3lm"
        and model.bd3lm_block_size > 0
        and model.bd3lm_clean_stream_mode == "causal"
        and not model.causal
        and not train.bd3lm_uniform_t2t_batch_fraction
        and train.bd3lm_recipe_mechanism == "none"
    )
    if not (
        data.content_budget_segmented_packing
        and data.prepend_bos
        and data.append_eos
        and data.loss_on_eos
        and not data.loss_on_bos
        and not data.boundary_aware_packing
        and not data.short_document_curriculum
        and not data.synthetic_dataset
        and data.eos_mask_tail_mode == "none"
        and model.max_seq_len >= 2
        and model.output_factorization == "flat"
        and not model.block_causal_pilot_size
        and (not model.bd3lm_block_size or block)
        and model.mtp_mode == "none"
        and model.recurrent_iterations == 1
        and not model.auxiliary_head_layers
        and not model.length_head
        and not model.target_length_conditioning
        and (noise.type == "mdlm" or block)
        and not noise.time_conditioning
        and not noise.suffix_masking
        and not noise.soft_input_curriculum
        and not noise.per_token_noise_levels
        and train.mdlm_auxiliary.mode == "none"
        and not train.token_edit_corruption_batch_fraction
        and not train.bd3lm_causal_ar_loss_weight
        and not train.auxiliary_loss_weight
        and not train.eos_hazard_loss_weight
        and not train.length_head_loss_weight
    ):
        raise ValueError(
            "BOS packing requires plain segmented AR/MDLM or isolated causal-clean blocks with protected BOS and supervised EOS"
        )
