"""Opt-in visible EOS-tail packing and its disabled legacy projection."""

DEFAULTS = {"eos_mask_tail_mode": "none", "eos_mask_tail_size": 16}


def strip_disabled_mask_tail(value: dict) -> None:
    for name, default in DEFAULTS.items():
        if value["data"].pop(name, default) != default:
            raise ValueError("historical configuration requires disabled EOS MASK tails")


def with_mask_tail_defaults(data: dict) -> dict:
    """Only fill the two newly introduced fields; never rewrite old evidence."""
    return {**DEFAULTS, **data}


def validate_mask_tail(config) -> None:
    data, model = config.data, config.model
    if data.eos_mask_tail_mode not in {"none", "random", "aligned"}:
        raise ValueError("data.eos_mask_tail_mode must be none, random or aligned")
    size = data.eos_mask_tail_size
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("data.eos_mask_tail_size must be a positive integer")
    if data.eos_mask_tail_mode == "none":
        if size != DEFAULTS["eos_mask_tail_size"]:
            raise ValueError("disabled EOS MASK tails require the default size")
        return
    plain_mdlm = config.noise.type == "mdlm" and not model.bd3lm_block_size
    causal_block_m2t = (
        config.noise.type == "bd3lm"
        and model.bd3lm_block_size > 0
        and model.bd3lm_clean_stream_mode == "causal"
        and config.train.bd3lm_uniform_t2t_batch_fraction == 0
        and config.train.bd3lm_causal_ar_loss_weight == 0
        and config.train.bd3lm_recipe_mechanism == "none"
        and config.train.bd3lm_block_size_schedule == "fixed"
    )
    if not (
        data.content_budget_segmented_packing
        and data.append_eos
        and data.loss_on_eos
        and not data.loss_on_bos
        and not data.synthetic_dataset
        and not data.boundary_aware_packing
        and not data.short_document_curriculum
        and (plain_mdlm or causal_block_m2t)
        and not model.causal
        and not model.block_causal_pilot_size
        and model.output_factorization == "flat"
        and not config.train.token_edit_corruption_batch_fraction
        and config.train.mdlm_auxiliary.mode == "none"
    ):
        raise ValueError(
            "EOS MASK tails require plain segmented MDLM or causal-clean block M2T "
            "with supervised EOS and protected BOS"
        )
    if size + 2 > model.max_seq_len:
        raise ValueError("EOS MASK tail and a two-token terminal fragment must fit the context")
    if data.eos_mask_tail_mode == "aligned" and model.max_seq_len % size:
        raise ValueError("aligned EOS MASK tails require context divisible by the alignment size")
