"""Explicit segment-local block partitions, independent of corruption RNG."""

import random

import torch


def validate_partition_config(config) -> None:
    train = config.train
    mode, lengths = train.bd3lm_partition_mode, train.bd3lm_partition_lengths
    if mode not in {"fixed", "mixed", "irregular"}:
        raise ValueError("unknown bd3lm_partition_mode")
    if not isinstance(lengths, list) or any(type(n) is not int or n <= 0 for n in lengths):
        raise ValueError("bd3lm_partition_lengths must be a list of positive integers")
    if type(train.bd3lm_partition_seed) is not int or train.bd3lm_partition_seed < 0:
        raise ValueError("bd3lm_partition_seed must be a nonnegative integer")
    if mode == "fixed":
        if lengths or train.bd3lm_partition_seed:
            raise ValueError("fixed partitions require empty lengths and zero partition seed")
        return
    if (
        config.noise.type != "bd3lm"
        or config.model.bd3lm_clean_stream_mode != "causal"
        or train.bd3lm_block_size_schedule != "fixed"
        or train.bd3lm_recipe_mechanism != "none"
    ):
        raise ValueError("sampled partitions require plain causal-clean BD3 with no BL curriculum or recipe mechanism")
    if not lengths or lengths != sorted(set(lengths)) or lengths[-1] > config.model.max_seq_len:
        raise ValueError("partition lengths must be increasing, unique and within context")
    if mode == "irregular" and len(lengths) != 2:
        raise ValueError("irregular partition lengths specify inclusive [minimum, maximum]")


def partition_ids(positions: torch.Tensor, *, mode: str, lengths: list[int], seed: int, step: int) -> torch.Tensor:
    """One sampled layout per optimizer step, shared across rows/ranks/microbatches.

    Document-reset positions index the same layout from zero. The independent,
    step-keyed stream makes restart exact without changing corruption RNG state.
    Irregular blocks sample inclusive integer lengths; the context's last block
    may be clipped. No implicit regrouping of already generated tokens is used.
    """
    if positions.ndim != 2 or positions.dtype != torch.long or step < 0:
        raise ValueError("partition construction requires int64 [batch, sequence] positions and a nonnegative step")
    if mode not in {"mixed", "irregular"} or not lengths:
        raise ValueError("partition construction requires a sampled mode and lengths")
    rng = random.Random(f"nanodllm-block-partition-v1:{seed}:{step}")
    # Segment positions cannot exceed the source row length in this route.
    capacity = positions.shape[1]
    if bool(((positions < 0) | (positions >= capacity)).any()):
        raise ValueError("segment positions must lie within the source row")
    if mode == "mixed":
        width = rng.choice(lengths)
        table = [i // width for i in range(capacity)]
    else:
        table = []
        block = 0
        while len(table) < capacity:
            table.extend([block] * rng.randint(lengths[0], lengths[1]))
            block += 1
        table = table[:capacity]
    lookup = torch.tensor(table, dtype=torch.long, device=positions.device)
    return lookup[positions]


def training_partition_ids(config, tokens: torch.Tensor, positions: torch.Tensor | None, step: int):
    if config.train.bd3lm_partition_mode == "fixed":
        return None
    if positions is None:
        positions = torch.arange(tokens.shape[1], device=tokens.device).expand_as(tokens)
    return partition_ids(
        positions,
        mode=config.train.bd3lm_partition_mode,
        lengths=config.train.bd3lm_partition_lengths,
        seed=config.train.bd3lm_partition_seed,
        step=step,
    )


def checked_block_ids(block_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Validate external partition metadata; reset boundaries may restart IDs."""
    if block_ids.dtype != torch.long or block_ids.shape != positions.shape or block_ids.device != positions.device:
        raise ValueError("explicit block IDs must match position shape/device with int64 dtype")
    reset = positions[:, 1:] == 0
    delta = block_ids[:, 1:] - block_ids[:, :-1]
    if bool((block_ids < 0).any()) or bool(((delta < 0) & ~reset).any()):
        raise ValueError("block IDs must be nonnegative and nondecreasing inside each segment")
    return block_ids
