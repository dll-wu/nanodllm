"""Stateless, compute-balanced inner-depth schedules for plain AR/MDLM.

Every update visits [1, 2, 2, 3] in a seed/update-keyed order. Scheduling uses
neither the data nor corruption RNG, and exact resume needs no mutable sampler
state. Only completed updates have the fixed-R2 compute budget; partial failed
attempts remain additional wall-clock work, not accepted-lineage exposure.
"""

import hashlib
import itertools
import os

import torch

SCHEDULE = "balanced_1_2_2_3"
ORDERS = tuple(sorted(set(itertools.permutations((1, 2, 2, 3)))))


def enabled(config):
    return config.train.recurrent_depth_schedule != "fixed"


def validate_config(config):
    from .model_recurrence import validate_recurrence_config

    plan = validate_recurrence_config(config, mtp_enabled=config.model.mtp_mode != "none")
    value = config.train.recurrent_depth_schedule
    if value not in ("fixed", SCHEDULE):
        raise ValueError("unsupported train.recurrent_depth_schedule")
    if value == "fixed":
        return
    if (plan.n_layer, plan.recurrent_core_start, plan.recurrent_core_length, plan.recurrent_iterations) != (4, 1, 2, 2):
        raise ValueError("variable depth requires the D4 shared middle core with default R2")
    micro = config.train.device_batch_size * config.model.max_seq_len
    if micro <= 0 or config.train.total_batch_size != 4 * micro:
        raise ValueError("variable depth requires explicit EB = 4 * DB * sequence length at world size 1")
    if config.runtime.init_single_rank_process_group:
        raise ValueError("variable depth requires single-process training")


def validate_process(config):
    if enabled(config) and (
        "RANK" in os.environ or int(os.environ.get("WORLD_SIZE", "1")) != 1 or torch.distributed.is_initialized()
    ):
        raise ValueError("variable depth requires single-process training")


def update_iterations(seed, step):
    """Zero-based optimizer update; modulo bias is negligible, balance is exact."""
    if type(seed) is not int or seed < 0 or type(step) is not int or step < 0:
        raise ValueError("depth schedule seed and step must be nonnegative integers")
    digest = hashlib.sha256(f"nanodllm-variable-depth-v1:{seed}:{step}".encode()).digest()
    return ORDERS[int.from_bytes(digest[:8], "big") % len(ORDERS)]


def training_seed(config):
    seed = config.runtime.training_seed
    return config.runtime.seed if seed < 0 else seed


def microbatch_kwargs(config, step, micro, accumulation):
    if not enabled(config):
        return {}
    if accumulation != 4 or type(micro) is not int or not 0 <= micro < 4:
        raise ValueError("balanced variable depth requires four microbatches per update")
    return {"recurrent_iterations": update_iterations(training_seed(config), step)[micro]}


def metrics(config, completed_steps):
    if not enabled(config):
        return {}
    if type(completed_steps) is not int or completed_steps < 0:
        raise ValueError("completed_steps must be a nonnegative integer")
    return {
        "recurrent_depth_schedule": SCHEDULE,
        "recurrent_depth_schedule_seed": training_seed(config),
        "recurrent_training_iterations": [1, 2, 3],
        "recurrent_microbatch_counts": {"1": completed_steps, "2": 2 * completed_steps, "3": completed_steps},
        "recurrent_mean_iterations": 2.0,
        "recurrent_mean_executed_blocks": 6.0,
        "recurrent_count_scope": "Scheduled full microbatches in completed updates; excludes rejected attempts and evaluation.",
    }
