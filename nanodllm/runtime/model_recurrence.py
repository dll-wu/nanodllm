"""Fixed inner recurrence over existing, uniquely registered Transformer blocks.

Integration: add the three recurrent_* fields to both config dataclasses and
their resolver. Validate the experiment, then retain the runtime validator's
plan on the model. In the ordinary forward, iterate ``plan.block_indices`` and
call ``self.blocks[index](x, cos, sin, attention_mask=attention_mask)``. Prepare
embeddings and RoPE once and leave final normalization/readout after the loop.
Do not register another ModuleList, detach hidden states, or enter a cache path.
The validators check MTP mode; cache owners must pass their enabled predicate.

Accounting adds ``plan.extra_training_flops(...)`` to the existing estimate,
using one instantiated parameter count per unique block and the actual attention
length. Add ``plan.fingerprint()`` to the architecture payload before hashing
only when it is not None. Parameter inventories continue to count unique weights.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import ExperimentConfig, ModelConfig
    from .model import DiffusionConfig


def _integer(name: str, value: int, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class FixedInnerRecurrence:
    """A prefix, a contiguous shared core executed K times, and a suffix.

    ``n_layer`` is unique module depth; iterations include the core's first
    execution. A valid nonempty core with one iteration is an ordinary forward.
    An empty core requires the canonical disabled start=0, iterations=1 values.
    The core may touch either end of the stack (an empty prefix or suffix).
    """

    n_layer: int
    recurrent_core_start: int = 0
    recurrent_core_length: int = 0
    recurrent_iterations: int = 1
    block_indices: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        _integer("n_layer", self.n_layer, 1)
        _integer("recurrent_core_start", self.recurrent_core_start, 0)
        _integer("recurrent_core_length", self.recurrent_core_length, 0)
        _integer("recurrent_iterations", self.recurrent_iterations, 1)
        start = self.recurrent_core_start
        stop = start + self.recurrent_core_length
        if start >= self.n_layer or stop > self.n_layer:
            raise ValueError("recurrent core must lie within the unique block depth")
        if self.recurrent_core_length == 0 and (start != 0 or self.recurrent_iterations != 1):
            raise ValueError("an empty recurrent core requires recurrent_core_start=0 and recurrent_iterations=1")
        indices = tuple(range(start)) + tuple(range(start, stop)) * self.recurrent_iterations
        object.__setattr__(self, "block_indices", indices + tuple(range(stop, self.n_layer)))

    @property
    def enabled(self) -> bool:
        return self.recurrent_core_length > 0 and self.recurrent_iterations > 1

    @property
    def executed_blocks(self) -> int:
        return len(self.block_indices)

    @property
    def extra_block_executions(self) -> int:
        return self.executed_blocks - self.n_layer

    def indices_for(self, iterations: int | None = None) -> tuple[int, ...]:
        """Explicit per-forward depth, without changing the checkpoint's plan."""
        if iterations is None:
            return self.block_indices
        _integer("recurrent_iterations override", iterations, 1)
        if not self.enabled or iterations not in (1, 2, 3):
            raise ValueError("depth override requires an enabled shared core and R in {1,2,3}")
        start, stop = self.recurrent_core_start, self.recurrent_core_start + self.recurrent_core_length
        return tuple(range(start)) + tuple(range(start, stop)) * iterations + tuple(range(stop, self.n_layer))

    def extra_training_flops(self, block_parameter_counts: Sequence[int], *, n_embd: int, sequence_len: int) -> int:
        """Estimate *additional* forward/backward FLOPs per input token.

        Counts must describe each unique block's instantiated parameters, in
        module order. The ordinary NanoLLaDA blocks contain projection matrices
        and parameter-free norms. Each extra use contributes approximately 6P
        projection FLOPs and 12 * width * attention_length attention FLOPs per
        token. Embeddings, the final norm and the output head are not repeated.
        This follows the existing dense-attention estimate, including for causal
        masks; it is not an exact operation count or a measured kernel cost.
        """
        _integer("n_embd", n_embd, 1)
        _integer("sequence_len", sequence_len, 1)
        if len(block_parameter_counts) != self.n_layer:
            raise ValueError("block_parameter_counts must contain one count per unique block")
        for index, count in enumerate(block_parameter_counts):
            _integer(f"block_parameter_counts[{index}]", count, 0)
        start = self.recurrent_core_start
        stop = start + self.recurrent_core_length
        repeated_parameters = sum(block_parameter_counts[start:stop]) * (self.recurrent_iterations - 1)
        return 6 * repeated_parameters + 12 * n_embd * sequence_len * self.extra_block_executions

    def fingerprint(self) -> dict[str, object] | None:
        """Return enabled-only details; omit the entire entry for old hashes."""
        if not self.enabled:
            return None
        return {
            "method": "fixed_contiguous_shared_core_v1",
            "recurrent_core_start": self.recurrent_core_start,
            "recurrent_core_length": self.recurrent_core_length,
            "recurrent_iterations": self.recurrent_iterations,
            "unique_blocks": self.n_layer,
            "executed_blocks": self.executed_blocks,
            "block_execution_indices": list(self.block_indices),
            "weight_sharing": "reuse_same_block_modules",
            "gradient_flow": "full_unroll",
            "attention_and_positions": "unchanged_per_execution",
            "additional_parameters": 0,
            "extra_training_flops_per_token_method": "6x_repeated_block_parameters_plus_quadratic_attention",
        }


def _plan(config: ModelConfig | DiffusionConfig, n_layer: int) -> FixedInnerRecurrence:
    # Missing fields are the historical disabled configuration. Do not coerce
    # provided values: bools, floats and numeric strings must fail validation.
    return FixedInnerRecurrence(
        n_layer=n_layer,
        recurrent_core_start=getattr(config, "recurrent_core_start", 0),
        recurrent_core_length=getattr(config, "recurrent_core_length", 0),
        recurrent_iterations=getattr(config, "recurrent_iterations", 1),
    )


def _validate_model_scope(config: ModelConfig | DiffusionConfig, *, mtp_enabled: bool, cache_enabled: bool) -> None:
    from .model_profiles import NANOLLADA_PROFILE, validate_backbone_profile

    if config.backbone_profile != NANOLLADA_PROFILE:
        raise ValueError("fixed inner recurrence requires backbone_profile='nanollada_v1'")
    validate_backbone_profile(config)
    if config.bd3lm_block_size:
        raise ValueError("fixed inner recurrence does not support BD3 or prefix caches")
    if type(mtp_enabled) is not bool or type(cache_enabled) is not bool:
        raise ValueError("MTP and cache enabled predicates must be booleans")
    if getattr(config, "mtp_mode", "none") != "none" or mtp_enabled or cache_enabled:
        raise ValueError("fixed inner recurrence cannot be combined with MTP or caches")


def validate_recurrence_runtime(
    config: DiffusionConfig, *, mtp_enabled: bool = False, cache_enabled: bool = False
) -> FixedInnerRecurrence:
    """Validate direct model construction and return its immutable execution plan.

    Explicit predicates can reject cache or MTP routes beyond the model's MTP
    mode. Disabled plans impose no new architecture restrictions. Experiment-only
    restrictions need the validator below as well, before resolving model config.
    """
    plan = _plan(config, config.n_layer)
    if not plan.enabled:
        return plan
    _validate_model_scope(config, mtp_enabled=mtp_enabled, cache_enabled=cache_enabled)
    if config.time_conditioning:
        raise ValueError("fixed inner recurrence does not support time conditioning")
    ordinary_mechanics = {
        "parameterization_mode": "ordinary",
        "parameterization_rule_set": "",
        "parameterization_mechanics_sha256": "",
        "parameterization_hidden_init_std": -1.0,
        "parameterization_embedding_head_init_std": -1.0,
        "parameterization_residual_multiplier": 1.0,
        "parameterization_output_multiplier": 1.0,
        "parameterization_attention_scale": -1.0,
        "parameterization_attention_scale_rule": "",
    }
    for name, expected in ordinary_mechanics.items():
        if getattr(config, name) != expected:
            raise ValueError(f"fixed inner recurrence requires ordinary mechanics: {name}={expected!r}")
    return plan


def validate_recurrence_config(
    config: ExperimentConfig, *, mtp_enabled: bool = False, cache_enabled: bool = False
) -> FixedInnerRecurrence:
    """Admit only plain ordinary NanoLLaDA AR/MDLM for the first recurrence batch."""
    plan = _plan(config.model, config.model.depth)
    if not plan.enabled:
        return plan
    _validate_model_scope(config.model, mtp_enabled=mtp_enabled, cache_enabled=cache_enabled)
    if (
        config.parameterization.mode != "ordinary"
        or config.parameterization.execution_scope != "ordinary"
        or config.parameterization.hidden_init_std_multiplier != 1.0
    ):
        raise ValueError("fixed inner recurrence requires ordinary unscoped parameterization")
    # Required binding and a populated contract digest are provenance, not
    # nonordinary mechanics. Their owner validates/binds them, including resume.
    if config.optim.type != "adamw":
        raise ValueError("fixed inner recurrence requires ordinary NanoLLaDA AdamW")
    # In this repository AR uses model.causal=True with noise.type='mdlm'.
    if config.noise.type != "mdlm" or config.noise.time_conditioning or config.noise.soft_input_curriculum:
        raise ValueError("fixed inner recurrence requires unconditioned plain AR or MDLM")
    if config.model.block_causal_pilot_size or config.eval.sampler in {"bd3lm", "block_causal_pilot"}:
        raise ValueError("fixed inner recurrence does not support block objectives or cache routes")
    if config.train.mdlm_auxiliary.mode != "none":
        raise ValueError("fixed inner recurrence does not support MDLM auxiliary views")
    if getattr(config.train, "mtp_loss_weight", 0):
        raise ValueError("fixed inner recurrence cannot be combined with an MTP loss")
    for name in (
        "auxiliary_loss_weight",
        "label_smoothing_epsilon",
        "entropy_bonus_beta",
        "confidence_penalty_weight",
        "z_loss_weight",
        "eos_hazard_loss_weight",
        "length_head_loss_weight",
        "position_decile_balanced_loss",
        "token_edit_corruption_batch_fraction",
        "token_edit_prefix_enabled",
        "bd3lm_uniform_t2t_batch_fraction",
        "bd3lm_causal_ar_loss_weight",
        "bd3lm_full_mask_batch_fraction",
    ):
        if getattr(config.train, name):
            raise ValueError(f"fixed inner recurrence does not support train.{name}")
    if config.train.bd3lm_recipe_mechanism != "none":
        raise ValueError("fixed inner recurrence does not support block-recipe extensions")
    return plan
