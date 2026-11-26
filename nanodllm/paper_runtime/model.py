from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import ar_width_mup, nanochat_backbone
from .model_accounting import PARAMETER_ACCOUNTING_VIEW_COMPONENTS as PARAMETER_ACCOUNTING_VIEW_COMPONENTS
from .model_accounting import PARAMETER_COMPONENT_NAMES as PARAMETER_COMPONENT_NAMES
from .model_accounting import PARAMETER_COMPONENT_PREFIXES as PARAMETER_COMPONENT_PREFIXES
from .model_accounting import ModelAccounting
from .model_attention import BidirectionalAttention, Linear, rms_norm
from .model_attention import apply_rotary as apply_rotary
from .model_attention import precompute_rotary as precompute_rotary
from .model_hierarchical import HierarchicalOutput
from .model_initialization import embedding_head_init_std as embedding_head_init_std
from .model_initialization import initialize_weights
from .model_initialization import residual_output_projection_init_std as residual_output_projection_init_std
from .model_length_output import LengthOutput
from .model_mtp import AuxiliaryMTP, mtp_output, validate_mtp_model, validate_mtp_request
from .model_parameterization import validate_parameterization_runtime
from .model_profiles import (
    NANOLLADA_PROFILE,
    ParameterFreeRMSNorm,
    SquaredReLUMLP,
    validate_nanollada_runtime,
)
from .model_recurrence import validate_recurrence_runtime
from .model_self_conditioning import BlockSelfConditioning, conditioned_embeddings
from .nanochat_backbone_config import PROFILE as NANOCHAT_PROFILE
from .training_bd3lm import bd3lm_attention_mask as bd3lm_attention_mask
from .training_bd3lm import duplicated_position_ids as duplicated_position_ids


@dataclass
class DiffusionConfig:
    sequence_len: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    mask_token_id: int
    mlp_hidden_dim: int = -1
    attention_scale: float = -1.0
    local_attention_lower_layers: int = 0
    local_attention_window: int = 0
    auxiliary_head_layers: tuple[int, ...] = ()
    time_conditioning: bool = False
    bd3lm_block_size: int = 0
    bd3lm_self_conditioning_dim: int = 0
    output_factorization: str = "flat"
    causal: bool = False
    residual_output_projection_init: str = "zero"
    embedding_head_init_std_multiplier: float = 1.0
    length_head: bool = False
    target_length_conditioning: bool = False
    mask_position_embedding: bool = False
    token_edit_adapter_bottleneck_dim: int = 0
    token_edit_adapter_top_layers: int = 0
    token_edit_provenance_embedding: bool = False
    token_edit_provenance_confidence_bins: int = 0
    parameterization_mode: str = "ordinary"
    parameterization_rule_set: str = ""
    parameterization_base_depth: int = 0
    parameterization_base_width: int = 0
    parameterization_mechanics_sha256: str = ""
    parameterization_hidden_init_std: float = -1.0
    parameterization_embedding_head_init_std: float = -1.0
    parameterization_residual_multiplier: float = 1.0
    parameterization_output_multiplier: float = 1.0
    parameterization_attention_scale: float = -1.0
    parameterization_attention_scale_rule: str = ""
    backbone_profile: str = "nanodllm"
    mtp_mode: str = "none"
    recurrent_core_start: int = 0
    recurrent_core_length: int = 0
    recurrent_iterations: int = 1


def resolved_mlp_hidden_dim(config: DiffusionConfig) -> int:
    hidden_dim = int(config.mlp_hidden_dim)
    return 4 * int(config.n_embd) if hidden_dim == -1 else hidden_dim


def resolved_attention_scale(config: DiffusionConfig) -> float:
    head_dim = int(config.n_embd) // int(config.n_head)
    if config.parameterization_mode != "ordinary":
        return float(config.parameterization_attention_scale)
    return (
        head_dim**-0.5
        if resolved_attention_scale_rule(config) == "pytorch_default_one_over_sqrt_head_dim"
        else float(config.attention_scale)
    )


def resolved_attention_scale_rule(config: DiffusionConfig) -> str:
    """Return the named rule that resolves the configured attention scale."""

    if config.parameterization_mode != "ordinary":
        return str(config.parameterization_attention_scale_rule)
    return "pytorch_default_one_over_sqrt_head_dim" if float(config.attention_scale) == -1.0 else "explicit"


class MLP(nn.Module):
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        hidden_dim = resolved_mlp_hidden_dim(config)
        self.gate = Linear(config.n_embd, hidden_dim, bias=False)
        self.up = Linear(config.n_embd, hidden_dim, bias=False)
        self.down = Linear(hidden_dim, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TokenEditResidualAdapter(nn.Module):
    """Zero-output-initialized nonlinear residual for a T2T-only path."""

    def __init__(self, n_embd: int, bottleneck_dim: int):
        super().__init__()
        self.down = Linear(n_embd, bottleneck_dim, bias=False)
        self.up = Linear(bottleneck_dim, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(F.silu(self.down(rms_norm(x))))


class DiffBlock(nn.Module):
    def __init__(self, config: DiffusionConfig, layer_idx: int):
        super().__init__()
        local_window = (
            int(config.local_attention_window) if layer_idx < int(config.local_attention_lower_layers) else None
        )
        self.attn = BidirectionalAttention(config, local_window=local_window, causal=config.causal)
        self.mlp = SquaredReLUMLP(config.n_embd) if config.backbone_profile == NANOLLADA_PROFILE else MLP(config)
        self.residual_multiplier = float(config.parameterization_residual_multiplier)
        self.norm_epsilon = ar_width_mup.NORM_EPSILON if ar_width_mup.runtime_active(config) else None

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attention_output = self.attn(rms_norm(x, epsilon=self.norm_epsilon), cos, sin, attention_mask=attention_mask)
        x = x + (attention_output if self.residual_multiplier == 1.0 else self.residual_multiplier * attention_output)
        mlp_output = self.mlp(rms_norm(x, epsilon=self.norm_epsilon))
        x = x + (mlp_output if self.residual_multiplier == 1.0 else self.residual_multiplier * mlp_output)
        return x


class DiffusionTransformer(ModelAccounting, HierarchicalOutput, LengthOutput, nn.Module):
    def __init__(
        self,
        config: DiffusionConfig,
        rope_base: float = 10000.0,
        compute_dtype: torch.dtype = torch.bfloat16,
        pad_vocab_to: int = 64,
        token_class_ids: torch.Tensor | None = None,
    ):
        super().__init__()
        self.config = config
        self.rope_base = rope_base
        self.compute_dtype = compute_dtype
        self.norm_epsilon = ar_width_mup.NORM_EPSILON if ar_width_mup.runtime_active(config) else None
        if ar_width_mup.runtime_active(config) and (
            rope_base != 10000 or compute_dtype not in (torch.bfloat16, torch.float32, torch.float64)
        ):
            raise ValueError("AR width muP requires RoPE10000 and BF16/FP32 or FP64 reference compute")
        validate_nanollada_runtime(config)
        validate_mtp_model(config)
        self.recurrence = validate_recurrence_runtime(config, mtp_enabled=config.mtp_mode != "none")
        if config.backbone_profile == NANOCHAT_PROFILE and rope_base != 100000:
            raise ValueError("nanochat_block_v1 requires RoPE100000")
        if config.residual_output_projection_init not in ("zero", "scaled_normal"):
            raise ValueError(
                "residual_output_projection_init must be 'zero' or 'scaled_normal', "
                f"got {config.residual_output_projection_init!r}"
            )
        if config.embedding_head_init_std_multiplier <= 0:
            raise ValueError(
                "embedding_head_init_std_multiplier must be positive, "
                f"got {config.embedding_head_init_std_multiplier!r}"
            )
        if resolved_mlp_hidden_dim(config) <= 0:
            raise ValueError(f"mlp_hidden_dim must be positive or -1, got {config.mlp_hidden_dim!r}")
        if float(config.attention_scale) != -1.0 and float(config.attention_scale) <= 0:
            raise ValueError(f"attention_scale must be positive or -1, got {config.attention_scale!r}")
        if int(config.token_edit_adapter_bottleneck_dim) < 0:
            raise ValueError("token_edit_adapter_bottleneck_dim must be non-negative")
        if int(config.token_edit_adapter_top_layers) < 0 or int(config.token_edit_adapter_top_layers) > int(
            config.n_layer
        ):
            raise ValueError("token_edit_adapter_top_layers must be between zero and n_layer")
        if (int(config.token_edit_adapter_bottleneck_dim) > 0) != (int(config.token_edit_adapter_top_layers) > 0):
            raise ValueError("token-edit adapter bottleneck and top-layer count must be enabled together")
        validate_parameterization_runtime(config)
        self.padded_vocab_size = ((config.vocab_size + pad_vocab_to - 1) // pad_vocab_to) * pad_vocab_to
        self.auxiliary_head_layers = tuple(config.auxiliary_head_layers)
        self._auxiliary_head_index = {layer_idx: i for i, layer_idx in enumerate(self.auxiliary_head_layers)}
        self.output_factorization = config.output_factorization
        if self.output_factorization not in ("flat", "hierarchical_frequency_decile"):
            raise ValueError(f"unknown output factorization: {self.output_factorization!r}")
        if self.output_factorization == "hierarchical_frequency_decile":
            if token_class_ids is None:
                raise ValueError("hierarchical_frequency_decile requires token_class_ids")
            token_class_ids = token_class_ids.detach().to(dtype=torch.long, device="cpu").reshape(-1)
            if token_class_ids.numel() != int(config.vocab_size):
                raise ValueError(
                    f"token_class_ids must have one entry per vocab item, got {token_class_ids.numel()} "
                    f"for vocab_size={config.vocab_size}"
                )
            if int(token_class_ids.min().item()) < 0:
                raise ValueError("token_class_ids entries must be non-negative")
            self.output_class_count = int(token_class_ids.max().item()) + 1
            if self.output_class_count < 2:
                raise ValueError("hierarchical output requires at least two token classes")
            token_local_ids = torch.empty_like(token_class_ids)
            class_token_counts = torch.zeros(self.output_class_count, dtype=torch.long)
            for class_idx in range(self.output_class_count):
                ids = (token_class_ids == class_idx).nonzero(as_tuple=True)[0]
                if ids.numel() == 0:
                    raise ValueError(f"hierarchical output class {class_idx} has no tokens")
                token_local_ids[ids] = torch.arange(ids.numel(), dtype=torch.long)
                class_token_counts[class_idx] = ids.numel()
            self.register_buffer("token_class_ids", token_class_ids, persistent=True)
            self.register_buffer("token_local_ids", token_local_ids, persistent=True)
            self.register_buffer("class_token_counts", class_token_counts, persistent=True)
        else:
            self.output_class_count = 0
            self.register_buffer("token_class_ids", torch.empty(0, dtype=torch.long), persistent=False)
            self.register_buffer("token_local_ids", torch.empty(0, dtype=torch.long), persistent=False)
            self.register_buffer("class_token_counts", torch.empty(0, dtype=torch.long), persistent=False)
        self.wte = nn.Embedding(self.padded_vocab_size, config.n_embd)
        block_type = nanochat_backbone.Block if config.backbone_profile == NANOCHAT_PROFILE else DiffBlock
        self.blocks = nn.ModuleList([block_type(config, layer_idx=i) for i in range(config.n_layer)])
        self.nanochat_extras = (
            nanochat_backbone.Extras(config, self.padded_vocab_size)
            if config.backbone_profile == NANOCHAT_PROFILE
            else None
        )
        self.final_norm = (
            ParameterFreeRMSNorm(epsilon=self.norm_epsilon)
            if config.backbone_profile in {NANOLLADA_PROFILE, NANOCHAT_PROFILE}
            else nn.LayerNorm(config.n_embd)
        )
        self.lm_head = Linear(config.n_embd, self.padded_vocab_size, bias=False)
        self.class_head = (
            Linear(config.n_embd, self.output_class_count, bias=False)
            if self.output_factorization == "hierarchical_frequency_decile"
            else None
        )
        self.length_head = Linear(config.n_embd, config.sequence_len, bias=True) if config.length_head else None
        self.target_length_embedding = (
            nn.Embedding(config.sequence_len, config.n_embd) if config.target_length_conditioning else None
        )
        self.target_distance_proj = Linear(3, config.n_embd, bias=False) if config.target_length_conditioning else None
        self.mask_position_embedding = (
            nn.Embedding(config.sequence_len, config.n_embd) if config.mask_position_embedding else None
        )
        self.token_edit_adapters = nn.ModuleList(
            TokenEditResidualAdapter(config.n_embd, int(config.token_edit_adapter_bottleneck_dim))
            for _ in range(int(config.token_edit_adapter_top_layers))
        )
        provenance_states = 3 + int(config.token_edit_provenance_confidence_bins)
        self.token_edit_provenance_embedding = (
            nn.Embedding(provenance_states, config.n_embd) if config.token_edit_provenance_embedding else None
        )
        self.intermediate_heads = nn.ModuleList(
            [Linear(config.n_embd, self.padded_vocab_size, bias=False) for _ in self.auxiliary_head_layers]
        )
        self.noise_mlp = (
            nn.Sequential(
                Linear(1, config.n_embd, bias=False),
                nn.SiLU(),
                Linear(config.n_embd, config.n_embd, bias=False),
            )
            if config.time_conditioning
            else None
        )

        head_dim = config.n_embd // config.n_head
        cos, sin = precompute_rotary(config.sequence_len, head_dim, rope_base, compute_dtype)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.init_weights()
        if config.bd3lm_self_conditioning_dim < 0:
            raise ValueError("bd3lm_self_conditioning_dim must be nonnegative")
        if config.bd3lm_self_conditioning_dim and (
            not config.bd3lm_block_size
            or config.causal
            or config.output_factorization != "flat"
            or config.time_conditioning
            or config.parameterization_mode != "ordinary"
            or config.backbone_profile != "nanodllm"
        ):
            raise ValueError("self-conditioning requires an ordinary flat block-diffusion model")
        self.block_self_conditioning = (
            BlockSelfConditioning(config.n_embd, config.bd3lm_self_conditioning_dim)
            if config.bd3lm_self_conditioning_dim
            else None
        )
        self.block_self_conditioning_enabled = True
        # Construct after base initialization: enabling MTP keeps the seeded
        # trunk/head weights identical, and disabling it consumes no extra RNG.
        self.mtp = AuxiliaryMTP(config, DiffBlock(config, layer_idx=0)) if config.mtp_mode != "none" else None

    def init_weights(self):
        initialize_weights(self)

    def forward(
        self,
        z_t: torch.Tensor,
        logits_indices: torch.Tensor | None = None,
        return_aux_logits: bool = False,
        return_hierarchical_logits: bool = False,
        return_length_logits: bool = False,
        attention_mask: torch.Tensor | None = None,
        noise_level: torch.Tensor | float | None = None,
        position_ids: torch.Tensor | None = None,
        input_embeddings: torch.Tensor | None = None,
        input_provenance: torch.Tensor | None = None,
        target_lengths: torch.Tensor | None = None,
        detach_final_hidden: bool = False,
        use_token_edit_adapter: bool = False,
        return_mtp_logits: bool = False,
        return_mtp_hidden: bool = False,
        recurrent_iterations: int | None = None,
    ):
        block_indices = self.recurrence.indices_for(recurrent_iterations)
        if return_mtp_logits or return_mtp_hidden:
            validate_mtp_request(
                self,
                z_t,
                attention_mask,
                return_mtp_logits,
                return_mtp_hidden,
                logits_indices,
                (
                    return_aux_logits,
                    return_hierarchical_logits,
                    return_length_logits,
                    input_embeddings,
                    input_provenance,
                    target_lengths,
                    noise_level,
                    detach_final_hidden,
                    use_token_edit_adapter,
                ),
            )
        if detach_final_hidden and use_token_edit_adapter:
            raise ValueError("detach_final_hidden and use_token_edit_adapter are mutually exclusive")
        if (detach_final_hidden or use_token_edit_adapter) and (
            return_aux_logits or return_hierarchical_logits or return_length_logits
        ):
            raise ValueError("isolated token-edit routes are supported only for the flat token-output path")
        if use_token_edit_adapter and not self.token_edit_adapters:
            raise ValueError("use_token_edit_adapter requires an enabled token-edit adapter")
        b, t = z_t.shape
        if position_ids is None:
            assert t <= self.config.sequence_len
            cos = self.rope_cos[:t]
            sin = self.rope_sin[:t]
        else:
            position_ids = position_ids.to(device=z_t.device, dtype=torch.long)
            if position_ids.dim() == 1:
                if position_ids.numel() != t:
                    raise ValueError(f"position_ids must have length {t}, got {position_ids.numel()}")
                flat_positions = position_ids
                shape = (t, 1, -1)
            elif position_ids.dim() == 2:
                if position_ids.shape != (b, t):
                    raise ValueError(f"position_ids must have shape {(b, t)}, got {tuple(position_ids.shape)}")
                flat_positions = position_ids.reshape(-1)
                shape = (b, t, 1, -1)
            else:
                raise ValueError("position_ids must be a 1D or 2D tensor")
            if torch.compiler.is_compiling():
                # Keep range enforcement inside the compiled graph without a GPU-to-host
                # sync. Inductor treats negative gather indices as Python-style wraparound,
                # so map every invalid value to an unambiguously out-of-bounds index and
                # let index_select retain its normal on-device bounds check.
                valid_positions = (flat_positions >= 0) & (flat_positions < self.config.sequence_len)
                invalid_position = torch.full_like(flat_positions, self.config.sequence_len)
                flat_positions = torch.where(valid_positions, flat_positions, invalid_position)
            elif flat_positions.numel() and (
                int(flat_positions.min().item()) < 0 or int(flat_positions.max().item()) >= self.config.sequence_len
            ):
                raise ValueError("position_ids entries must be valid RoPE positions")
            cos = self.rope_cos.index_select(0, flat_positions).view(shape)
            sin = self.rope_sin.index_select(0, flat_positions).view(shape)
        if (
            self.block_self_conditioning is not None
            and self.block_self_conditioning_enabled
            and input_embeddings is None
        ):
            if (
                return_aux_logits
                or return_hierarchical_logits
                or return_length_logits
                or input_provenance is not None
                or target_lengths is not None
                or detach_final_hidden
                or use_token_edit_adapter
                or noise_level is not None
            ):
                raise ValueError("self-conditioning is isolated to the flat block-token forward")
            input_embeddings = conditioned_embeddings(
                self, z_t, attention_mask=attention_mask, position_ids=position_ids
            )
        if input_embeddings is None:
            x = self.wte(z_t).to(dtype=self.compute_dtype)
        else:
            expected_shape = (b, t, self.config.n_embd)
            if tuple(input_embeddings.shape) != expected_shape:
                raise ValueError(
                    f"input_embeddings must have shape {expected_shape}, got {tuple(input_embeddings.shape)}"
                )
            x = input_embeddings.to(device=z_t.device, dtype=self.compute_dtype)
        if self.config.backbone_profile == NANOLLADA_PROFILE:
            x = rms_norm(x, epsilon=self.norm_epsilon)
        if input_provenance is not None:
            if self.token_edit_provenance_embedding is None:
                raise ValueError("input_provenance requires an enabled token-edit provenance embedding")
            if input_provenance.dtype != torch.long or tuple(input_provenance.shape) != (b, t):
                raise ValueError(f"input_provenance must be int64 with shape {(b, t)}")
            input_provenance = input_provenance.to(device=z_t.device)
            if torch.compiler.is_compiling():
                maximum_provenance = int(self.token_edit_provenance_embedding.num_embeddings) - 1
                valid_provenance = (input_provenance >= 0) & (input_provenance <= maximum_provenance)
                input_provenance = torch.where(
                    valid_provenance,
                    input_provenance,
                    torch.full_like(input_provenance, maximum_provenance + 1),
                )
            elif input_provenance.numel() and (
                int(input_provenance.min().item()) < 0
                or int(input_provenance.max().item()) >= int(self.token_edit_provenance_embedding.num_embeddings)
            ):
                raise ValueError(
                    "input_provenance entries must be in [0, "
                    f"{int(self.token_edit_provenance_embedding.num_embeddings) - 1}]"
                )
            x = x + self.token_edit_provenance_embedding(input_provenance).to(dtype=self.compute_dtype)
        if self.mask_position_embedding is not None:
            if position_ids is None:
                mask_positions = torch.arange(t, device=z_t.device, dtype=torch.long).view(1, t).expand(b, -1)
            elif position_ids.dim() == 1:
                mask_positions = position_ids.view(1, t).expand(b, -1)
            else:
                mask_positions = position_ids
            mask_position_values = self.mask_position_embedding(mask_positions).to(dtype=self.compute_dtype)
            mask_selector = (z_t == int(self.config.mask_token_id)).unsqueeze(-1).to(dtype=self.compute_dtype)
            x = x + mask_position_values * mask_selector
        if self.noise_mlp is not None:
            if noise_level is None:
                noise_level = (z_t == int(self.config.mask_token_id)).float().mean(dim=1)
            if not torch.is_tensor(noise_level):
                noise_level = torch.full((b,), float(noise_level), device=z_t.device)
            noise_level = noise_level.to(device=z_t.device, dtype=self.compute_dtype).reshape(-1)
            if noise_level.numel() == 1:
                noise_level = noise_level.expand(b)
            if noise_level.numel() != b:
                raise ValueError(f"noise_level must have one value or batch values, got {noise_level.numel()}")
            x = x + self.noise_mlp(noise_level[:, None])[:, None, :]
        target_conditioning = self._target_length_conditioning_embeddings(
            b,
            t,
            target_lengths,
            position_ids,
            z_t.device,
        )
        if target_conditioning is not None:
            x = x + target_conditioning
        aux_logits = [] if return_aux_logits else None
        adapter_start_layer = len(self.blocks) - len(self.token_edit_adapters)
        if self.nanochat_extras is not None:
            x = nanochat_backbone.hidden(self, z_t, x, cos, sin, attention_mask, position_ids)
        for layer_idx in block_indices if self.nanochat_extras is None else ():
            block = self.blocks[layer_idx]
            x = block(x, cos, sin, attention_mask=attention_mask)
            if use_token_edit_adapter and layer_idx >= adapter_start_layer:
                x = self.token_edit_adapters[layer_idx - adapter_start_layer](x)
            if return_aux_logits and layer_idx in self._auxiliary_head_index:
                aux_hidden = rms_norm(x)
                if logits_indices is not None:
                    aux_hidden = aux_hidden.reshape(b * t, -1).index_select(0, logits_indices)
                head_idx = self._auxiliary_head_index[layer_idx]
                aux_logits.append(self.intermediate_heads[head_idx](aux_hidden)[..., : self.config.vocab_size].float())
        mtp_hidden = x if return_mtp_logits or return_mtp_hidden else None
        x = self.final_norm(x)
        full_hidden = x
        length_logits = self._length_logits_from_hidden(full_hidden, attention_mask) if return_length_logits else None
        if detach_final_hidden:
            x = x.detach()
        if logits_indices is not None:
            x = x.reshape(b * t, -1).index_select(0, logits_indices)
        token_logits = self.lm_head(x)[..., : self.config.vocab_size]
        output_multiplier = float(self.config.parameterization_output_multiplier)
        if output_multiplier != 1.0:
            token_logits = token_logits * output_multiplier
        if not (ar_width_mup.runtime_active(self.config) and token_logits.dtype == torch.float64):
            token_logits = token_logits.float()
        if return_mtp_logits or return_mtp_hidden:
            return mtp_output(self, z_t, mtp_hidden, token_logits, cos, sin, attention_mask, return_mtp_hidden)
        if self.nanochat_extras is not None:
            token_logits = nanochat_backbone.softcap(token_logits)
        if self.uses_hierarchical_output():
            if self.class_head is None:
                raise RuntimeError("hierarchical output is enabled but class_head is missing")
            class_logits = self.class_head(x).float()
            if return_hierarchical_logits:
                if return_aux_logits:
                    if return_length_logits:
                        return class_logits, token_logits, aux_logits, length_logits
                    return class_logits, token_logits, aux_logits
                if return_length_logits:
                    return class_logits, token_logits, length_logits
                return class_logits, token_logits
            logits = self.hierarchical_token_log_probs(class_logits, token_logits)
        else:
            if return_hierarchical_logits:
                if return_aux_logits:
                    if return_length_logits:
                        return token_logits, None, aux_logits, length_logits
                    return token_logits, None, aux_logits
                if return_length_logits:
                    return token_logits, None, length_logits
                return token_logits, None
            logits = token_logits
        if return_aux_logits:
            if return_length_logits:
                return logits, aux_logits, length_logits
            return logits, aux_logits
        if return_length_logits:
            return logits, length_logits
        return logits


def estimate_transformer_params(n_layer: int, n_embd: int, mlp_hidden_dim: int = -1) -> int:
    """Compatibility helper backed by a meta-device model inventory."""
    counts = _standard_shape_parameter_counts(n_layer, n_embd, mlp_hidden_dim)
    return counts["transformer_blocks"]


def estimate_non_embedding_params(n_layer: int, n_embd: int, mlp_hidden_dim: int = -1) -> int:
    """Compatibility helper for the historical transformer-body view."""
    counts = _standard_shape_parameter_counts(n_layer, n_embd, mlp_hidden_dim)
    return counts["transformer_body_parameters"]


def _standard_shape_parameter_counts(n_layer: int, n_embd: int, mlp_hidden_dim: int = -1) -> dict[str, int]:
    """Instantiate a minimal meta model and return its inventory-derived counts."""
    if int(n_layer) <= 0 or int(n_embd) <= 0:
        raise ValueError("n_layer and n_embd must be positive")
    config = DiffusionConfig(
        sequence_len=1,
        vocab_size=1,
        n_layer=int(n_layer),
        n_head=1,
        n_embd=int(n_embd),
        mask_token_id=0,
        mlp_hidden_dim=int(mlp_hidden_dim),
    )
    with torch.device("meta"):
        model = DiffusionTransformer(config, compute_dtype=torch.float32, pad_vocab_to=1)
    return model.parameter_counts()


def estimate_parameter_counts(
    config: DiffusionConfig,
    *,
    pad_vocab_to: int = 64,
    output_class_count: int = 0,
) -> dict[str, int]:
    """Instantiate a meta-device model and read its named-parameter inventory."""
    if pad_vocab_to <= 0:
        raise ValueError("pad_vocab_to must be positive")
    token_class_ids = None
    if config.output_factorization == "hierarchical_frequency_decile":
        if int(output_class_count) < 2:
            raise ValueError("hierarchical output accounting requires output_class_count >= 2")
        token_class_ids = torch.arange(int(config.vocab_size), dtype=torch.long) % int(output_class_count)
    with torch.device("meta"):
        model = DiffusionTransformer(
            config,
            compute_dtype=torch.float32,
            pad_vocab_to=int(pad_vocab_to),
            token_class_ids=token_class_ids,
        )
    return model.parameter_counts()
