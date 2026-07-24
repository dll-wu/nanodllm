import hashlib
import json
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

PARAMETER_COMPONENT_PREFIXES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("token_embedding", ("wte.",)),
    ("output_head", ("lm_head.",)),
    ("transformer_blocks", ("blocks.",)),
    ("final_norm", ("final_norm.",)),
    ("class_output_head", ("class_head.",)),
    ("length_head", ("length_head.",)),
    ("target_length_conditioning", ("target_length_embedding.", "target_distance_proj.")),
    ("mask_position_embedding", ("mask_position_embedding.",)),
    ("auxiliary_output_heads", ("intermediate_heads.",)),
    ("time_conditioning", ("noise_mlp.",)),
)
PARAMETER_COMPONENT_NAMES = tuple(component for component, _prefixes in PARAMETER_COMPONENT_PREFIXES) + (
    "other_parameters",
)
PARAMETER_ACCOUNTING_VIEW_COMPONENTS: dict[str, tuple[str, ...]] = {
    "total_parameters": PARAMETER_COMPONENT_NAMES,
    "transformer_body_parameters": tuple(
        component for component in PARAMETER_COMPONENT_NAMES if component not in {"token_embedding", "output_head"}
    ),
    "nanochat_scaling_parameters": ("transformer_blocks", "output_head"),
    "non_input_embedding_parameters": tuple(
        component for component in PARAMETER_COMPONENT_NAMES if component != "token_embedding"
    ),
}


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
    output_factorization: str = "flat"
    causal: bool = False
    residual_output_projection_init: str = "zero"
    embedding_head_init_std_multiplier: float = 1.0
    length_head: bool = False
    target_length_conditioning: bool = False
    mask_position_embedding: bool = False


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


def resolved_mlp_hidden_dim(config: DiffusionConfig) -> int:
    hidden_dim = int(config.mlp_hidden_dim)
    return 4 * int(config.n_embd) if hidden_dim == -1 else hidden_dim


def resolved_attention_scale(config: DiffusionConfig) -> float:
    head_dim = int(config.n_embd) // int(config.n_head)
    return head_dim**-0.5 if float(config.attention_scale) == -1.0 else float(config.attention_scale)


def precompute_rotary(seq_len: int, head_dim: int, base: float, dtype: torch.dtype, device=None):
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32, device=device) / half))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos().to(dtype)[:, None, :], freqs.sin().to(dtype)[:, None, :]


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat((x1 * cos + x2 * sin, -x1 * sin + x2 * cos), dim=-1)


def duplicated_position_ids(seq_len: int, device=None) -> torch.Tensor:
    """Return RoPE positions for an ``x_t || x_0`` concatenation."""
    positions = torch.arange(seq_len, dtype=torch.long, device=device)
    return torch.cat((positions, positions), dim=0)


def residual_output_projection_init_std(n_layer: int) -> float:
    return 0.02 / math.sqrt(2 * n_layer)


def embedding_head_init_std(n_embd: int, multiplier: float = 1.0) -> float:
    return 0.4 / math.sqrt(n_embd) * multiplier


def bd3lm_attention_mask(attention_mask: torch.Tensor, block_size: int) -> torch.Tensor:
    """Build the BD3LM block-diagonal / offset-causal / block-causal mask.

    ``attention_mask`` is the base clean-sequence key mask with shape ``[B, L]``.
    The returned mask has shape ``[B, 2L, 2L]`` and uses PyTorch SDPA semantics:
    true entries are visible query-key pairs.
    """
    if attention_mask.dim() != 2:
        raise ValueError(f"attention_mask must have shape [batch, seq], got {tuple(attention_mask.shape)}")
    if block_size <= 0:
        raise ValueError("bd3lm block_size must be positive")

    batch_size, seq_len = attention_mask.shape
    idx = torch.arange(seq_len * 2, device=attention_mask.device)
    q_idx = idx[:, None]
    kv_idx = idx[None, :]
    q_is_x0 = q_idx >= seq_len
    kv_is_x0 = kv_idx >= seq_len
    block_q = torch.where(q_is_x0, (q_idx - seq_len) // block_size, q_idx // block_size)
    block_kv = torch.where(kv_is_x0, (kv_idx - seq_len) // block_size, kv_idx // block_size)

    block_diagonal = (block_q == block_kv) & (q_is_x0 == kv_is_x0)
    offset_block_causal = (block_q > block_kv) & kv_is_x0 & ~q_is_x0
    block_causal = (block_q >= block_kv) & kv_is_x0 & q_is_x0
    mask = block_diagonal | offset_block_causal | block_causal

    valid_keys = torch.cat((attention_mask, attention_mask), dim=1).to(torch.bool)
    return mask.view(1, seq_len * 2, seq_len * 2).expand(batch_size, -1, -1) & valid_keys[:, None, :]


class Linear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype), self.bias)


class BidirectionalAttention(nn.Module):
    def __init__(self, config: DiffusionConfig, local_window: int | None = None, causal: bool = False):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.attention_scale = None if float(config.attention_scale) == -1.0 else float(config.attention_scale)
        self.local_window = local_window
        self.causal = causal
        self.c_q = Linear(config.n_embd, config.n_embd, bias=False)
        self.c_k = Linear(config.n_embd, config.n_embd, bias=False)
        self.c_v = Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj = Linear(config.n_embd, config.n_embd, bias=False)

    def _build_attention_mask(
        self,
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        attention_mask: torch.Tensor | None,
        query_indices: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Resolve model masks, optionally for only a subset of query rows."""
        key_positions = torch.arange(seq_len, device=device)
        query_positions = key_positions if query_indices is None else query_indices
        query_count = int(query_positions.numel())

        attn_mask = None
        if self.causal:
            attn_mask = query_positions[:, None] >= key_positions[None, :]
            attn_mask = attn_mask.view(1, 1, query_count, seq_len)
        elif self.local_window is not None:
            attn_mask = (query_positions[:, None] - key_positions[None, :]).abs() <= self.local_window
            attn_mask = attn_mask.view(1, 1, query_count, seq_len)

        if attention_mask is None:
            return attn_mask

        if attention_mask.dim() == 2 and attention_mask.shape == (batch_size, seq_len):
            external_mask = attention_mask[:, None, None, :].to(device=device, dtype=torch.bool)
        elif attention_mask.dim() == 2 and attention_mask.shape == (seq_len, seq_len):
            external_mask = attention_mask.to(device=device, dtype=torch.bool)
            if query_indices is not None:
                external_mask = external_mask.index_select(0, query_indices)
            external_mask = external_mask[None, None, :, :]
        elif attention_mask.dim() == 3 and attention_mask.shape[-2:] == (seq_len, seq_len):
            external_mask = attention_mask.to(device=device, dtype=torch.bool)
            if query_indices is not None:
                external_mask = external_mask.index_select(1, query_indices)
            external_mask = external_mask[:, None, :, :]
        elif attention_mask.dim() == 4 and attention_mask.shape[-2:] == (seq_len, seq_len):
            external_mask = attention_mask.to(device=device, dtype=torch.bool)
            if query_indices is not None:
                external_mask = external_mask.index_select(2, query_indices)
        else:
            raise ValueError(
                "attention_mask must be [batch, seq], [seq, seq], [batch, seq, seq], "
                f"or [batch, heads, seq, seq]; got {tuple(attention_mask.shape)} "
                f"for batch={batch_size}, seq={seq_len}"
            )
        return external_mask if attn_mask is None else (attn_mask & external_mask)

    @torch.no_grad()
    def sampled_attention_diagnostics(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        query_indices: torch.Tensor | list[int] | tuple[int, ...],
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, float | int | list[int]]:
        """Measure exact attention distributions for selected query positions.

        Q/K projections, post-RoPE RMSNorm, scale, and mask semantics match
        :meth:`forward`. Only ``len(query_indices) * seq_len`` attention logits
        are materialized per batch/head rather than the full square matrix.
        Entropy and KL divergence use natural logarithms, with the uniform
        reference restricted to keys visible from each selected query.
        """
        if x.dim() != 3:
            raise ValueError(f"x must have shape [batch, seq, embedding], got {tuple(x.shape)}")
        batch_size, seq_len, channels = x.shape
        if channels != self.n_head * self.head_dim:
            raise ValueError(f"x embedding dimension must be {self.n_head * self.head_dim}, got {channels}")

        selected_queries = torch.as_tensor(query_indices, dtype=torch.long, device=x.device)
        if selected_queries.dim() != 1 or selected_queries.numel() == 0:
            raise ValueError("query_indices must be a non-empty one-dimensional sequence")
        if bool(((selected_queries < 0) | (selected_queries >= seq_len)).any()):
            raise ValueError(f"query_indices must be in [0, {seq_len})")
        if int(torch.unique(selected_queries).numel()) != int(selected_queries.numel()):
            raise ValueError("query_indices must not contain duplicates")

        q = self.c_q(x).view(batch_size, seq_len, self.n_head, self.head_dim)
        k = self.c_k(x).view(batch_size, seq_len, self.n_head, self.head_dim)
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        q, k = rms_norm(q), rms_norm(k)
        q = q.index_select(1, selected_queries).transpose(1, 2).float()
        k = k.transpose(1, 2).float()

        scale = self.head_dim**-0.5 if self.attention_scale is None else self.attention_scale
        logits = torch.matmul(q, k.transpose(-2, -1)) * scale
        query_count = int(selected_queries.numel())
        resolved_mask = self._build_attention_mask(
            batch_size=batch_size,
            seq_len=seq_len,
            device=x.device,
            attention_mask=attention_mask,
            query_indices=selected_queries,
        )
        if resolved_mask is None:
            visible = torch.ones(
                (batch_size, self.n_head, query_count, seq_len),
                dtype=torch.bool,
                device=x.device,
            )
        else:
            try:
                visible = resolved_mask.expand(batch_size, self.n_head, query_count, seq_len)
            except RuntimeError as exc:
                raise ValueError(
                    "attention_mask batch/head dimensions are not broadcastable to "
                    f"[{batch_size}, {self.n_head}, {query_count}, {seq_len}]"
                ) from exc

        visible_key_count = visible.sum(dim=-1)
        valid_rows = visible_key_count > 0
        valid_row_count = int(valid_rows.sum().item())
        if valid_row_count == 0:
            raise ValueError("selected queries have no visible keys")

        masked_logits = logits.masked_fill(~visible, float("-inf"))
        safe_logits = torch.where(valid_rows[..., None], masked_logits, torch.zeros_like(masked_logits))
        probabilities = torch.softmax(safe_logits, dim=-1)
        probabilities = torch.where(visible & valid_rows[..., None], probabilities, 0.0)

        log_probabilities = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
        entropy = -(probabilities * log_probabilities).sum(dim=-1)
        max_probability = probabilities.max(dim=-1).values
        kl_from_uniform = (visible_key_count.float().log() - entropy).clamp_min(0.0)

        q_rms = q.square().mean(dim=-1).sqrt()
        k_rms = k.square().mean(dim=-1).sqrt()
        visible_keys = visible.any(dim=2)

        def valid_mean(values: torch.Tensor) -> float:
            return float(values.masked_select(valid_rows).mean().item())

        return {
            "selected_query_indices": [int(index) for index in selected_queries.cpu().tolist()],
            "selected_query_count": query_count,
            "valid_attention_row_count": valid_row_count,
            "all_masked_attention_row_count": int(valid_rows.numel()) - valid_row_count,
            "resolved_attention_scale": float(scale),
            "q_rms_mean": valid_mean(q_rms),
            "k_rms_mean": float(k_rms.masked_select(visible_keys).mean().item()),
            "visible_key_count_mean": valid_mean(visible_key_count.float()),
            "visible_key_count_min": int(visible_key_count.masked_select(valid_rows).min().item()),
            "visible_key_count_max": int(visible_key_count.masked_select(valid_rows).max().item()),
            "attention_entropy_mean": valid_mean(entropy),
            "attention_max_probability_mean": valid_mean(max_probability),
            "attention_kl_from_uniform_mean": valid_mean(kl_from_uniform),
        }

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ):
        b, t, c = x.shape
        h, d = self.n_head, self.head_dim
        q = self.c_q(x).view(b, t, h, d)
        k = self.c_k(x).view(b, t, h, d)
        v = self.c_v(x).view(b, t, h, d)
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        q, k = rms_norm(q), rms_norm(k)
        attn_mask = self._build_attention_mask(
            batch_size=b,
            seq_len=t,
            device=x.device,
            attention_mask=attention_mask,
        )
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=attn_mask,
            is_causal=False,
            scale=self.attention_scale,
        )
        return self.c_proj(y.transpose(1, 2).contiguous().view(b, t, c))


class MLP(nn.Module):
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        hidden_dim = resolved_mlp_hidden_dim(config)
        self.gate = Linear(config.n_embd, hidden_dim, bias=False)
        self.up = Linear(config.n_embd, hidden_dim, bias=False)
        self.down = Linear(hidden_dim, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class DiffBlock(nn.Module):
    def __init__(self, config: DiffusionConfig, layer_idx: int):
        super().__init__()
        local_window = (
            int(config.local_attention_window) if layer_idx < int(config.local_attention_lower_layers) else None
        )
        self.attn = BidirectionalAttention(config, local_window=local_window, causal=config.causal)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(rms_norm(x), cos, sin, attention_mask=attention_mask)
        x = x + self.mlp(rms_norm(x))
        return x


class DiffusionTransformer(nn.Module):
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
        self.blocks = nn.ModuleList([DiffBlock(config, layer_idx=i) for i in range(config.n_layer)])
        self.final_norm = nn.LayerNorm(config.n_embd)
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

    @torch.no_grad()
    def init_weights(self):
        if next(self.parameters()).device.type == "meta":
            return
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = precompute_rotary(
            self.config.sequence_len,
            head_dim,
            self.rope_base,
            self.compute_dtype,
            device=self.wte.weight.device,
        )
        self.rope_cos.copy_(cos)
        self.rope_sin.copy_(sin)

        sigma_base = embedding_head_init_std(self.config.n_embd)
        sigma_embedding_head = embedding_head_init_std(
            self.config.n_embd,
            float(self.config.embedding_head_init_std_multiplier),
        )
        sigma_aux = residual_output_projection_init_std(self.config.n_layer)
        nn.init.normal_(self.wte.weight, std=sigma_embedding_head)
        nn.init.normal_(self.lm_head.weight, std=sigma_embedding_head)
        if self.class_head is not None:
            nn.init.normal_(self.class_head.weight, std=sigma_base)
        if self.length_head is not None:
            nn.init.normal_(self.length_head.weight, std=sigma_base)
            nn.init.zeros_(self.length_head.bias)
        if self.target_length_embedding is not None:
            nn.init.normal_(self.target_length_embedding.weight, std=sigma_base)
        if self.target_distance_proj is not None:
            nn.init.normal_(self.target_distance_proj.weight, std=sigma_base)
        if self.mask_position_embedding is not None:
            nn.init.normal_(self.mask_position_embedding.weight, std=sigma_base)
        for head in self.intermediate_heads:
            nn.init.normal_(head.weight, std=sigma_base)
        if self.noise_mlp is not None:
            nn.init.normal_(self.noise_mlp[0].weight, std=sigma_base)
            nn.init.zeros_(self.noise_mlp[2].weight)
        for block in self.blocks:
            for p in block.parameters():
                if p.dim() >= 2:
                    nn.init.normal_(p, std=sigma_aux)
            if self.config.residual_output_projection_init == "zero":
                nn.init.zeros_(block.attn.c_proj.weight)
                nn.init.zeros_(block.mlp.down.weight)
        nn.init.ones_(self.final_norm.weight)
        nn.init.zeros_(self.final_norm.bias)

    def parameter_inventory(self) -> dict[str, object]:
        """Inventory every instantiated named parameter exactly once.

        Components classify physical parameters. Accounting views are explicit
        sums of those components, so no architecture equation is an authority
        for counts or budgets.
        """
        components = {component: 0 for component in PARAMETER_COMPONENT_NAMES}
        component_members = {component: [] for component in PARAMETER_COMPONENT_NAMES}
        named_parameters: list[dict[str, object]] = []
        for name, parameter in self.named_parameters():
            component = "other_parameters"
            for candidate, prefixes in PARAMETER_COMPONENT_PREFIXES:
                if name.startswith(prefixes):
                    component = candidate
                    break
            num_parameters = int(parameter.numel())
            components[component] += num_parameters
            component_members[component].append(name)
            named_parameters.append(
                {
                    "name": name,
                    "component": component,
                    "shape": list(parameter.shape),
                    "num_parameters": num_parameters,
                }
            )

        instantiated_total = sum(int(parameter.numel()) for parameter in self.parameters())
        component_total = sum(components.values())
        if component_total != instantiated_total:
            raise RuntimeError(
                "parameter inventory is incomplete: "
                f"components={component_total}, instantiated={instantiated_total}, "
                f"delta={instantiated_total - component_total}"
            )

        views: dict[str, dict[str, object]] = {}
        for view_name, view_components in PARAMETER_ACCOUNTING_VIEW_COMPONENTS.items():
            views[view_name] = {
                "components": list(view_components),
                "num_parameters": sum(components[component] for component in view_components),
            }
        if int(views["total_parameters"]["num_parameters"]) != instantiated_total:
            raise RuntimeError("total_parameters accounting view does not match the instantiated model")

        return {
            "method": "instantiated_named_parameters_numel",
            "components": components,
            "component_members": component_members,
            "views": views,
            "named_parameters": named_parameters,
        }

    def parameter_counts(self) -> dict[str, int]:
        """Return component/view counts derived only from :meth:`parameter_inventory`.

        Historical flat keys remain because existing launchers and result
        readers consume them.
        """
        inventory = self.parameter_inventory()
        components = inventory["components"]
        views = inventory["views"]
        assert isinstance(components, dict)
        assert isinstance(views, dict)
        counts = {
            "token_embedding": int(components["token_embedding"]),
            "output_head": int(components["output_head"]),
            "transformer_blocks": int(components["transformer_blocks"]),
            "final_norm": int(components["final_norm"]),
            "class_output_head": int(components["class_output_head"]),
            "length_head": int(components["length_head"]),
            "target_length_conditioning": int(components["target_length_conditioning"]),
            "mask_position_embedding": int(components["mask_position_embedding"]),
            "auxiliary_output_heads": int(components["auxiliary_output_heads"]),
            "time_conditioning": int(components["time_conditioning"]),
            "other_parameters": int(components["other_parameters"]),
            "transformer_body_parameters": int(views["transformer_body_parameters"]["num_parameters"]),
            "nanochat_scaling_parameters": int(views["nanochat_scaling_parameters"]["num_parameters"]),
            "non_input_embedding_parameters": int(views["non_input_embedding_parameters"]["num_parameters"]),
            "total_parameters": int(views["total_parameters"]["num_parameters"]),
        }
        return {
            **counts,
            # Compatibility aliases. Each is an alias of one named component or
            # accounting view above; none is computed independently.
            "wte": counts["token_embedding"],
            "lm_head": counts["output_head"],
            "class_head": counts["class_output_head"],
            "auxiliary_heads": counts["auxiliary_output_heads"],
            "transformer": counts["transformer_blocks"],
            "norm": counts["final_norm"],
            "noise_conditioning": counts["time_conditioning"],
            "non_embedding": counts["transformer_body_parameters"],
            "scaling": counts["nanochat_scaling_parameters"],
            "total": counts["total_parameters"],
        }

    def num_scaling_params(self) -> dict[str, int]:
        """Compatibility alias for older callers; prefer :meth:`parameter_counts`."""
        return self.parameter_counts()

    def estimate_flops(self) -> int:
        counts = self.parameter_counts()
        n_head = self.config.n_head
        head_dim = self.config.n_embd // n_head
        seq_len = self.config.sequence_len
        n_layer = self.config.n_layer
        return 6 * counts["non_input_embedding_parameters"] + 12 * n_head * head_dim * seq_len * n_layer

    def architecture_fingerprint(self) -> dict[str, object]:
        """Return a stable architecture and accounting record for run provenance."""
        inventory = self.parameter_inventory()
        counts = self.parameter_counts()
        head_dim = int(self.config.n_embd) // int(self.config.n_head)
        payload: dict[str, object] = {
            "architecture": {
                "depth": int(self.config.n_layer),
                "n_embd": int(self.config.n_embd),
                "n_head": int(self.config.n_head),
                "head_dim": head_dim,
                "max_seq_len": int(self.config.sequence_len),
                "vocab_size": int(self.config.vocab_size),
                "padded_vocab_size": int(self.padded_vocab_size),
                "mlp_type": "swiglu",
                "mlp_hidden_dim": resolved_mlp_hidden_dim(self.config),
                "causal": bool(self.config.causal),
                "embedding_head_tied": self.wte.weight is self.lm_head.weight,
                "output_factorization": str(self.config.output_factorization),
            },
            "attention": {
                "rope_base": float(self.rope_base),
                "qk_norm": "parameter_free_rms_norm_after_rope",
                "configured_scale": float(self.config.attention_scale),
                "resolved_scale": resolved_attention_scale(self.config),
                "scale_mode": "pytorch_default_one_over_sqrt_head_dim"
                if float(self.config.attention_scale) == -1.0
                else "explicit",
                "local_attention_lower_layers": int(self.config.local_attention_lower_layers),
                "local_attention_window": int(self.config.local_attention_window),
            },
            "initialization": {
                "embedding_head_std": embedding_head_init_std(
                    int(self.config.n_embd), float(self.config.embedding_head_init_std_multiplier)
                ),
                "block_matrix_std": residual_output_projection_init_std(int(self.config.n_layer)),
                "residual_output_projection_init": str(self.config.residual_output_projection_init),
            },
            "parameter_inventory": inventory,
            "parameter_counts": counts,
            "parameter_count_method": "instantiated_named_parameters_numel",
            "training_flops_per_token_estimate": self.estimate_flops(),
            "flops_estimate_method": "6x_non_input_embedding_parameter_view_plus_quadratic_attention",
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return {
            "schema_version": 2,
            "fingerprint_sha256": hashlib.sha256(encoded).hexdigest(),
            **payload,
        }

    def uses_hierarchical_output(self) -> bool:
        return self.output_factorization == "hierarchical_frequency_decile"

    def _class_token_ids(self, class_idx: int) -> torch.Tensor:
        return (self.token_class_ids == int(class_idx)).nonzero(as_tuple=True)[0]

    def hierarchical_token_log_probs(
        self,
        class_logits: torch.Tensor,
        token_logits: torch.Tensor,
    ) -> torch.Tensor:
        if not self.uses_hierarchical_output():
            return token_logits
        class_log_probs = F.log_softmax(class_logits.float(), dim=-1)
        token_logits = token_logits.float()
        log_probs = torch.empty_like(token_logits)
        copy_dim = log_probs.dim() - 1
        for class_idx in range(self.output_class_count):
            class_token_ids = self._class_token_ids(class_idx)
            selected = token_logits.index_select(-1, class_token_ids)
            selected = selected - torch.logsumexp(selected, dim=-1, keepdim=True)
            selected = selected + class_log_probs[..., class_idx].unsqueeze(-1)
            log_probs.index_copy_(copy_dim, class_token_ids, selected)
        return log_probs

    def hierarchical_ce(
        self,
        class_logits: torch.Tensor,
        token_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.uses_hierarchical_output():
            raise ValueError("hierarchical_ce requires hierarchical output factorization")
        targets = targets.to(dtype=torch.long)
        original_shape = targets.shape
        flat_targets = targets.reshape(-1)
        class_targets = self.token_class_ids.index_select(0, flat_targets)
        flat_class_logits = class_logits.reshape(-1, self.output_class_count).float()
        flat_token_logits = token_logits.reshape(-1, token_logits.shape[-1]).float()
        class_ce = F.cross_entropy(flat_class_logits, class_targets, reduction="none")
        within_ce = torch.empty_like(class_ce)
        for class_idx in range(self.output_class_count):
            row_ids = (class_targets == class_idx).nonzero(as_tuple=True)[0]
            if row_ids.numel() == 0:
                continue
            class_token_ids = self._class_token_ids(class_idx)
            class_logits_slice = flat_token_logits.index_select(0, row_ids).index_select(1, class_token_ids)
            class_targets_slice = self.token_local_ids.index_select(0, flat_targets.index_select(0, row_ids))
            within_values = F.cross_entropy(class_logits_slice, class_targets_slice, reduction="none")
            within_ce.index_copy_(0, row_ids, within_values)
        class_ce = class_ce.view(original_shape)
        within_ce = within_ce.view(original_shape)
        return class_ce + within_ce, class_ce, within_ce

    def _length_logits_from_hidden(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.length_head is None:
            raise RuntimeError("length head is not enabled")
        if attention_mask is not None and attention_mask.dim() == 2 and attention_mask.shape == hidden.shape[:2]:
            weights = attention_mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
            pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        else:
            pooled = hidden.mean(dim=1)
        return self.length_head(pooled).float()

    def _target_length_conditioning_embeddings(
        self,
        batch_size: int,
        seq_len: int,
        target_lengths: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        device,
    ) -> torch.Tensor | None:
        if self.target_length_embedding is None or self.target_distance_proj is None:
            return None
        if target_lengths is None:
            raise ValueError("target_lengths must be provided when target_length_conditioning is enabled")
        target_lengths = target_lengths.to(device=device, dtype=torch.long).reshape(-1)
        if target_lengths.numel() == 1:
            target_lengths = target_lengths.expand(batch_size)
        if target_lengths.numel() != batch_size:
            raise ValueError(f"target_lengths must have one value or batch values, got {target_lengths.numel()}")
        if target_lengths.numel() and (
            int(target_lengths.min().item()) < 1 or int(target_lengths.max().item()) > int(self.config.sequence_len)
        ):
            raise ValueError("target_lengths entries must be in [1, sequence_len]")

        target_positions = (target_lengths - 1).clamp(0, int(self.config.sequence_len) - 1)
        global_embedding = self.target_length_embedding(target_positions).to(dtype=self.compute_dtype)[:, None, :]

        if position_ids is None:
            positions = torch.arange(seq_len, device=device, dtype=torch.long).view(1, seq_len).expand(batch_size, -1)
        elif position_ids.dim() == 1:
            positions = position_ids.to(device=device, dtype=torch.long).view(1, seq_len).expand(batch_size, -1)
        else:
            positions = position_ids.to(device=device, dtype=torch.long)
        if positions.shape != (batch_size, seq_len):
            raise ValueError(
                f"conditioning positions must have shape {(batch_size, seq_len)}, got {tuple(positions.shape)}"
            )

        denom = float(max(1, int(self.config.sequence_len) - 1))
        signed_distance = (target_positions[:, None].float() - positions.float()) / denom
        abs_distance = signed_distance.abs()
        before_or_at_target = (positions <= target_positions[:, None]).to(dtype=torch.float32)
        features = torch.stack((signed_distance, abs_distance, before_or_at_target), dim=-1)
        distance_embedding = self.target_distance_proj(features.to(dtype=self.compute_dtype))
        return global_embedding + distance_embedding

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
        target_lengths: torch.Tensor | None = None,
    ):
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
        if input_embeddings is None:
            x = self.wte(z_t).to(dtype=self.compute_dtype)
        else:
            expected_shape = (b, t, self.config.n_embd)
            if tuple(input_embeddings.shape) != expected_shape:
                raise ValueError(
                    f"input_embeddings must have shape {expected_shape}, got {tuple(input_embeddings.shape)}"
                )
            x = input_embeddings.to(device=z_t.device, dtype=self.compute_dtype)
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
        for layer_idx, block in enumerate(self.blocks):
            x = block(x, cos, sin, attention_mask=attention_mask)
            if return_aux_logits and layer_idx in self._auxiliary_head_index:
                aux_hidden = rms_norm(x)
                if logits_indices is not None:
                    aux_hidden = aux_hidden.reshape(b * t, -1).index_select(0, logits_indices)
                head_idx = self._auxiliary_head_index[layer_idx]
                aux_logits.append(self.intermediate_heads[head_idx](aux_hidden)[..., : self.config.vocab_size].float())
        x = self.final_norm(x)
        full_hidden = x
        length_logits = self._length_logits_from_hidden(full_hidden, attention_mask) if return_length_logits else None
        if logits_indices is not None:
            x = x.reshape(b * t, -1).index_select(0, logits_indices)
        token_logits = self.lm_head(x)[..., : self.config.vocab_size].float()
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
