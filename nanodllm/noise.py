import random

import torch


def sample_positive_uniform_scalar(t_min: float, t_max: float) -> float:
    """Sample uniformly while rejecting the singular t=0 endpoint."""

    low = float(t_min)
    high = float(t_max)
    if not 0.0 <= low <= high <= 1.0:
        raise ValueError("noise times must satisfy 0 <= t_min <= t_max <= 1")
    if low == high:
        if low <= 0.0:
            raise ValueError("schedule-weighted MDLM cannot use a fixed t=0")
        return low
    while True:
        value = random.uniform(low, high)
        if value > 0.0:
            return value


def sample_positive_uniform_tensor(
    shape: tuple[int, ...] | torch.Size,
    *,
    device: torch.device | str,
    t_min: float,
    t_max: float,
) -> torch.Tensor:
    """Tensor equivalent of ``sample_positive_uniform_scalar``."""

    low = float(t_min)
    high = float(t_max)
    if not 0.0 <= low <= high <= 1.0:
        raise ValueError("noise times must satisfy 0 <= t_min <= t_max <= 1")
    if low == high:
        if low <= 0.0:
            raise ValueError("schedule-weighted MDLM cannot use a fixed t=0")
        return torch.full(shape, low, device=device, dtype=torch.float32)
    if low == 0.0:
        # torch.rand(float32) has a 24-bit grid that includes zero. Sampling the
        # nonzero grid points directly is its rejection distribution without a
        # device synchronization in every training microbatch.
        resolution = 1 << 24
        unit = torch.randint(1, resolution, shape, device=device, dtype=torch.int32).to(dtype=torch.float32)
        return unit.mul_(high / resolution)
    return torch.empty(shape, device=device, dtype=torch.float32).uniform_(low, high)


class MDLMNoise:
    def __init__(self, mask_token_id: int):
        self.mask_token_id = mask_token_id

    def corrupt(
        self,
        x_0: torch.Tensor,
        t: float | torch.Tensor,
        maskable: torch.Tensor | None = None,
        *,
        uniforms: torch.Tensor | None = None,
    ):
        if isinstance(t, torch.Tensor):
            t_threshold = t.detach().to(device=x_0.device, dtype=torch.float32)
            if t_threshold.ndim == 0:
                pass
            elif t_threshold.ndim == 1 and t_threshold.shape[0] == x_0.shape[0]:
                t_threshold = t_threshold[:, None]
            else:
                raise ValueError(
                    "MDLM scalar corruption t must be a scalar or per-row tensor "
                    f"with shape ({x_0.shape[0]},), got {tuple(t_threshold.shape)}"
                )
        else:
            t_threshold = float(t)
        if uniforms is None:
            uniforms = torch.rand(x_0.shape, device=x_0.device)
        elif uniforms.shape != x_0.shape:
            raise ValueError(
                "MDLM corruption uniforms must match the token tensor shape, "
                f"got {tuple(uniforms.shape)} and {tuple(x_0.shape)}"
            )
        elif uniforms.device != x_0.device:
            raise ValueError(
                f"MDLM corruption uniforms must be on the token tensor device, got {uniforms.device} and {x_0.device}"
            )
        elif not uniforms.dtype.is_floating_point:
            raise TypeError(f"MDLM corruption uniforms must be floating point, got {uniforms.dtype}")
        mask = uniforms < t_threshold
        if maskable is not None:
            mask &= maskable
        z_t = x_0.clone()
        z_t[mask] = self.mask_token_id
        return z_t, mask.float()

    def corrupt_per_token_t(
        self,
        x_0: torch.Tensor,
        t_min: float,
        t_max: float,
        maskable: torch.Tensor | None = None,
    ):
        t_values = sample_positive_uniform_tensor(
            x_0.shape,
            device=x_0.device,
            t_min=float(t_min),
            t_max=float(t_max),
        )
        mask = torch.rand(x_0.shape, device=x_0.device) < t_values
        if maskable is not None:
            mask &= maskable
        z_t = x_0.clone()
        z_t[mask] = self.mask_token_id
        return z_t, mask.float(), t_values

    def corrupt_with_complement(self, x_0: torch.Tensor, t: float, maskable: torch.Tensor | None = None):
        base_mask = torch.rand(x_0.shape, device=x_0.device) < t
        if maskable is not None:
            std_mask = base_mask & maskable
            comp_mask = (~base_mask) & maskable
        else:
            std_mask = base_mask
            comp_mask = ~base_mask
        z_std = x_0.masked_fill(std_mask, self.mask_token_id)
        z_comp = x_0.masked_fill(comp_mask, self.mask_token_id)
        w_std = std_mask.float()
        w_comp = comp_mask.float()
        n = x_0.shape[0]
        comp_rows = torch.zeros(n, dtype=torch.bool, device=x_0.device)
        comp_rows[torch.randperm(n, device=x_0.device)[: n // 2]] = True
        comp_rows = comp_rows[:, None]
        return torch.where(comp_rows, z_comp, z_std), torch.where(comp_rows, w_comp, w_std)

    def corrupt_suffix(self, x_0: torch.Tensor, t: float | torch.Tensor, maskable: torch.Tensor | None = None):
        mask = torch.zeros_like(x_0, dtype=torch.bool)
        if maskable is None:
            maskable = torch.ones_like(x_0, dtype=torch.bool)
        if isinstance(t, torch.Tensor):
            if t.ndim != 1 or t.shape[0] != x_0.shape[0]:
                raise ValueError(f"suffix masking tensor t must have shape ({x_0.shape[0]},), got {tuple(t.shape)}")
            t_values = t.detach().to(device=x_0.device, dtype=torch.float32)
        else:
            t_values = None
        for row_idx in range(x_0.shape[0]):
            row_maskable = maskable[row_idx].nonzero(as_tuple=True)[0]
            n = row_maskable.numel()
            if n == 0:
                continue
            row_t = float(t_values[row_idx].item()) if t_values is not None else float(t)
            k = max(1, int(round(row_t * n)))
            k = min(k, n)
            cutoff_pos = int(row_maskable[n - k].item())
            mask[row_idx, cutoff_pos:] = True
        mask &= maskable
        z_t = x_0.clone()
        z_t[mask] = self.mask_token_id
        return z_t, mask.float()


class UniformNoise:
    def __init__(self, mask_token_id: int, vocab_size: int):
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.uniform_support_size = vocab_size - 1 if 0 <= mask_token_id < vocab_size else vocab_size

    def sample_uniform_tokens(self, shape, device) -> torch.Tensor:
        tokens = torch.randint(0, self.uniform_support_size, shape, device=device)
        if 0 <= self.mask_token_id < self.vocab_size:
            tokens = tokens + (tokens >= self.mask_token_id).long()
        return tokens

    def corrupt(self, x_0: torch.Tensor, t: float, maskable: torch.Tensor | None = None):
        keep = torch.rand(x_0.shape, device=x_0.device) < (1.0 - float(t))
        if maskable is None:
            maskable = torch.ones_like(x_0, dtype=torch.bool)
        z_t = x_0.clone()
        corrupt_positions = (~keep) & maskable
        n_corrupt = int(corrupt_positions.sum().item())
        if n_corrupt > 0:
            z_t[corrupt_positions] = self.sample_uniform_tokens((n_corrupt,), device=x_0.device)
        return z_t, maskable.float()

    def corrupt_with_hierarchical_block_t(
        self,
        x_0: torch.Tensor,
        t_min: float,
        t_max: float,
        coarse_block_size: int,
        fine_block_size: int,
        maskable: torch.Tensor | None = None,
    ):
        batch_size, seq_len = x_0.shape
        n_coarse_blocks = (seq_len + coarse_block_size - 1) // coarse_block_size
        n_fine_blocks = (seq_len + fine_block_size - 1) // fine_block_size
        t_coarse_blocks = torch.empty(
            (batch_size, n_coarse_blocks),
            device=x_0.device,
            dtype=torch.float32,
        ).uniform_(float(t_min), float(t_max))
        t_fine_blocks = torch.empty(
            (batch_size, n_fine_blocks),
            device=x_0.device,
            dtype=torch.float32,
        ).uniform_(float(t_min), float(t_max))

        positions = torch.arange(seq_len, device=x_0.device)
        coarse_ids = torch.div(positions, int(coarse_block_size), rounding_mode="floor")
        fine_ids = torch.div(positions, int(fine_block_size), rounding_mode="floor")
        t_i = (0.5 * t_coarse_blocks.index_select(1, coarse_ids) + 0.5 * t_fine_blocks.index_select(1, fine_ids)).clamp(
            float(t_min), float(t_max)
        )
        keep = torch.rand_like(t_i) < (1.0 - t_i)
        if maskable is None:
            maskable = torch.ones_like(x_0, dtype=torch.bool)

        z_t = x_0.clone()
        corrupt_positions = (~keep) & maskable
        n_corrupt = int(corrupt_positions.sum().item())
        if n_corrupt > 0:
            z_t[corrupt_positions] = self.sample_uniform_tokens((n_corrupt,), device=x_0.device)
        return z_t, maskable.float()


class MaskUniformNoise(UniformNoise):
    def __init__(self, mask_token_id: int, vocab_size: int, uniform_mix_prob: float):
        super().__init__(mask_token_id=mask_token_id, vocab_size=vocab_size)
        self.uniform_mix_prob = float(uniform_mix_prob)

    def corrupt(self, x_0: torch.Tensor, t: float, maskable: torch.Tensor | None = None):
        if maskable is None:
            maskable = torch.ones_like(x_0, dtype=torch.bool)
        corrupt_positions = (torch.rand(x_0.shape, device=x_0.device) < float(t)) & maskable
        uniform_positions = corrupt_positions & (torch.rand(x_0.shape, device=x_0.device) < self.uniform_mix_prob)
        mask_positions = corrupt_positions & ~uniform_positions

        z_t = x_0.clone()
        z_t[mask_positions] = self.mask_token_id
        n_uniform = int(uniform_positions.sum().item())
        if n_uniform > 0:
            z_t[uniform_positions] = self.sample_uniform_tokens((n_uniform,), device=x_0.device)
        return z_t, corrupt_positions.float()

    def sample_terminal_tokens(self, shape, device) -> torch.Tensor:
        z_t = torch.full(shape, self.mask_token_id, dtype=torch.long, device=device)
        uniform_positions = torch.rand(shape, device=device) < self.uniform_mix_prob
        n_uniform = int(uniform_positions.sum().item())
        if n_uniform > 0:
            z_t[uniform_positions] = self.sample_uniform_tokens((n_uniform,), device=device)
        return z_t


def build_noise(noise_config, mask_token_id: int, vocab_size: int):
    if noise_config.type in ("mdlm", "bd3lm"):
        return MDLMNoise(mask_token_id)
    if noise_config.type == "uniform":
        return UniformNoise(mask_token_id, vocab_size)
    if noise_config.type == "mask_uniform":
        return MaskUniformNoise(mask_token_id, vocab_size, noise_config.mask_uniform_mix_prob)
    raise ValueError(f"unknown noise type: {noise_config.type!r}")
