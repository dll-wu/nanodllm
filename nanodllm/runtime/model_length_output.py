"""Length prediction and conditioning methods; model composition owns their parameters."""

import torch


class LengthOutput:
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
