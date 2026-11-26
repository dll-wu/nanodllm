"""Model admission and gold-safe native attention for continuation NELBO."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .core_mask_plans import MaskDraw
from .core_prompting import TargetSequence
from .model import DiffusionTransformer
from .training_bd3lm import bd3lm_attention_mask, bd3lm_concat_position_ids


@dataclass(frozen=True)
class TargetGroup:
    start: int
    end: int


class DiffusionContinuationAdapter:
    """Score the M2T predictor, never a uniform-noise/T2T joint likelihood.

    The caller supplies the checkpoint's declared noise and clean-stream mode;
    those cannot be inferred from tensor shapes. Dropout must already be disabled.
    """

    def __init__(
        self,
        model: DiffusionTransformer,
        *,
        noise_type: str,
        clean_stream_mode: str = "block_causal",
        device: torch.device | str = "cpu",
    ) -> None:
        config = model.config
        if model.training:
            raise ValueError("continuation scoring requires model.eval()")
        if config.causal:
            raise ValueError("causal AR models cannot use conditional masked NELBO")
        if config.time_conditioning:
            raise ValueError("cardinality integration requires a time-independent predictor")
        if config.target_length_conditioning or config.output_factorization != "flat":
            raise ValueError("this adapter requires ordinary flat logits without target-length conditioning")
        if noise_type == "mdlm" and config.bd3lm_block_size == 0:
            family = "masked_mdlm"
        elif noise_type == "bd3lm" and config.bd3lm_block_size > 0:
            family = "block_bd3lm"
        else:
            raise ValueError("noise/model mismatch: only absorbing MDLM or native BD3-LM is supported")
        if clean_stream_mode not in {"causal", "block_causal"}:
            raise ValueError("unsupported BD3-LM clean-stream mode")
        self.model = model
        self.family = family
        self.block_size = int(config.bd3lm_block_size)
        self.clean_stream_mode = clean_stream_mode
        self.device = torch.device(device)

    def target_groups(self, sequence: TargetSequence) -> tuple[TargetGroup, ...]:
        self.validate_sequence(sequence)
        if not self.block_size:
            return (TargetGroup(sequence.start, sequence.end),)
        groups = []
        start = sequence.start
        while start < sequence.end:
            end = min(sequence.end, (start // self.block_size + 1) * self.block_size)
            groups.append(TargetGroup(start, end))
            start = end
        return tuple(groups)

    def validate_sequence(self, sequence: TargetSequence) -> None:
        if sequence.end != len(sequence.tokens):
            raise ValueError("continuation targets must end at the last input token")
        if sequence.end > self.model.config.sequence_len:
            raise ValueError("continuation exceeds model context capacity")
        if any(token >= self.model.config.vocab_size for token in sequence.tokens):
            raise ValueError("input contains an out-of-vocabulary token")
        if self.model.config.mask_token_id in sequence.tokens:
            raise ValueError("clean inputs cannot contain the reserved MASK token")
        if self.block_size and sequence.cropped_tokens % self.block_size:
            raise ValueError("block scoring requires block-aligned cropping; use prepare_example(crop_multiple=BL)")

    @torch.inference_mode()
    def _masked_forward(
        self,
        sequence: TargetSequence,
        group: TargetGroup,
        draws: tuple[MaskDraw, ...],
        *,
        masked_tail_tokens: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.model.training:
            raise ValueError("continuation scoring requires model.eval()")
        if group not in self.target_groups(sequence) or not draws:
            raise ValueError("draws must address a nonempty native target group")
        if type(masked_tail_tokens) is not int or masked_tail_tokens < 0:
            raise ValueError("masked_tail_tokens must be a nonnegative integer")
        if masked_tail_tokens and self.block_size:
            raise ValueError("masked-tail readouts require ordinary MDLM, not block diffusion")
        if group.end + masked_tail_tokens > self.model.config.sequence_len:
            raise ValueError("readout canvas exceeds model context capacity")
        for draw in draws:
            if (
                not draw.masked
                or tuple(sorted(set(draw.masked))) != draw.masked
                or any(type(i) is not int or not 0 <= i < group.end - group.start for i in draw.masked)
            ):
                raise ValueError("mask subset must contain unique in-group target indices")
        # Future blocks do not even enter the forward. Only partially revealed
        # gold in this block and completed prior blocks can be conditioning.
        length = group.end
        clean = torch.tensor(sequence.tokens[:length], dtype=torch.long, device=self.device)
        z_t = clean.unsqueeze(0).repeat(len(draws), 1)
        selected = torch.zeros_like(z_t, dtype=torch.bool)
        for row, draw in enumerate(draws):
            selected[row, [group.start + i for i in draw.masked]] = True
        z_t[selected] = self.model.config.mask_token_id
        if masked_tail_tokens:
            # Readout-only continuation space: visible, entirely MASK, and
            # never a scored target. The NELBO path never supplies this option.
            z_t = F.pad(z_t, (0, masked_tail_tokens), value=self.model.config.mask_token_id)
            selected = F.pad(selected, (0, masked_tail_tokens), value=False)
        valid = torch.ones_like(z_t, dtype=torch.bool)
        positions = torch.arange(length + masked_tail_tokens, device=self.device)
        if self.block_size:
            x_0 = clean.unsqueeze(0).repeat(len(draws), 1)
            # No current-block labels in the clean half, even where the native
            # mask already forbids them. Previous clean blocks remain gold.
            x_0[:, group.start // self.block_size * self.block_size :] = self.model.config.mask_token_id
            inputs = torch.cat((z_t, x_0), dim=1)
            attention = bd3lm_attention_mask(
                valid, self.block_size, position_ids=positions, clean_stream_mode=self.clean_stream_mode
            )
            positions = bd3lm_concat_position_ids(positions, batch_size=len(draws), seq_len=length, device=self.device)
        else:
            inputs, attention = z_t, valid
        rows, columns = selected.nonzero(as_tuple=True)
        indices = rows * inputs.shape[1] + columns
        logits = self.model(inputs, logits_indices=indices, attention_mask=attention, position_ids=positions)
        if logits.shape != (len(rows), self.model.config.vocab_size) or not logits.is_floating_point():
            raise ValueError("model must return selected [masked position, vocabulary] logits")
        if not torch.isfinite(logits).all():
            raise ValueError("scored logits must be finite")
        return logits, clean[columns]

    @torch.inference_mode()
    def masked_predictions(
        self, sequence: TargetSequence, group: TargetGroup, *, masked_tail_tokens: int = 0
    ) -> tuple[int, ...]:
        """Raw-vocabulary argmax with this entire target group masked.

        Prior blocks remain gold; this is not a generated-prefix decoder.
        No temperature, token filtering, remasking, or confidence ordering.
        An explicit ordinary-MDLM masked tail adds unscored visible space.
        """
        draw = MaskDraw(tuple(range(group.end - group.start)), 1.0, 0)
        logits, _ = self._masked_forward(sequence, group, (draw,), masked_tail_tokens=masked_tail_tokens)
        return tuple(logits.argmax(dim=-1).cpu().tolist())

    @torch.inference_mode()
    def masked_means(
        self, sequence: TargetSequence, group: TargetGroup, draws: tuple[MaskDraw, ...]
    ) -> tuple[float, ...]:
        logits, targets = self._masked_forward(sequence, group, draws)
        # Float32 CE for half-precision inference; preserve float64 in analytic
        # CPU checks. The estimator itself accumulates in Python float64.
        if logits.dtype in (torch.float16, torch.bfloat16):
            logits = logits.float()
        losses = F.cross_entropy(logits, targets, reduction="none")
        if not torch.isfinite(losses).all():
            raise ValueError("scored losses must be finite")
        # Reduce in a fixed token order on CPU, avoiding device scatter/atomic
        # reduction noise and leaving only model-kernel batch roundoff.
        values = losses.double().cpu().tolist()
        offset = 0
        means = []
        for draw in draws:
            count = len(draw.masked)
            means.append(sum(values[offset : offset + count]) / count)
            offset += count
        return tuple(means)
