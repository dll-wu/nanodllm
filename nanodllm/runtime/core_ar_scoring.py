"""CORE causal-logit reduction; model admission remains the caller's responsibility.

Only teacher-forced causal AR logits are meaningful here. These functions do
not select an attention mask or define an MDLM/block-diffusion adaptation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .core_prompting import PreparedExample


def padded_inputs(example: PreparedExample, *, device: torch.device | str = "cpu") -> torch.Tensor:
    """Right-pad with BOS; valid targets never include the padding positions."""
    width = max(len(sequence.tokens) for sequence in example.sequences)
    result = torch.full((len(example.sequences), width), example.bos_token_id, dtype=torch.long, device=device)
    for index, sequence in enumerate(example.sequences):
        result[index, : len(sequence.tokens)] = torch.tensor(sequence.tokens, dtype=torch.long, device=device)
    return result


@dataclass(frozen=True)
class ContinuationScore:
    target_tokens: tuple[int, ...]
    token_nll: tuple[float, ...]
    predicted_tokens: tuple[int, ...]
    mean_nll: float


@dataclass(frozen=True)
class ARExampleScore:
    continuations: tuple[ContinuationScore, ...]
    predicted_choice: int | None
    correct: bool


@torch.no_grad()
def score_ar_logits(example: PreparedExample, logits: torch.Tensor) -> ARExampleScore:
    """Reduce [candidate, position, vocabulary] next-token logits in their dtype.

    CPU float32 parity is tested against pinned nanochat. Accelerator/low-
    precision parity and model-family admission are separate future work.
    """
    inputs = padded_inputs(example, device=logits.device)
    if logits.ndim != 3 or logits.shape[:2] != inputs.shape or not logits.is_floating_point():
        raise ValueError("logits must be floating [candidate, padded_position, vocabulary]")
    if logits.shape[-1] <= int(inputs.max()):
        raise ValueError("input token ID is outside the logit vocabulary")
    return score_ar_target_logits(
        example,
        tuple(logits[i, s.start - 1 : s.end - 1] for i, s in enumerate(example.sequences)),
    )


@torch.no_grad()
def score_ar_target_logits(example: PreparedExample, target_logits: tuple[torch.Tensor, ...]) -> ARExampleScore:
    """Reduce only projected causal predecessor logits; identical span rules."""
    if len(target_logits) != len(example.sequences):
        raise ValueError("one target-logit tensor is required per candidate")
    scores: list[ContinuationScore] = []
    for sequence, selected in zip(example.sequences, target_logits, strict=True):
        if selected.ndim != 2 or selected.shape[0] != sequence.end - sequence.start or not selected.is_floating_point():
            raise ValueError("target logits must be floating [target_tokens, vocabulary]")
        target = torch.tensor(sequence.tokens[sequence.start : sequence.end], device=selected.device, dtype=torch.long)
        if selected.shape[1] <= int(target.max()):
            raise ValueError("target token ID is outside the logit vocabulary")
        if not torch.isfinite(selected).all():
            raise ValueError("scored logits must be finite")
        losses = F.cross_entropy(selected, target, reduction="none")
        if not torch.isfinite(losses).all():
            raise ValueError("scored losses must be finite")
        predictions = selected.argmax(dim=-1)
        scores.append(
            ContinuationScore(
                tuple(target.tolist()), tuple(losses.tolist()), tuple(predictions.tolist()), losses.mean().item()
            )
        )
    if example.task_type == "language_modeling":
        choice = None
        correct = scores[0].predicted_tokens == scores[0].target_tokens
    else:
        choice = min(range(len(scores)), key=lambda index: scores[index].mean_nll)
        correct = choice == example.gold
    return ARExampleScore(tuple(scores), choice, correct)
