"""Pure, stateless window/prefix plans; hidden gold never enters a model canvas."""

import hashlib
import math
import random
from dataclasses import dataclass

import torch

from .mdlm_auxiliary_config import MDLMAuxiliaryConfig


def keyed_seed(seed: int, microbatch: int, stream: str) -> int:
    digest = hashlib.sha256(f"mdlm-aux-v1:{seed}:{microbatch}:{stream}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def generator(device, seed: int, microbatch: int, stream: str) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(keyed_seed(seed, microbatch, stream))


@dataclass(frozen=True)
class AuxiliaryView:
    canvas: torch.Tensor
    visible: torch.Tensor
    targets: torch.Tensor
    suffix: torch.Tensor
    segments: tuple[tuple[int, int, int, int, int], ...]
    excluded_segments: int


def make_view(
    x_0,
    position_ids,
    config: MDLMAuxiliaryConfig,
    *,
    microbatch: int,
    bos_id: int,
    eos_id: int,
    mask_id: int,
    special_ids: tuple[int, ...],
    attention_mask: torch.Tensor | None = None,
    loss_mask: torch.Tensor | None = None,
) -> AuxiliaryView:
    """Dispatch strict frontiers or select legacy nonterminal lexical endpoints.

    Historical modes use fully populated packed rows without the row-level
    prefix/length augmentation. Their endpoints precede the original segment
    end; prefix_frontier instead preserves the supplied visibility/loss masks.
    """
    if x_0.ndim != 2 or position_ids is None or position_ids.shape != x_0.shape:
        raise ValueError("auxiliary views require packed 2D tokens and segment position IDs")
    if config.mode == "prefix_frontier":
        return make_frontier_view(
            x_0, position_ids, attention_mask, loss_mask, config, microbatch=microbatch, mask_id=mask_id
        )
    positions, tokens = position_ids.detach().cpu().tolist(), x_0.detach().cpu().tolist()
    visible = torch.zeros_like(x_0, dtype=torch.bool)
    suffix = torch.zeros_like(visible)
    segments, excluded = [], 0
    blocked = set(special_ids) | {bos_id, eos_id, mask_id}
    for row, (ids, pos) in enumerate(zip(tokens, positions, strict=True)):
        starts = [0] + [i for i in range(1, len(pos)) if pos[i] != pos[i - 1] + 1] + [len(pos)]
        for ordinal, (start, end) in enumerate(zip(starts[:-1], starts[1:], strict=True)):
            length = end - start
            lo = start + max(1, math.ceil(config.window_min_fraction * length))
            hi = min(end - 1, start + math.floor(config.window_max_fraction * length))
            prefix_counts = [0]
            for token in ids[start:end]:
                prefix_counts.append(prefix_counts[-1] + int(token not in blocked))
            candidates = [
                cut for cut in range(lo, hi + 1) if ids[cut - 1] not in blocked and prefix_counts[cut - start] >= 2
            ]
            if not candidates:
                excluded += 1
                continue
            rng = random.Random(keyed_seed(config.seed, microbatch, f"window:{row}:{ordinal}"))
            cut = candidates[rng.randrange(len(candidates))]
            eligible = [i for i in range(start, cut) if ids[i] not in blocked]
            prefix_rng = random.Random(keyed_seed(config.seed, microbatch, f"prefix:{row}:{ordinal}"))
            frac = prefix_rng.uniform(config.prefix_min_fraction, config.prefix_max_fraction)
            boundary = min(eligible[-1], max(eligible[0] + 1, start + round(frac * (cut - start))))
            visible[row, start:cut] = True
            suffix[row, boundary:cut] = True
            segments.append((row, start, cut, boundary, end))
    lossable = visible.clone()
    for token in blocked:
        lossable &= x_0.ne(token)
    suffix &= lossable
    targets = lossable if config.mode == "boundary" else suffix
    canvas = x_0.masked_fill(~visible, mask_id)
    if config.mode != "boundary":
        canvas = canvas.masked_fill(suffix, mask_id)
    return AuxiliaryView(canvas, visible, targets, suffix, tuple(segments), excluded)


def make_frontier_view(x_0, position_ids, attention_mask, loss_mask, config, *, microbatch, mask_id):
    """Draw a uniform discrete cutoff in each segment's eligible suffix.

    Preserve the packer's crop and existing clean-prefix augmentation. Unlike
    historical full_suffix views, do not crop lexical endpoints or exclude EOS.
    All attended positions after the cutoff are MASK, never scattered gold.
    """
    if any(
        value is None or value.shape != x_0.shape or value.dtype != torch.bool for value in (attention_mask, loss_mask)
    ):
        raise ValueError("prefix frontier requires explicit boolean visibility and loss masks")
    if bool((loss_mask & ~attention_mask).any()):
        raise ValueError("prefix frontier targets must be visible")
    positions = position_ids.detach().cpu().tolist()
    visibility, eligibility = attention_mask.cpu().tolist(), loss_mask.cpu().tolist()
    targets, suffix = torch.zeros_like(loss_mask), torch.zeros_like(loss_mask)
    segments, excluded = [], 0
    for row, pos in enumerate(positions):
        starts = [0] + [i for i in range(1, len(pos)) if pos[i] != pos[i - 1] + 1] + [len(pos)]
        for ordinal, (start, end) in enumerate(zip(starts[:-1], starts[1:], strict=True)):
            active = [i for i in range(start, end) if visibility[row][i]]
            eligible = [i for i in active if eligibility[row][i]]
            if not eligible:
                excluded += 1
                continue
            if active != list(range(start, active[-1] + 1)) or eligible != list(range(eligible[0], active[-1] + 1)):
                raise ValueError("prefix frontier requires a visible segment prefix and contiguous lossable suffix")
            count = len(eligible)
            lo = min(count - 1, math.floor(config.prefix_min_fraction * count))
            hi = min(count - 1, math.floor(config.prefix_max_fraction * count))
            rng = random.Random(keyed_seed(config.seed, microbatch, f"frontier:{row}:{ordinal}"))
            boundary = eligible[rng.randint(lo, hi)]
            cut = active[-1] + 1
            suffix[row, boundary:cut] = True
            targets[row, boundary:cut] = True
            segments.append((row, start, cut, boundary, end))
    canvas = x_0.masked_fill(suffix | ~attention_mask, mask_id)
    return AuxiliaryView(canvas, attention_mask.clone(), targets, suffix, tuple(segments), excluded)


def fill_mask(shape, targets, config, microbatch):
    return (
        torch.rand(shape, device=targets.device, generator=generator(targets.device, config.seed, microbatch, "fill"))
        < config.draft_fraction
    ) & targets


def legal_tokens(vocab_size: int, forbidden: tuple[int, ...], device):
    support = torch.ones(vocab_size, dtype=torch.bool, device=device)
    for token in forbidden:
        if 0 <= token < vocab_size:
            support[token] = False
    result = support.nonzero(as_tuple=True)[0]
    if result.numel() == 0:
        raise ValueError("empty draft vocabulary")
    return result


def correction_loss(logits, gold, weights, denominator: int):
    """All selected suffix targets count, including correct visible drafts."""
    if denominator <= 0 or logits.shape[0] != gold.numel() or weights.shape != gold.shape:
        raise ValueError("invalid auxiliary loss support")
    ce = torch.nn.functional.cross_entropy(logits.float(), gold, reduction="none")
    return (ce * weights).sum() / denominator, ce.mean()
