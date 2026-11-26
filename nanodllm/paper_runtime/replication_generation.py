"""Token sampling and termination metrics shared across block decoders."""

import math
from collections import Counter

import torch


def draw(logits, *, temperature, top_p, generator, forbidden):
    """Sample tokens but rank commitment by their untempered probabilities."""
    if not math.isfinite(temperature) or temperature < 0 or not 0 < top_p <= 1:
        raise ValueError("invalid temperature or top-p")
    scores = logits.float().clone()
    if forbidden:
        scores[..., list(forbidden)] = -torch.inf
    if bool(torch.isnan(scores).any()) or not bool(torch.isfinite(scores.max(-1).values).all()):
        raise ValueError("no finite generation support")
    raw = scores.softmax(-1)
    if temperature == 0:
        tokens = scores.argmax(-1)
    else:
        scores /= temperature
        if top_p < 1:
            ordered, indices = scores.sort(-1, descending=True)
            remove = ordered.softmax(-1).cumsum(-1) > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            scores.masked_fill_(torch.zeros_like(remove).scatter(-1, indices, remove), -torch.inf)
        flat = scores.softmax(-1).reshape(-1, scores.shape[-1])
        tokens = torch.multinomial(flat, 1, generator=generator).reshape(scores.shape[:-1])
    return tokens, raw.gather(-1, tokens[..., None]).squeeze(-1)


def token_metrics(raw, *, eos, mask, actions, capacity):
    eos_position = raw.index(eos) if eos in raw else None
    response = raw if eos_position is None else raw[: eos_position + 1]
    content = response if eos_position is None else response[:-1]
    residual = [i for i, value in enumerate(response) if value in {mask, *actions}]
    metrics = {
        "raw_token_ids": raw,
        "token_ids": response,
        "content_tokens": len(content),
        "eos_position": eos_position,
        "eos_present": eos_position is not None,
        "capacity_hit": eos_position is None and len(raw) >= capacity,
        "residual_positions": residual,
        "valid_completion": not residual,
    }
    for width in (1, 2, 3, 4):
        grams = [tuple(content[i : i + width]) for i in range(max(0, len(content) - width + 1))]
        counts = Counter(grams)
        metrics[f"distinct_{width}"] = len(counts) / len(grams) if grams else None
        metrics[f"repeated_{width}_fraction"] = (len(grams) - len(counts)) / len(grams) if grams else None
    return metrics
