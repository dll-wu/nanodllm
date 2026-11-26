"""Student-forced queries and first-error weighting, without hidden gold targets."""

import random

import torch
import torch.nn.functional as F

from .mdlm_auxiliary_views import keyed_seed


def training_prefixes(tokens, positions, visible, *, seed, index, maximum, eos_id):
    """Choose one visible document prefix per row; no following gold enters a query."""
    result = []
    for row, (ids, pos, valid) in enumerate(zip(tokens.tolist(), positions.tolist(), visible.tolist(), strict=True)):
        starts = [0] + [i for i in range(1, len(pos)) if pos[i] != pos[i - 1] + 1] + [len(pos)]
        candidates = []
        for start, end in zip(starts[:-1], starts[1:], strict=True):
            length = 0
            for i in range(start, min(end, start + maximum)):
                if not valid[i] or ids[i] == eos_id:
                    break
                length += 1
            if length:
                candidates.append((start, length))
        if not candidates:
            continue
        rng = random.Random(keyed_seed(seed, index, f"proposal-prefix:{row}"))
        start, length = candidates[rng.randrange(len(candidates))]
        count = rng.randint(1, length)
        result.append(ids[start : start + count])
    return result


def query_batch(prefixes, *, count, tail, mask_id, device, proposals=None):
    """Student queries are same-position; causal teacher queries are shifted by one."""
    if not prefixes or any(not p for p in prefixes) or count < 1 or tail < count:
        raise ValueError("empty prefix or invalid proposal window")
    causal = proposals is not None
    length = max(map(len, prefixes)) + (count if causal else tail)
    tokens = torch.full((len(prefixes), length), mask_id, device=device, dtype=torch.long)
    visible = torch.zeros_like(tokens, dtype=torch.bool)
    indices = []
    for row, prefix in enumerate(prefixes):
        n = len(prefix)
        values = prefix + (proposals[row].tolist() if causal else [])
        tokens[row, : len(values)] = torch.tensor(values, device=device)
        visible[row, : n + (count if causal else tail)] = True
        indices.extend(row * length + n + i - int(causal) for i in range(count))
    return tokens, visible, torch.tensor(indices, device=device)


def proposal_logits(model, prefixes, *, count, tail, mask_id, device, proposals=None):
    tokens, visible, indices = query_batch(
        prefixes, count=count, tail=tail, mask_id=mask_id, device=device, proposals=proposals
    )
    logits = model(tokens, attention_mask=visible, logits_indices=indices).float()
    if not torch.isfinite(logits).all():
        raise RuntimeError("nonfinite proposal logits")
    return logits.reshape(len(prefixes), count, -1), {
        "forward_calls": 1,
        "input_positions": tokens.numel(),
        "input_length_squared": len(prefixes) * tokens.shape[1] ** 2,
        "projected_positions": indices.numel(),
    }


def legal_argmax(logits, support):
    return support[logits.detach().index_select(-1, support).argmax(-1)]


def first_error_loss(logits, targets, proposals, *, eos_id, first_error_weight):
    """Normalize each span's weights, so emphasis does not multiply its loss scale."""
    if logits.shape[:-1] != targets.shape or targets.shape != proposals.shape:
        raise ValueError("proposal loss shapes differ")
    active = proposals.eq(eos_id).cumsum(-1) - proposals.eq(eos_id).long() == 0
    disagreement = proposals.ne(targets) & active
    first = disagreement & (disagreement.long().cumsum(-1) == 1)
    weights = (1 + (first_error_weight - 1) * first.float()) * active
    ce = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none").reshape_as(targets)
    loss = ((ce * weights).sum(-1) / weights.sum(-1)).mean()
    return loss, {
        "targets": int(active.sum()),
        "first_errors": int(first.sum()),
        "teacher_agreements": int((~disagreement & active).sum()),
        "teacher_eos_targets": int((targets.eq(eos_id) & active).sum()),
        "target_mean_ce": float(ce.detach()[active].mean()),
    }


def advance_prefixes(prefixes, proposals, eos_id):
    """Never generate a second round after a student's EOS; do not correct its prefix."""
    return [p + y for p, y in zip(prefixes, proposals.tolist(), strict=True) if eos_id not in y]


def soft_teacher_loss(logits, teacher_logits, proposals, *, eos_id, support):
    """Temperature-one forward cross-entropy; frozen teacher on legal support.

    The student normalizer remains full-vocabulary, like hard-target CE. Teacher
    mass is normalized over generation-legal tokens (including EOS); no gradient
    flows through the teacher or the generated prefix.
    """
    if logits.shape != teacher_logits.shape or logits.shape[:-1] != proposals.shape:
        raise ValueError("soft proposal loss shapes differ")
    active = proposals.eq(eos_id).cumsum(-1) - proposals.eq(eos_id).long() == 0
    teacher_logp = teacher_logits.detach().float().index_select(-1, support).log_softmax(-1)
    probabilities = teacher_logp.exp()
    student_logp = logits.float().log_softmax(-1).index_select(-1, support)
    ce = -(probabilities * student_logp).sum(-1)
    entropy = -(probabilities * teacher_logp).sum(-1)
    targets = legal_argmax(teacher_logits, support)
    disagreement = proposals.ne(targets) & active
    first = disagreement & (disagreement.long().cumsum(-1) == 1)
    loss = ((ce * active).sum(-1) / active.sum(-1)).mean()
    return loss, {
        "targets": int(active.sum()),
        "first_errors": int(first.sum()),
        "teacher_agreements": int((~disagreement & active).sum()),
        "teacher_eos_targets": int((targets.eq(eos_id) & active).sum()),
        "target_mean_ce": float(ce.detach()[active].mean()),
        "teacher_entropy": float(entropy[active].mean()),
        "forward_kl": float((ce.detach() - entropy)[active].mean()),
    }
