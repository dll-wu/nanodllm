"""Position selection, cached block generation and boundary queries for block recipes."""

import time

import torch

from .causal_prefix_cache import CausalPrefixKVCache
from .core_evaluation_cells import require
from .replication_eval_support import device_of, generator_for
from .replication_generation import draw, token_metrics


def legal_prob(logits, forbidden):
    legal = logits.float().clone()
    legal[..., forbidden] = -torch.inf
    return legal.softmax(-1)


def select(prob, remaining, k, policy, *, sampled_confidence=None, permutation=None):
    require(1 <= k <= int(remaining.sum()), "invalid commitment count")
    if policy == "left_to_right":
        return remaining.nonzero().flatten()[:k]
    if policy == "random":
        require(permutation is not None, "random order needs an independent permutation")
        return permutation[remaining[permutation]][:k]
    if policy == "max_probability":
        score = prob.max(-1).values
    elif policy == "probability_margin":
        top = prob.topk(2, dim=-1).values
        score = top[:, 0] - top[:, 1]
    else:
        require(
            policy in {"confidence", "confidence_resample"} and sampled_confidence is not None,
            "invalid sampled-token policy",
        )
        score = sampled_confidence
    return score.masked_fill(~remaining, -torch.inf).argsort(descending=True, stable=True)[:k]


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def generate(model, tokenizer, prompt, point, proposal, *, capacity=256, block_length=32):
    """The only prompt fields read are identity and prefix; gold cannot select tokens."""
    require(not model.training, "evaluation model required")
    require(type(block_length) is int and block_length > 0, "positive block length required")
    require(type(point["k"]) is int and point["k"] > 0, "positive commitment count required")
    prefix = list(prompt["prefix_ids"])
    require(prefix and capacity > 0 and len(prefix) + capacity <= model.config.sequence_len, "invalid canvas")
    device, mask, eos = device_of(model), tokenizer.get_mask_token_id(), tokenizer.get_eos_token_id()
    forbidden = sorted({mask, *tokenizer.get_generation_blocked_token_ids()})
    require(eos not in forbidden and mask not in prefix, "invalid generation support")
    ar = point["policy"] == "ar"
    require(bool(model.config.causal) == ar, "AR/diffusion model mismatch")
    rng = generator_for(prompt["id"], proposal, "block-refresh-generation", device=device)
    commit_rng = generator_for(prompt["id"], proposal, "independent-commitment", device=device)
    order_rng = generator_for(prompt["id"], proposal, "random-position-order", device=device)
    sampling = {"temperature": point["temperature"], "top_p": point["top_p"], "forbidden": forbidden}
    work = {"model_forwards": 0, "active_forwards": 0, "clean_commit_forwards": 0, "processed_query_tokens": 0}
    _sync(device)
    started = time.perf_counter()
    cache = None if ar else CausalPrefixKVCache(model, batch_size=1, capacity=len(prefix) + capacity)

    def append(tokens):
        cache.append(torch.tensor([tokens], device=device))
        work["model_forwards"] += 1
        work["clean_commit_forwards"] += 1
        work["processed_query_tokens"] += len(tokens)

    if cache is not None:
        append(prefix)
    raw, first_groups = [], []
    while len(raw) < capacity:
        if ar:
            ids = torch.tensor([prefix + raw], device=device)
            z = model(ids, logits_indices=ids.new_tensor([ids.shape[1] - 1]))
            tokens, _ = draw(z, generator=rng, **sampling)
            raw.append(int(tokens[0]))
            work["model_forwards"] += 1
            work["active_forwards"] += 1
            work["processed_query_tokens"] += ids.numel()
            if raw[-1] == eos:
                break
            continue
        width = min(block_length, capacity - len(raw))
        active = torch.full((width,), mask, device=device, dtype=torch.long)
        permutation = torch.randperm(width, generator=order_rng, device=device)
        while bool(active.eq(mask).any()):
            z = cache.logits(active[None])[0]
            prob = legal_prob(z, forbidden)
            proposed, confidence = draw(z, generator=rng, **sampling)
            remaining = active.eq(mask)
            indices = select(
                prob,
                remaining,
                min(point["k"], int(remaining.sum())),
                point["policy"],
                sampled_confidence=confidence,
                permutation=permutation,
            )
            committed = (
                draw(z, generator=commit_rng, **sampling)[0] if point["policy"] == "confidence_resample" else proposed
            )
            if not raw:
                first_groups.append(indices.tolist())
            # Every member uses the same pre-commit forward, including K4.
            active[indices] = committed[indices]
            work["model_forwards"] += 1
            work["active_forwards"] += 1
            work["processed_query_tokens"] += width
        tokens = active.tolist()
        raw.extend(tokens)
        if eos in tokens or len(raw) == capacity:
            break
        append(tokens)
    _sync(device)
    elapsed = time.perf_counter() - started
    result = token_metrics(raw, eos=eos, mask=mask, actions=[], capacity=capacity)
    return {
        **result,
        "text": tokenizer.decode(result["token_ids"]),
        "first_block_groups": first_groups,
        "work": work,
        "elapsed_seconds": elapsed,
        "content_tokens_per_second": result["content_tokens"] / elapsed,
        "allocated_kv_bytes": cache.allocated_kv_bytes if cache else 0,
    }


@torch.inference_mode()
def boundary_probe(model, tokenizer, prompt, *, noisy=False):
    prefix = list(prompt["prefix_ids"])
    device = device_of(model)
    mask, eos = tokenizer.get_mask_token_id(), tokenizer.get_eos_token_id()
    if noisy:
        clean, edge = prefix[:-31], prefix[-31:]
        require(len(edge) == 31 and clean, "short noisy-edge prefix")
        rng = generator_for(prompt["id"], 20261103, "boundary-noisy-edge", device="cpu")
        positions = torch.randperm(31, generator=rng)[:16].tolist()
        edge = [mask if i in positions else x for i, x in enumerate(edge)]
        active, query = edge + [mask], 31
    else:
        clean, active, query = prefix, [mask] * 32, 0
    cache = CausalPrefixKVCache(model, batch_size=1, capacity=len(clean) + len(active))
    cache.append(torch.tensor([clean], device=device))
    z = cache.logits(torch.tensor([active], device=device))[0, query]
    prob = legal_prob(z, sorted({mask, *tokenizer.get_generation_blocked_token_ids()}))
    target = eos if prompt["condition"] == "complete" else prompt["gold_ids"][0]
    logp = prob.clamp_min(torch.finfo(torch.float32).tiny).log()
    return {
        "noisy": noisy,
        "eos_top1": int(prob.argmax()) == eos,
        "eos_probability": float(prob[eos]),
        "target_nll": float(-logp[target]),
        "non_eos_target_nll": None
        if target == eos
        else float(-logp[target] + torch.log1p(-prob[eos].clamp(max=1 - 1e-7))),
    }
