"""Gold-prefix categorical and native conditional reconstruction scores on BL32 panels."""

import math

import torch
import torch.nn.functional as F

from .paper_runtime.replication_eval_support import device_of, generator_for

BLOCK_SEEDS = (2026110201, 2026110202)


def token_ids(values, *, name, vocab_size, allow_empty=False):
    if not isinstance(values, (list, tuple)) or (not values and not allow_empty):
        raise ValueError(f"{name} must contain token IDs")
    if any(type(x) is not int or not 0 <= x < vocab_size for x in values):
        raise ValueError(f"{name} contains an invalid token ID")
    return list(values)


def visibility(length, prefix_length, *, topology, block_length, device):
    """No active/future information enters completed-prefix states."""
    positions = torch.arange(length, device=device)
    q, k = positions[:, None], positions[None, :]
    if topology == "full":
        return torch.ones((1, length, length), dtype=torch.bool, device=device)
    if topology == "causal_ar":
        return (q >= k)[None]
    if topology == "causal_prefix":
        return torch.where(q < prefix_length, k <= q, torch.ones_like(q >= k))[None]
    if topology == "block_bidirectional":
        # Known tokens inside the CURRENT block are noised-stream tokens with
        # visible gold content, not completed-prefix states. They must still
        # attend to the unresolved current block, exactly as in BD3 training.
        completed = prefix_length // block_length * block_length
        clean = (q // block_length >= k // block_length) & (k < completed)
        return torch.where(q < completed, clean, torch.ones_like(clean))[None]
    raise ValueError("shifted or unknown topology requires its explicit adapter")


def active_logits(model, prefix, active, policy, *, time=None):
    """Differentiable [B,active,V] logits; inputs contain NO hidden future gold.

    prefix and active are separate so shifted predictors and clean-prefix
    attention cannot accidentally infer the boundary from MASK positions.
    """
    if prefix.ndim != 2 or active.ndim != 2 or prefix.shape[0] != active.shape[0] or not active.shape[1]:
        raise ValueError("prefix/active must be rectangular matching batches")
    if policy["topology"] == "causal_ar":
        raise ValueError("AR uses next-token shifted indexing, not active-position logits")
    tokens = torch.cat((prefix, active), dim=1)
    if tokens.shape[1] > model.config.sequence_len:
        raise ValueError("evaluation exceeds model context; no implicit truncation")
    attention = visibility(
        tokens.shape[1],
        prefix.shape[1],
        topology=(
            "block_bidirectional"
            if policy["topology"] == "causal_prefix" and policy.get("clean_stream_mode") == "block_causal"
            else policy["topology"]
        ),
        block_length=policy["block_length"],
        device=tokens.device,
    )
    logits = model(tokens, attention_mask=attention, noise_level=time)
    if not isinstance(logits, torch.Tensor) or logits.shape[:2] != tokens.shape:
        raise ValueError("model must return position-aligned logits")
    result = logits[:, prefix.shape[1] :].float()
    if policy.get("exclude_mask_logit", False):
        result = result.clone()
        result[..., model.config.mask_token_id] = -torch.inf
    return result


def next_token_logits(model, prefix, policy):
    if prefix.ndim != 2 or prefix.shape[1] < 1:
        raise ValueError("next-token prediction needs a nonempty clean prefix")
    if policy["topology"] == "causal_ar":
        length = prefix.shape[1]
        attention = visibility(length, length, topology="causal_ar", block_length=1, device=prefix.device)
        return model(prefix, attention_mask=attention)[:, -1].float()
    count = 1
    if policy.get("common_canvas") == "block_end":
        count = policy["block_length"] - prefix.shape[1] % policy["block_length"]
    active = prefix.new_full((prefix.shape[0], count), model.config.mask_token_id)
    return active_logits(model, prefix, active, policy)[:, 0]


def categorical_metrics(logits, target):
    """Raw-vocabulary probabilities; ties get competition rank, separately tagged."""
    if logits.ndim != 1 or not 0 <= target < len(logits):
        raise ValueError("invalid categorical target")
    if (
        bool(torch.isnan(logits).any())
        or not bool(torch.isfinite(logits.max()))
        or not bool(torch.isfinite(logits[target]))
    ):
        raise ValueError("invalid categorical logit support")
    logits = logits.float()
    rank = int((logits > logits[target]).sum()) + 1
    return {
        "nll_nats": float(-logits.log_softmax(-1)[target]),
        "rank": rank,
        "mrr": 1 / rank,
        "hit1": rank == 1,
        "hit5": rank <= 5,
        "hit10": rank <= 10,
        "tied_with_gold": int((logits == logits[target]).sum()) - 1,
    }


@torch.inference_mode()
def common_prediction(model, row, *, policy, common_tokens=32):
    prefix = token_ids(row["prefix_ids"], name="prefix", vocab_size=model.config.vocab_size)
    targets = token_ids(row["gold_ids"], name="gold", vocab_size=model.config.vocab_size)
    token_bytes = row["gold_token_bytes"]
    if len(token_bytes) != len(targets) or any(type(n) is not int or n < 0 for n in token_bytes):
        raise ValueError("gold byte counts must match targets")
    if type(common_tokens) is not int or common_tokens < 1:
        raise ValueError("common_tokens must be a positive integer")
    count = min(common_tokens, len(targets))
    if len(prefix) + count > model.config.sequence_len:
        raise ValueError("common next-token panel exceeds context")
    positions = []
    for offset in range(count):
        # The next unknown token never enters its own input, nor does any future
        # gold. Previously scored targets form the teacher-forced clean past.
        context = torch.tensor([prefix + targets[:offset]], device=device_of(model))
        logits = next_token_logits(model, context, policy)[0]
        score = categorical_metrics(logits, targets[offset])
        positions.append(
            {
                **score,
                "offset": offset,
                "target": targets[offset],
                "target_bytes": token_bytes[offset],
                "prefix_tokens": len(prefix) + offset,
                "content_bpb": score["nll_nats"] / (math.log(2) * token_bytes[offset]) if token_bytes[offset] else None,
            }
        )
    content = [position for position in positions if position["target_bytes"] > 0]
    nbytes = sum(position["target_bytes"] for position in content)
    content_nll = sum(position["nll_nats"] for position in content)
    return {
        **positions[0],
        "positions": positions,
        "aggregate": {
            "targets": count,
            "nll_nats": sum(p["nll_nats"] for p in positions),
            "content_nll_nats": content_nll,
            "content_bytes": nbytes,
            "content_bpb": content_nll / (math.log(2) * nbytes) if nbytes else None,
            "mrr": sum(p["mrr"] for p in positions) / count,
            "hit1": sum(p["hit1"] for p in positions) / count,
        },
        "metric": "common_left_context_single_token",
        "teacher_forced_positions": count,
        "native_likelihood": False,
    }


@torch.inference_mode()
def conditional_native_nelbo(model, row, *, policy, seed, draws=16, batch_size=8):
    """Linear absorbing conditional reference integral over full (0,1).

    Uniform midpoint strata with independent mask realizations estimate
    E_t[sum_i 1[z_i=MASK] CE_i/t]. Block arms condition on clean *past*
    gold blocks, never future blocks. This is quadrature/Monte Carlo, not exact
    log likelihood. It does not evaluate the training-specific BD3 truncation.
    """
    if type(draws) is not int or draws < 1 or type(batch_size) is not int or batch_size < 1:
        raise ValueError("estimator draws and batch size must be positive")
    vocab, device = model.config.vocab_size, device_of(model)
    prefix = token_ids(row["prefix_ids"], name="prefix", vocab_size=vocab)
    gold = token_ids(row["gold_ids"], name="gold", vocab_size=vocab)
    byte_counts = row["gold_token_bytes"]
    if len(byte_counts) != len(gold) or any(type(x) is not int or x < 0 for x in byte_counts) or sum(byte_counts) <= 0:
        raise ValueError("native bound needs positive matched continuation byte accounting")
    block = int(policy["block_length"])
    if len(prefix) % block:
        raise ValueError("native block bound requires block-aligned clean prefix")
    if len(prefix) + len(gold) > model.config.sequence_len:
        raise ValueError("native bound exceeds trained context")
    weighted_sum, content_sum, calls, query_tokens = 0.0, 0.0, 0, 0
    for start in range(0, len(gold), block):
        clean = torch.tensor(gold[start : start + block], device=device)
        context = torch.tensor([prefix + gold[:start]], device=device)
        bytes_here = torch.tensor(byte_counts[start : start + block], device=device)
        times = (torch.arange(draws, dtype=torch.float64) + 0.5) / draws
        rng = generator_for(row["id"], seed, f"native_nelbo/block/{start}")
        times = times[torch.randperm(draws, generator=rng)].float()
        masks = torch.rand((draws, len(clean)), generator=rng) < times[:, None]
        # Draw the entire plan before device batching: DB cannot change masks.
        for offset in range(0, draws, batch_size):
            selected = masks[offset : offset + batch_size].to(device)
            t = times[offset : offset + batch_size].to(device)
            active = clean[None].expand(len(t), -1).clone().masked_fill(selected, model.config.mask_token_id)
            logits = active_logits(model, context.expand(len(t), -1), active, policy, time=t)
            ce = F.cross_entropy(logits.reshape(-1, vocab), clean.repeat(len(t)), reduction="none").view_as(active)
            weights = selected / t[:, None]
            weighted_sum += float((ce * weights).sum()) / draws
            content_sum += float((ce * weights * bytes_here.gt(0)[None]).sum()) / draws
            calls += 1
            query_tokens += len(t) * (context.shape[1] + len(clean))
    denominator = math.log(2) * sum(byte_counts)
    return {
        "status": "ok",
        "metric": "conditional_absorbing_reference_nelbo",
        "nats": weighted_sum,
        "content_nats": content_sum,
        "bytes": sum(byte_counts),
        "bpb": weighted_sum / denominator,
        "content_bpb": content_sum / denominator,
        "is_exact_likelihood": False,
        "schedule": "linear_absorbing_t0_clean_t1_mask",
        "integration_interval": [0, 1],
        "time_sampling": "per_block_permuted_midpoints",
        "draws_per_block": draws,
        "bound_estimate_scope": "gold continuation conditional on frozen clean prefix; full-support reference",
        "content_only_is_bound": False,
        "training_noise_interval_not_reused": True,
        "model_forwards": calls,
        "processed_query_tokens": query_tokens,
    }


def block_scores(loaded, prompts):
    model = loaded.model
    if model.config.causal and not model.config.bd3lm_block_size:
        topology = "causal_ar"
    elif not model.config.causal and model.config.bd3lm_block_size == 32 and loaded.noise_type == "bd3lm":
        topology = "causal_prefix"
    else:
        raise ValueError("block panels require a causal AR parent or a native BL32 diffusion model")
    if loaded.clean_stream_mode not in ("causal", "block_causal"):
        raise ValueError("unknown clean stream attention")
    policy = dict(topology=topology, block_length=32, clean_stream_mode=loaded.clean_stream_mode)
    records = []
    for prompt in prompts:
        metrics = {}
        for canvas in ("legacy", "block_end"):
            values = common_prediction(model, prompt, policy={**policy, "common_canvas": canvas})["aggregate"]
            metrics["common/" + canvas] = [values["content_nll_nats"], values["content_bytes"]]
        if topology != "causal_ar":
            estimates = []
            for seed in BLOCK_SEEDS:
                value = conditional_native_nelbo(model, prompt, policy=policy, seed=seed, draws=16, batch_size=4)
                metrics[f"native/{seed}"] = [value["nats"], value["bytes"]]
                estimates.append(value["nats"])
            metrics["native/mean_eval_seeds"] = [sum(estimates) / len(estimates), sum(prompt["gold_token_bytes"])]
        records.append(dict(id=prompt["id"], stratum=prompt["length_stratum"], metrics=metrics))
    return records
