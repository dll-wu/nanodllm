"""Component-specific prediction and reconstruction on the public dev32 panel."""

from __future__ import annotations

import math
import random

import numpy as np
import torch
from torch.nn import functional as F

from .paper_components_fastv2 import forward_block_logits
from .paper_eval_block import (
    BLOCK_SEEDS,
    active_logits,
    categorical_metrics,
    common_prediction,
    conditional_native_nelbo,
)
from .paper_runtime.replication_eval_support import device_of, generator_for


def _full_bound(model, row, policy, seed):
    """A full-sequence arm masks the entire continuation in one draw, including partial final blocks."""
    gold = torch.tensor(row["gold_ids"], device=device_of(model))
    prefix = gold.new_tensor([row["prefix_ids"]])
    rng = generator_for(row["id"], seed, "native_nelbo/block/0")
    draws = 16
    times = ((torch.arange(draws, dtype=torch.float64) + 0.5) / draws)[torch.randperm(draws, generator=rng)].float()
    masks = torch.rand((draws, len(gold)), generator=rng) < times[:, None]
    nats = 0.0
    for selected, t in zip(masks.split(4), times.split(4), strict=True):
        selected, t = selected.to(gold.device), t.to(gold.device)
        active = gold.expand(len(t), -1).masked_fill(selected, model.config.mask_token_id)
        logits = active_logits(model, prefix.expand(len(t), -1), active, policy, time=t)
        ce = F.cross_entropy(logits.flatten(0, 1), gold.repeat(len(t)), reduction="none").view_as(active)
        nats += float((ce * (selected / t[:, None])).sum()) / draws
    return nats


def _shifted_prediction(model, row, canvas):
    """The input to target i contains the gold prefix up to i and MASKs through the block end."""
    prefix = torch.tensor([row["prefix_ids"]], device=device_of(model))
    nats = 0.0
    for target, size in zip(row["gold_ids"][:32], row["gold_token_bytes"][:32], strict=True):
        width = 32 - prefix.shape[1] % 32 if canvas == "block_end" else 1
        active = prefix.new_full((1, width), model.config.mask_token_id)
        logits = forward_block_logits(model, prefix, active, policy={"block_size": 32})[0, 0]
        if size > 0:
            nats += categorical_metrics(logits, target)["nll_nats"]
        prefix = torch.cat((prefix, prefix.new_tensor([[target]])), dim=1)
    return nats


@torch.inference_mode()
def score_panel(model, rows, *, objective):
    if model.training or model.config.causal:
        raise ValueError("prediction requires a diffusion endpoint in evaluation mode")
    topology = {
        "F1": "full",
        "F2": "full",
        "F3": "block_bidirectional",
        "F4": "block_bidirectional",
        "B1": "causal_prefix",
        "B2": "causal_prefix",
        "native-bl32": "causal_prefix",
        "B3": "fast_v2",
    }[objective]
    policy = {
        "topology": topology,
        "block_length": 32,
        "clean_stream_mode": "causal" if objective == "native-bl32" else "block_causal",
        "exclude_mask_logit": objective in {"F1", "F3", "F4"},
    }
    result = []
    for row in rows:
        sizes = row["gold_token_bytes"]
        if not 32 <= len(sizes) == len(row["gold_ids"]) <= 128 or len(row["prefix_ids"]) != 192:
            raise ValueError("expected a 192-token prefix and 32 to 128 continuation tokens")
        common = {}
        native = {}
        if objective == "B3":
            # Both canvases pad to the block end. Preserve the number of projected logits for numerical replay.
            common = {canvas: _shifted_prediction(model, row, canvas) for canvas in ("legacy", "block_end")}
        else:
            for canvas in ("legacy", "block_end"):
                scores = common_prediction(model, row, policy={**policy, "common_canvas": canvas})
                common[canvas] = scores["aggregate"]["content_nll_nats"]
            for seed in BLOCK_SEEDS:
                if topology == "full":
                    nats = _full_bound(model, row, policy, seed)
                else:
                    nats = conditional_native_nelbo(model, row, policy=policy, seed=seed, draws=16, batch_size=4)[
                        "nats"
                    ]
                native[str(seed)] = nats
        result.append(
            {
                "id": row["id"],
                "content_bytes": sum(sizes[:32]),
                "common_nats": common,
                "native_bytes": sum(sizes),
                "native_nats": native or None,
            }
        )
    if not result or len({r["id"] for r in result}) != len(result):
        raise ValueError("expected distinct panel documents")
    total_bytes = sum(r["content_bytes"] for r in result)
    return {
        "documents": result,
        "targets": len(result) * 32,
        "content_bytes": total_bytes,
        "common_bpb": {
            c: sum(r["common_nats"][c] for r in result) / (math.log(2) * total_bytes) for c in ("block_end", "legacy")
        },
        "native_bpb_by_seed": None
        if objective == "B3"
        else {
            str(s): sum(r["native_nats"][str(s)] for r in result)
            / (math.log(2) * sum(r["native_bytes"] for r in result))
            for s in BLOCK_SEEDS
        },
    }


def compare(candidate, control):
    """Use each paper contrast's document bootstrap, paired across arms and fitted seeds."""
    pairs = {
        ("bd3/calibrated", "bd3/fixed"),
        ("llada1/masking", "llada1/mdlm"),
        ("llada2/curriculum", "llada2/fixed"),
        ("fastv2/full", "fastv2/token-causal"),
        ("fastv2/full", "fastv2/block-causal"),
        ("fastv2/matched", "fastv2/block-causal"),
    }
    docs = sizes = contrast = panel_id = None
    diffs = []
    for a, b, seed in zip(
        sorted(candidate, key=lambda r: r["seed"]), sorted(control, key=lambda r: r["seed"]), (48, 49), strict=True
    ):
        pair = (a["arm"], b["arm"])
        if (
            pair not in pairs
            or (contrast and pair != contrast)
            or (a["seed"], b["seed"]) != (seed, seed)
            or a["ar_parent_sha256"] != b["ar_parent_sha256"]
        ):
            raise ValueError("expected a reference contrast with matched seeds and AR parent")
        if a["phase"] != b["phase"] or a["phase"] != ("tail" if pair[0].startswith("llada2/") else "main"):
            raise ValueError("expected matched endpoint phases")
        if a["panel_sha256"] != b["panel_sha256"] or (panel_id and panel_id != a["panel_sha256"]):
            raise ValueError("scored panels differ")
        panel_id, contrast = a["panel_sha256"], pair
        indexed = [{r["id"]: r for r in side["documents"]} for side in (a, b)]
        ids = sorted(indexed[0])
        if (
            not ids
            or ids != sorted(indexed[1])
            or (docs and ids != docs)
            or any(len(idx) != len(side["documents"]) for idx, side in zip(indexed, (a, b), strict=True))
        ):
            raise ValueError("document identities differ")
        counts = [indexed[0][i]["content_bytes"] for i in ids]
        if min(counts) <= 0 or counts != [indexed[1][i]["content_bytes"] for i in ids] or (sizes and counts != sizes):
            raise ValueError("document byte denominators differ")
        docs, sizes = ids, counts
        diffs.append(
            [indexed[0][i]["common_nats"]["block_end"] - indexed[1][i]["common_nats"]["block_end"] for i in ids]
        )
    n = len(docs)
    if contrast == ("fastv2/full", "fastv2/block-causal"):
        rng = random.Random(20261106)
        indices = np.array([rng.choices(range(n), k=n) for _ in range(10000)])
    else:
        indices = np.random.default_rng(20261106).integers(0, n, size=(10000, n))
    delta = np.array(diffs)
    denom = math.log(2) * np.array(sizes)
    by_seed = delta.sum(axis=1) / denom.sum()
    samples = delta.mean(axis=0)[indices].sum(axis=1) / denom[indices].sum(axis=1)
    return {
        "candidate": contrast[0],
        "control": contrast[1],
        "seeds": [48, 49],
        "delta_bpb": by_seed.tolist(),
        "mean_delta_bpb": float(by_seed.mean()),
        "paired_document_95": np.quantile(samples, [0.025, 0.975]).tolist(),
    }
