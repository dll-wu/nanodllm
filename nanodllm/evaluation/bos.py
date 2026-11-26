"""Document-start panel selection and paired full-sequence corruption scoring."""

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from ..runtime.core_evaluation_cells import require
from ..runtime.core_task_data import json_digest
from ..runtime.data import build_token_bytes
from ..runtime.nelbo_time_sampling import build_nelbo_time_plan
from ..runtime.noise import MDLMNoise
from ..runtime.training_objective import _sequence_major_nelbo_corruption_plan

CONDITIONS = ("document_start", "truncated", "truncated_bos")
SCOPES = ("full_content", "matched_suffix")


def resolve_conditions(conditions: Sequence[str] | None) -> tuple[str, ...]:
    """Keep the historical roster unless a nonempty, ordered subset is explicit."""
    if conditions is None:
        return CONDITIONS
    require(isinstance(conditions, Sequence) and not isinstance(conditions, str), "invalid context roster")
    selected = tuple(conditions)
    require(
        bool(selected)
        and all(isinstance(c, str) and c in CONDITIONS for c in selected)
        and len(set(selected)) == len(selected),
        "invalid context roster",
    )
    return selected


def prepare_panel(inputs, tokenizer, settings, *, conditions: Sequence[str] | None = None):
    """Select all genuine starts; only requested stress contexts require a suffix.

    Condition selection is not part of the panel identity: the same source and
    settings must produce identical targets and noise plans across campaigns.
    """
    conditions = resolve_conditions(conditions)
    require(type(settings["drop_prefix_tokens"]) is int and settings["drop_prefix_tokens"] >= 0, "invalid prefix drop")
    t = inputs.tensors
    token_bytes = build_token_bytes(tokenizer)
    bos, eos = tokenizer.get_bos_token_id(), tokenizer.get_eos_token_id()
    examples = []
    counts = {"genuine_segments": 0, "excluded_without_content": 0}
    for row, tokens in enumerate(t["x_0"]):
        starts = torch.where((t["position_ids"][row] == 0) & t["attention_mask"][row])[0].tolist()
        for start, end in zip(starts, starts[1:] + [len(tokens)], strict=True):
            genuine = bool(t["document_start_mask"][row, start])
            require(genuine == (int(tokens[start]) == bos), "BOS/start metadata disagreement")
            if not genuine:
                continue
            counts["genuine_segments"] += 1
            x, loss = tokens[start:end], t["loss_mask"][row, start:end]
            require(bool(t["attention_mask"][row, start:end].all()), "invisible source content")
            require(not bool(loss[0]) and not bool((x[1:] == bos).any()), "interior or lossable BOS")
            require(not bool((x[:-1] == eos).any()), "interior EOS in document segment")
            require(bool(((token_bytes[x] <= 0) | loss).all()), "unscored source content")
            require(bool(((token_bytes[x] > 0) | (x == bos) | (x == eos)).all()), "unexpected special token")
            if not bool((token_bytes[x] > 0).any()):
                counts["excluded_without_content"] += 1
                continue
            common_start = 1 + settings["drop_prefix_tokens"] + 1
            paired_bytes = int(token_bytes[x[common_start:]][loss[common_start:]].sum())
            examples.append(
                {
                    "id": len(examples),
                    "source_row": row,
                    "source_start": start,
                    "source_end": end,
                    "tokens": x.tolist(),
                    "loss_mask": loss.tolist(),
                    "content_bytes": int(token_bytes[x].sum()),
                    "content_tokens": int((token_bytes[x] > 0).sum()),
                    "suffix_bytes": paired_bytes,
                    "suffix_tokens": int(((token_bytes[x[common_start:]] > 0) & loss[common_start:]).sum()),
                    "common_target_sha256": json_digest(
                        {"tokens": x[common_start:].tolist(), "loss_mask": loss[common_start:].tolist()}
                    ),
                }
            )
    require(bool(examples), "empty document/suffix panel")
    require(
        conditions == ("document_start",) or any(e["suffix_bytes"] > 0 for e in examples),
        "empty document/suffix panel",
    )
    counts.update(
        documents=len(examples),
        source_tokens=sum(len(e["tokens"]) for e in examples),
        content_tokens=sum(e["content_tokens"] for e in examples),
        content_bytes=sum(e["content_bytes"] for e in examples),
        suffix_documents=sum(e["suffix_bytes"] > 0 for e in examples),
        suffix_tokens=sum(e["suffix_tokens"] for e in examples),
        suffix_bytes=sum(e["suffix_bytes"] for e in examples),
        min_length=min(len(e["tokens"]) for e in examples),
        max_length=max(len(e["tokens"]) for e in examples),
    )
    panel = {
        "settings": settings,
        "source_input_root": inputs.metadata["input_identity_root_sha256"],
        "counts": counts,
        "examples": examples,
    }
    return {**panel, "panel_sha256": json_digest(panel)}


def prepare_batch(examples, condition, tokenizer, *, drop):
    require(condition in CONDITIONS, "unknown document context")
    rows = []
    for e in examples:
        offset = 0 if condition == "document_start" else drop + 1
        require(offset == 0 or e["suffix_bytes"] > 0, "empty stress-test suffix")
        tokens = e["tokens"][offset:]
        loss = e["loss_mask"][offset:]
        coords = list(range(offset, len(e["tokens"])))
        if condition == "truncated_bos":
            tokens, loss, coords = [tokenizer.get_bos_token_id()] + tokens, [False] + loss, [-1] + coords
        rows.append((tokens, loss, coords))
    length = max(len(r[0]) for r in rows)
    shape = len(rows), length
    x = torch.full(shape, tokenizer.get_pad_token_id(), dtype=torch.long)
    loss = torch.zeros(shape, dtype=torch.bool)
    visible = torch.zeros(shape, dtype=torch.bool)
    origin = torch.full(shape, -1, dtype=torch.long)
    for i, (tokens, targets, coords) in enumerate(rows):
        n = len(tokens)
        x[i, :n], loss[i, :n] = torch.tensor(tokens), torch.tensor(targets, dtype=torch.bool)
        visible[i, :n], origin[i, :n] = True, torch.tensor(coords)
    return {
        "gold": x,
        "loss_mask": loss,
        "visible": visible,
        "origin": origin,
        "positions": torch.arange(length).expand(len(rows), -1),
        "matched_suffix": loss & (origin >= drop + 2),
    }


COUNTS = (
    "content_tokens",
    "content_bytes",
    "scored_content_tokens",
    "content_hits",
    "eos_predictions_on_content",
    "eos_tokens",
    "scored_eos_tokens",
)
NATS = ("content_nats", "masked_content_nats", "eos_nats")


def metric_name(causal, scope):
    if causal:
        return "ar_exact_content_bpb"
    return (
        "mdlm_sampled_nelbo_content_bpb" if scope == "full_content" else "mdlm_matched_suffix_integrated_denoising_bpb"
    )


def summarize(rows):
    require(bool(rows) and len({r["id"] for r in rows}) == len(rows), "empty/duplicate corpus rows")
    for row in rows:
        require(all(type(row[k]) is int and row[k] >= 0 for k in COUNTS), "invalid counts")
        require(all(math.isfinite(row[k]) and row[k] >= 0 for k in NATS), "invalid nats")
        require(
            row["content_hits"] <= row["scored_content_tokens"] <= row["content_tokens"] <= row["content_bytes"],
            "invalid content counts",
        )
        require(
            row["eos_predictions_on_content"] <= row["scored_content_tokens"]
            and row["scored_eos_tokens"] <= row["eos_tokens"],
            "invalid EOS counts",
        )
    result = {k: sum(r[k] for r in rows) for k in COUNTS}
    result.update({k: math.fsum(r[k] for r in rows) for k in NATS})
    require(result["content_bytes"] > 0, "empty byte denominator")
    result.update(
        documents=len(rows),
        bpb=result["content_nats"] / (math.log(2) * result["content_bytes"]),
        content_accuracy=result["content_hits"] / result["scored_content_tokens"]
        if result["scored_content_tokens"]
        else None,
        eos_prediction_rate_on_scored_content=result["eos_predictions_on_content"] / result["scored_content_tokens"]
        if result["scored_content_tokens"]
        else None,
        weighted_eos_nats_per_eos=result["eos_nats"] / result["eos_tokens"] if result["eos_tokens"] else None,
    )
    return result


def corruption_plan(panel, seed):
    settings = panel["settings"]
    count = len(panel["examples"])
    times = build_nelbo_time_plan(count, 0, 1, seed, settings["time_sampling"])
    draws, derived = _sequence_major_nelbo_corruption_plan(count, settings["sequence_length"], seed)
    import hashlib

    return (
        times.times,
        draws,
        {
            **times.metadata(),
            "mask_uniforms_sha256": hashlib.sha256(draws.numpy().astype("<f4", copy=False).tobytes()).hexdigest(),
            "derived_mask_seed": derived,
            "coordinate_rule": "original document coordinates, shared across all contexts and checkpoints",
        },
    )


def corrupt_batch(batch, examples, times, uniforms, tokenizer):
    ids = torch.tensor([e["id"] for e in examples])
    coords = batch["origin"].clamp_min(0)
    draws = uniforms[ids[:, None], coords]
    return MDLMNoise(tokenizer.get_mask_token_id()).corrupt(
        batch["gold"], times[ids], maskable=batch["loss_mask"], uniforms=draws
    )


def row_records(examples, batch, active, nats, ce, predictions, token_bytes, times, scope):
    gold = batch["gold"]
    selected = batch["loss_mask"] if scope == "full_content" else batch["matched_suffix"]
    content, eos = selected & (token_bytes[gold] > 0), selected & (gold == batch["eos_id"])
    result = []
    for i, e in enumerate(examples):
        if scope == "matched_suffix" and not e["suffix_bytes"]:
            continue
        lexical, ending = content[i], eos[i]
        scored = lexical & active[i]
        common_active = active[i][batch["matched_suffix"][i]].tolist()
        result.append(
            {
                "id": e["id"],
                "time": float(times[e["id"]]) if times is not None else None,
                "common_target_sha256": e["common_target_sha256"],
                "common_mask_sha256": json_digest(common_active),
                "content_tokens": int(lexical.sum()),
                "content_bytes": int(token_bytes[gold[i]][lexical].sum()),
                "scored_content_tokens": int(scored.sum()),
                "content_hits": int(((predictions[i] == gold[i]) & scored).sum()),
                "eos_predictions_on_content": int(((predictions[i] == batch["eos_id"]) & scored).sum()),
                "eos_tokens": int(ending.sum()),
                "scored_eos_tokens": int((ending & active[i]).sum()),
                "content_nats": float(nats[i][lexical].sum()),
                "masked_content_nats": float(ce[i][scored].sum()),
                "eos_nats": float(nats[i][ending].sum()),
            }
        )
    return result


@torch.inference_mode()
def score_corpus(model, tokenizer, panel, *, seed, device, db, conditions: Sequence[str] | None = None):
    """Score selected contexts without changing targets, random plans or metrics.

    Omitting conditions preserves the D20 report. Primary-only scoring retains
    its identical document_start cell, including the suffix diagnostic when one
    exists, but makes no truncated-context forwards.
    """
    conditions = resolve_conditions(conditions)
    require(type(db) is int and db > 0, "device batch must be a positive integer")
    require(not model.training and not model.config.bd3lm_block_size, "native frozen model required")
    causal = bool(model.config.causal)
    require((seed is None) == causal, "AR/MDLM seed mismatch")
    token_bytes = build_token_bytes(tokenizer)
    has_suffix = any(e["suffix_bytes"] > 0 for e in panel["examples"])
    require(conditions == ("document_start",) or has_suffix, "empty document/suffix panel")
    times, uniforms, receipt = (None, None, None) if causal else corruption_plan(panel, seed)
    result, forwards = {}, 0
    for condition in conditions:
        examples = [e for e in panel["examples"] if condition == "document_start" or e["suffix_bytes"] > 0]
        records = {"matched_suffix": []} if has_suffix else {}
        if condition == "document_start":
            records["full_content"] = []
        for start in range(0, len(examples), db):
            selected = examples[start : start + db]
            batch = prepare_batch(selected, condition, tokenizer, drop=panel["settings"]["drop_prefix_tokens"])
            gold = batch["gold"]
            require(gold.shape[1] <= model.config.sequence_len, "context exceeded")
            x, active = (
                (gold, batch["loss_mask"].clone())
                if causal
                else corrupt_batch(batch, selected, times, uniforms, tokenizer)
            )
            active = active.bool()
            if condition != "document_start":
                active &= batch["matched_suffix"]
            require(not causal or not bool(active[:, 0].any()), "AR target has no predecessor")
            r, target = active.nonzero(as_tuple=True)
            ce = torch.zeros_like(gold, dtype=torch.float64)
            predictions = torch.full_like(gold, -1)
            if r.numel():
                indices = r * x.shape[1] + target - int(causal)
                logits = model(
                    x.to(device),
                    attention_mask=batch["visible"].to(device),
                    position_ids=batch["positions"].to(device),
                    logits_indices=indices.to(device),
                ).float()
                require(bool(torch.isfinite(logits).all()) and logits.shape[0] == len(r), "invalid logits")
                ce[r, target] = F.cross_entropy(logits, gold[r, target].to(device), reduction="none").double().cpu()
                predictions[r, target] = logits.argmax(-1).cpu()
                forwards += 1
            nats = ce if causal else ce / times[torch.tensor([e["id"] for e in selected])][:, None].double()
            batch["eos_id"] = tokenizer.get_eos_token_id()
            for scope in records:
                records[scope].extend(
                    row_records(selected, batch, active, nats, ce, predictions, token_bytes, times, scope)
                )
        result[condition] = {
            scope: {"rows": rows, "summary": {"metric": metric_name(causal, scope), **summarize(rows)}}
            for scope, rows in records.items()
        }
    return {
        "conditions": result,
        "noise_plan": receipt,
        "model_forwards": forwards,
        "metric": "ar_exact_content_bpb" if causal else "mdlm_sampled_nelbo_content_bpb",
    }
