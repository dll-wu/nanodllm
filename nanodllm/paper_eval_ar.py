"""Exact shifted AR likelihood on frozen, document-segmented gold validation rows."""

import math

import torch
from torch.nn import functional as F

from .paper_eval_panels import SEGMENT_ATTENTION, FrozenInputs
from .paper_runtime.core_evaluation_cells import require
from .paper_runtime.data import build_token_bytes

COUNTS = (
    "target_tokens",
    "content_tokens",
    "content_bytes",
    "content_top1_hits",
    "eos_tokens",
    "eos_top1_hits",
    "eos_predictions_on_content",
)
NATS = ("target_nats", "content_nats", "eos_nats")
METRIC = "causal_ar_content_ce_bits_per_byte"


def summarize_rows(rows: list[dict], *, count: int, length: int) -> dict:
    require(len(rows) == count and count > 0, "AR row coverage differs")
    for i, row in enumerate(rows):
        require(
            set(row) == {"sequence_index", *COUNTS, *NATS} and row["sequence_index"] == i, "AR row identity differs"
        )
        require(all(type(row[k]) is int and row[k] >= 0 for k in COUNTS), "invalid AR counts")
        require(
            all(type(row[k]) in (float, int) and math.isfinite(row[k]) and row[k] >= 0 for k in NATS),
            "nonfinite AR loss",
        )
        require(
            row["content_tokens"] + row["eos_tokens"] <= row["target_tokens"] <= length - 1
            and row["content_top1_hits"] <= row["content_tokens"] <= row["content_bytes"]
            and row["eos_top1_hits"] <= row["eos_tokens"]
            and row["eos_predictions_on_content"] <= row["content_tokens"]
            and row["content_nats"] + row["eos_nats"] <= row["target_nats"] + 1e-8,
            "AR target accounting differs",
        )
    totals = {k: sum(r[k] for r in rows) for k in COUNTS}
    totals.update({k: math.fsum(r[k] for r in rows) for k in NATS})
    require(totals["content_bytes"] > 0 and totals["content_tokens"] > 0, "empty AR content denominator")
    return {
        "metric": METRIC,
        "sequence_count": count,
        "sequence_length": length,
        **totals,
        "content_bpb": totals["content_nats"] / (math.log(2) * totals["content_bytes"]),
        "content_top1_accuracy": totals["content_top1_hits"] / totals["content_tokens"],
        "eos_ce_nats": totals["eos_nats"] / totals["eos_tokens"] if totals["eos_tokens"] else None,
        "eos_top1_accuracy": totals["eos_top1_hits"] / totals["eos_tokens"] if totals["eos_tokens"] else None,
        "eos_prediction_rate_on_content": totals["eos_predictions_on_content"] / totals["content_tokens"],
        "scope": "Exact gold-prefix AR likelihood on this finite segmented panel; row0 and BOS are unscored. EOS is separate from content BPB. No diffusion Monte Carlo, rollout or cross-family NELBO comparison.",
    }


@torch.inference_mode()
def score_corpus(model, tokenizer, inputs: FrozenInputs, *, db: int, device: str) -> dict:
    require(
        not model.training and model.config.causal and model.config.bd3lm_block_size == 0,
        "native causal AR model required",
    )
    require(inputs.metadata["attention_layout"] == SEGMENT_ATTENTION, "segmented gold inputs required")
    count, length = (inputs.metadata[k] for k in ("sequence_count", "sequence_length"))
    require(length <= model.config.sequence_len, "AR evaluation exceeds trained context")
    token_bytes = build_token_bytes(tokenizer).to(device)
    bos, eos, mask = tokenizer.get_bos_token_id(), tokenizer.get_eos_token_id(), tokenizer.get_mask_token_id()
    require(bos is not None and eos is not None, "AR boundary tokenizer required")
    rows, forwards = [], 0
    for batch in inputs.batches(db, torch.device(device)):
        x = batch.x_0
        require(bool((x[batch.attention_mask] != mask).all()), "MASK in clean AR input")
        require(int(x.max()) < model.config.vocab_size, "AR input outside vocabulary")
        target_mask = batch.loss_mask[:, 1:] & batch.attention_mask[:, 1:]
        require(
            not bool((target_mask & ((x[:, 1:] == bos) | (batch.position_ids[:, 1:] == 0))).any()),
            "AR target crosses a document boundary",
        )
        require(bool((target_mask <= batch.attention_mask[:, :-1]).all()), "AR target has no visible predecessor")
        row_ids, predecessor = target_mask.nonzero(as_tuple=True)
        gold = x[row_ids, predecessor + 1]
        if gold.numel():
            logits = model(
                x,
                logits_indices=row_ids * length + predecessor,
                attention_mask=batch.document_boundary_mask,
                position_ids=batch.position_ids,
            ).float()
            require(
                logits.shape == (gold.numel(), model.config.vocab_size) and bool(torch.isfinite(logits).all()),
                "invalid AR logits",
            )
            ce = F.cross_entropy(logits, gold, reduction="none").double()
            pred = logits.argmax(dim=-1)
            forwards += 1
        else:
            ce = torch.empty(0, dtype=torch.float64, device=device)
            pred = gold
        byte_counts = token_bytes[gold]
        content, ending = byte_counts > 0, gold == eos
        for index in range(x.shape[0]):
            selected = row_ids == index
            lexical, eos_targets = selected & content, selected & ending
            rows.append(
                {
                    "sequence_index": len(rows),
                    "target_tokens": int(selected.sum()),
                    "target_nats": float(ce[selected].sum()),
                    "content_tokens": int(lexical.sum()),
                    "content_bytes": int(byte_counts[lexical].sum()),
                    "content_nats": float(ce[lexical].sum()),
                    "content_top1_hits": int(((pred == gold) & lexical).sum()),
                    "eos_tokens": int(eos_targets.sum()),
                    "eos_nats": float(ce[eos_targets].sum()),
                    "eos_top1_hits": int(((pred == eos) & eos_targets).sum()),
                    "eos_predictions_on_content": int(((pred == eos) & lexical).sum()),
                }
            )
    return {"summary": summarize_rows(rows, count=count, length=length), "rows": rows, "model_forwards": forwards}
