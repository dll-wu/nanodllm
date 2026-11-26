"""Score a replayed panel and retain one numerator and denominator per sampling unit."""

import math
from contextlib import nullcontext

import torch

from .paper_eval_ar import score_corpus as score_ar
from .paper_eval_block import BLOCK_SEEDS, block_scores
from .paper_eval_bos import score_corpus as score_bos
from .paper_eval_panels import FULL_SEEDS, FrozenInputs, reference_manifest
from .paper_runtime.core_evaluation_cells import require
from .paper_runtime.eval import evaluate_nelbo
from .paper_runtime.nelbo_time_sampling import build_nelbo_time_plan
from .paper_runtime.noise import MDLMNoise
from .paper_runtime.training_objective import _sequence_major_nelbo_corruption_plan

REPORT_SCHEMA = "nanodllm.paper.corpus-scores.v1"


def packed_nelbo(model, tokenizer, inputs, seed, db, device):
    count, width = inputs.tensors["x_0"].shape
    time_plan = build_nelbo_time_plan(count, 0, 1, seed)
    uniforms, _ = _sequence_major_nelbo_corruption_plan(count, width, seed)
    noise = MDLMNoise(tokenizer.get_mask_token_id())
    offset = 0

    def corrupt(x, loss):
        nonlocal offset
        stop = offset + x.shape[0]
        t, u = time_plan.times[offset:stop].to(device), uniforms[offset:stop]
        offset = stop
        z, active = noise.corrupt(x, t, maskable=loss, uniforms=u.to(device))
        return z, t, active, None, {"corruption_uniforms": u}

    result = evaluate_nelbo(
        model,
        None,
        tokenizer,
        corrupt,
        db,
        device,
        seq_len=width,
        eval_tokens=count * width,
        schedule_weight=True,
        return_diagnostics=True,
        batches=inputs.batches(db, device),
    )
    stats = result[-1]
    receipt = {
        k: stats[k]
        for k in ("input_identity_root_sha256", "corruption_uniform_root_sha256", "corruption_mask_root_sha256")
    }
    receipt.update(time_plan.metadata())
    return [(r["nelbo_nats_numerator"], int(r["byte_denominator"])) for r in stats["sequence_records"]], receipt


def sequence_scores(loaded, panel, db, device):
    model, tokenizer = loaded.model, loaded.tokenizer
    kind, data = panel["kind"], panel["data"]
    require(
        not model.config.bd3lm_block_size and (model.config.causal or loaded.noise_type == "mdlm"),
        "sequence panels require a native AR or full-sequence absorbing MDLM checkpoint",
    )
    if kind == "packed":
        count, width = data["x_0"].shape
        inputs = FrozenInputs(data)
        require(count % db == 0, "packed device batch must divide the row count")
    else:
        count, width = len(data["examples"]), data["settings"]["sequence_length"]
    require(width <= model.config.sequence_len, "panel exceeds the model context")
    records = [dict(id=i, stratum=kind, metrics={}) for i in range(count)]
    plans = {}
    seeds = (None,) if model.config.causal else FULL_SEEDS
    for seed in seeds:
        if kind == "bos":
            scored = score_bos(model, tokenizer, data, seed=seed, device=device, db=db, conditions=("document_start",))
            rows = scored["conditions"]["document_start"]["full_content"]["rows"]
            values = [(r["content_nats"], r["content_bytes"]) for r in rows]
            if seed is not None:
                plans[str(seed)] = scored["noise_plan"]
        elif seed is None:
            rows = score_ar(model, tokenizer, inputs, db=db, device=device)["rows"]
            values = [(r["content_nats"], r["content_bytes"]) for r in rows]
        else:
            values, plans[str(seed)] = packed_nelbo(model, tokenizer, inputs, seed, db, device)
        key = "ar/content" if seed is None else f"nelbo/{seed}"
        for record, value in zip(records, values, strict=True):
            record["metrics"][key] = list(value)
    if not model.config.causal:
        for record in records:
            values = list(record["metrics"].values())
            require(len({v[1] for v in values}) == 1, "denominator changed between plans")
            record["metrics"]["nelbo/mean_plans"] = [math.fsum(v[0] for v in values) / len(values), values[0][1]]
    return records, plans


def summaries(records):
    require(bool(records), "empty score report")
    result = {}
    for key in records[0]["metrics"]:
        nats = math.fsum(r["metrics"][key][0] for r in records)
        nbytes = sum(r["metrics"][key][1] for r in records)
        require(nbytes > 0 and math.isfinite(nats) and nats >= 0, "invalid score totals")
        result[key] = {"nats": nats, "bytes": nbytes, "bpb": nats / (math.log(2) * nbytes)}
    return result


@torch.inference_mode()
def score_panel(loaded, panel, *, device="cpu", dtype="float32", db=None, manifest_path=None):
    kind = panel["kind"]
    require(loaded.tokenizer_sha256 == panel["tokenizer_sha256"], "checkpoint/panel tokenizer mismatch")
    config = loaded.model.config
    require(
        not loaded.model.training
        and not config.time_conditioning
        and not config.target_length_conditioning
        and config.output_factorization == "flat",
        "flat unconditioned inference model required",
    )
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    db = (16 if kind == "packed" else 4) if db is None else db
    require(db > 0, "positive device batch required")
    with torch.autocast("cuda", dtype=torch.bfloat16) if dtype == "bfloat16" and device == "cuda" else nullcontext():
        if kind in ("packed", "bos"):
            rows, plans = sequence_scores(loaded, panel, db, device)
            if panel["reference"] and not config.causal:
                expected = reference_manifest(manifest_path)["full_sequence_plans"][kind]
                for seed, fields in expected.items():
                    require(all(plans[seed][k] == v for k, v in fields.items()), "reference corruption plan differs")
        else:
            require(db == 4, "block replay uses four native draws per device batch")
            rows, plans = block_scores(loaded, panel["data"]), {}
    seeds = [] if config.causal else list(FULL_SEEDS if kind in ("packed", "bos") else BLOCK_SEEDS)
    return dict(
        schema=REPORT_SCHEMA,
        kind=kind,
        reference=panel["reference"],
        panel_sha256=panel["panel_sha256"],
        tokenizer_sha256=loaded.tokenizer_sha256,
        checkpoint_sha256=loaded.checkpoint_sha256,
        depth=config.n_layer,
        device=device,
        dtype=dtype,
        device_batch=db,
        seeds=seeds,
        plans=plans,
        summary=summaries(rows),
        rows=rows,
    )
