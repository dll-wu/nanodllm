"""Causal Gemma 4 likelihood of fixed text, with bounded output projection."""

import math

import torch
import torch.nn.functional as F

from ..runtime.core_evaluation_cells import require


def encode_continuation(tokenizer, prefix, text, *, max_length):
    """Force an explicit segment boundary; never merge target bytes into the prompt."""
    require(isinstance(prefix, str) and isinstance(text, str), "text inputs required")
    # Special-token strings in ordinary text must remain ordinary text.
    kwargs = {"add_special_tokens": False, "split_special_tokens": True}
    context = [tokenizer.bos_token_id, *tokenizer.encode(prefix, **kwargs)]
    target = tokenizer.encode(text, **kwargs)
    require(len(context) + len(target) <= max_length, "scorer context exceeded; no truncation")
    require(bool(target) == bool(text), "tokenization dropped a nonempty continuation")
    require(tokenizer.bos_token_id is not None and tokenizer.pad_token_id is not None, "BOS/PAD required")
    return {
        "input_ids": context + target,
        "target_start": len(context),
        "target_count": len(target),
        "content_bytes": len(text.encode("utf-8")),
        "target_roundtrip_exact": tokenizer.decode(target, skip_special_tokens=False) == text,
    }


def make_batch(rows, *, pad_id, device):
    require(bool(rows), "empty batch")
    width = max(len(r["input_ids"]) for r in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        values = row["input_ids"]
        require(1 <= row["target_start"] <= len(values), "invalid continuation boundary")
        require(row["target_count"] == len(values) - row["target_start"], "invalid target count")
        ids[i, : len(values)] = torch.tensor(values, dtype=torch.long, device=device)
        mask[i, : len(values)] = 1
    return ids, mask


@torch.inference_mode()
def continuation_losses(encoder, head, rows, *, pad_id, softcap, chunk_size, device):
    """Shift causal hidden states by one; only project positions that predict targets."""
    require(chunk_size > 0 and softcap > 0, "invalid projection settings")
    ids, mask = make_batch(rows, pad_id=pad_id, device=device)
    output = encoder(input_ids=ids, attention_mask=mask, return_dict=True, use_cache=False)
    hidden = output.last_hidden_state
    # Never retain or reuse state between independent examples/batches.
    del output
    counts = [r["target_count"] for r in rows]
    selected, targets = [], []
    for i, row in enumerate(rows):
        start, count = row["target_start"], row["target_count"]
        selected.append(hidden[i, start - 1 : start + count - 1])
        targets.append(ids[i, start : start + count])
    features, labels = torch.cat(selected), torch.cat(targets)
    losses = []
    for start in range(0, len(labels), chunk_size):
        logits = head(features[start : start + chunk_size]).float()
        logits.div_(softcap).tanh_().mul_(softcap)
        losses.append(F.cross_entropy(logits, labels[start : start + chunk_size], reduction="none"))
    values = torch.cat(losses).cpu().tolist() if losses else []
    require(all(math.isfinite(v) and v >= 0 for v in values), "invalid reference loss")
    result, offset = [], 0
    for count in counts:
        result.append(values[offset : offset + count])
        offset += count
    return result


def reduce_text_losses(losses, content_bytes):
    count, total = len(losses), math.fsum(losses)
    require(all(math.isfinite(v) and v >= 0 for v in losses), "invalid token losses")
    require((count > 0 and content_bytes > 0) or (count == content_bytes == 0), "invalid byte denominator")
    mean = total / count if count else None
    return {
        "scored_tokens": count,
        "scored_bytes": content_bytes,
        "nll_sum": total,
        "mean_nll": mean,
        "perplexity": math.exp(mean) if mean is not None else None,
        "bpb": total / (content_bytes * math.log(2)) if count else None,
    }


class Gemma4Scorer:
    def __init__(self, path, device):
        self.device = device
        from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.model, loading = Gemma4ForConditionalGeneration.from_pretrained(
            path,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            attn_implementation="sdpa",
            output_loading_info=True,
        )
        require(
            not any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")),
            "non-exact pretrained checkpoint load",
        )
        self.model.to(device).eval().requires_grad_(False)
        # Preserve the public model's mask construction, including its text-only
        # multimodal-mask path; bypassing that changes SDPA kernels in BF16.
        self.encoder = self.model.model
        require(self.model.config.text_config.use_bidirectional_attention != "all", "causal text encoder required")
        self.head = self.model.lm_head
        self.softcap = float(self.model.config.text_config.final_logit_softcapping)

    def score(self, rows, *, chunk_size=512):
        return continuation_losses(
            self.encoder,
            self.head,
            rows,
            pad_id=self.tokenizer.pad_token_id,
            softcap=self.softcap,
            chunk_size=chunk_size,
            device=self.device,
        )


class LocalCausalScorer:
    """Other HF causal text checkpoints, using their own final logit transform."""

    def __init__(self, path, device):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                path,
                local_files_only=True,
                trust_remote_code=False,
                dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            )
            .to(device)
            .eval()
        )
        if self.model.config.is_encoder_decoder:
            raise ValueError("reference must be a causal language model")

    @torch.inference_mode()
    def score(self, rows):
        ids, mask = make_batch(rows, pad_id=self.tokenizer.pad_token_id, device=self.device)
        logits = self.model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        losses = []
        for i, row in enumerate(rows):
            start = row["target_start"]
            end = start + row["target_count"]
            values = F.cross_entropy(logits[i, start - 1 : end - 1].float(), ids[i, start:end], reduction="none")
            losses.append(values.cpu().tolist())
        return losses


def score_outputs(rows, prefixes, model_dir, device):
    """Score the fixed decoded continuations, with the reference model loaded after generation."""
    import hashlib
    import json
    from pathlib import Path

    from ..evaluation.checkpoint import stream_sha256

    root = Path(model_dir).resolve(strict=True)
    config = json.loads((root / "config.json").read_bytes())
    scorer = (Gemma4Scorer if config["model_type"] == "gemma4" else LocalCausalScorer)(root, device)
    files = {}
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix in {".json", ".model", ".safetensors", ".bin", ".txt", ".tiktoken"}:
            with path.open("rb") as stream:
                files[path.name] = stream_sha256(stream)
    content_sha256 = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    for i in range(0, len(rows), 2):
        batch = rows[i : i + 2]
        encoded = [
            encode_continuation(scorer.tokenizer, prefixes[r["prompt_id"]], r["text"], max_length=4096) for r in batch
        ]
        losses = scorer.score(encoded)
        for row, item, values in zip(batch, encoded, losses, strict=True):
            row["reference"] = reduce_text_losses(values, item["content_bytes"])
    nll = math.fsum(r["reference"]["nll_sum"] for r in rows)
    tokens = sum(r["reference"]["scored_tokens"] for r in rows)
    size = sum(r["reference"]["scored_bytes"] for r in rows)
    return {
        "model_type": config["model_type"],
        "content_sha256": content_sha256,
        "tokens": tokens,
        "bytes": size,
        "nll_sum": nll,
        "ppl": math.exp(nll / tokens) if tokens else None,
        "bpb": nll / (size * math.log(2)) if size else None,
    }
