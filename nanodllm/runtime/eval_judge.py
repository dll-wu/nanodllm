"""Causal language model judge and generation perplexity."""

import math
import os
from contextlib import ExitStack

import torch


class JudgeModel:
    def __init__(self, device, model_id: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = torch.device(device)
        self.model_id = model_id
        print(f"Loading judge model: {model_id}")
        self._tok = AutoTokenizer.from_pretrained(model_id)
        self._tok.pad_token = self._tok.eos_token
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(self.device).eval()

    @torch.no_grad()
    def score(self, texts: list[str]) -> float:
        enc = self._tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)
        labels = ids.clone()
        labels[mask == 0] = -100
        out = self._model(input_ids=ids, attention_mask=mask, labels=labels)
        return out.loss.item()


_judge_singleton: JudgeModel | None = None
_judge_device: torch.device | None = None
_judge_model_id: str | None = None


def _get_judge(device, model_id: str) -> JudgeModel:
    global _judge_singleton, _judge_device, _judge_model_id
    requested = torch.device(device)
    if _judge_singleton is None or _judge_device != requested or _judge_model_id != model_id:
        _judge_singleton = JudgeModel(requested, model_id)
        _judge_device = requested
        _judge_model_id = model_id
    return _judge_singleton


@torch.no_grad()
def evaluate_gen_ppl(
    model,
    tokenizer,
    sample_fn,
    batch_size: int,
    run_dir: str | None = None,
    device=None,
    judge_model_id: str = "Qwen/Qwen3-0.6B-Base",
    gen_samples: int = 512,
    gen_len: int = 512,
) -> float:
    device = next(model.parameters()).device if device is None else device
    judge = _get_judge(device, judge_model_id)
    n_batches = max(1, gen_samples // batch_size)
    total_ce = 0.0

    with ExitStack() as stack:
        log_file = None
        if run_dir is not None:
            os.makedirs(run_dir, exist_ok=True)
            log_file = stack.enter_context(open(os.path.join(run_dir, "gen_samples.log"), "w"))

        def log(msg: str):
            if log_file is not None:
                log_file.write(msg + "\n")
                log_file.flush()
            else:
                print(msg)

        logged_samples = 0
        for batch_idx in range(n_batches):
            tokens = sample_fn(model, batch_size, gen_len)
            texts = [tokenizer.decode(tokens[i].tolist()) for i in range(tokens.shape[0])]
            if logged_samples < 5:
                for i in range(min(batch_size, 5 - logged_samples)):
                    text = texts[i]
                    judge_len = judge._tok(text, return_tensors="pt", truncation=True, max_length=512)[
                        "input_ids"
                    ].shape[1]
                    log(f"[gen] batch={batch_idx} item={i} judge_len={judge_len}")
                    log(f"[gen] text={text[:300]!r}")
                    logged_samples += 1

            empty = sum(1 for txt in texts if len(txt.strip()) == 0)
            if empty:
                log(f"[gen] WARNING: batch {batch_idx} has {empty}/{batch_size} empty decoded strings")
            batch_ce = judge.score(texts)
            log(f"[gen] batch={batch_idx} batch_ce={batch_ce:.4f} batch_ppl={math.exp(batch_ce):.2f}")
            total_ce += batch_ce

    return math.exp(total_ce / n_batches)
