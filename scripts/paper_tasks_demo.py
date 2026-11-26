"""Create two tiny random byte checkpoints, original tasks and prompts for local CLI checks."""

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nanodllm.paper_runtime.config import ModelConfig  # noqa: E402
from nanodllm.paper_runtime.model import DiffusionConfig, DiffusionTransformer  # noqa: E402
from nanodllm.paper_weights import export_weights, file_sha256  # noqa: E402


def create(folder):
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.iterdir()):
        raise ValueError("demo output directory must be empty")
    config = DiffusionConfig(
        sequence_len=128, vocab_size=260, mask_token_id=257, n_layer=1, n_head=2, n_embd=32, bd3lm_block_size=8
    )
    settings = ModelConfig(
        depth=1,
        n_embd=32,
        n_head=2,
        max_seq_len=128,
        bd3lm_block_size=8,
        bd3lm_clean_stream_mode="causal",
        rope_base=10000.0,
        pad_vocab_to=1,
    )
    experiment = {"data": {"tokenizer": "byte"}, "model": asdict(settings), "train": {}, "noise": {"type": "bd3lm"}}
    with TemporaryDirectory() as temp:
        source = Path(temp) / "trainer.pt"
        for seed in (11, 12):
            torch.manual_seed(seed)
            model = DiffusionTransformer(config, rope_base=10000.0, compute_dtype=torch.float32, pad_vocab_to=1)
            model.init_weights()
            torch.save(
                {
                    "config": asdict(config),
                    "train_config": experiment,
                    "model_state_dict": model.state_dict(),
                    "step": 0,
                    "train_tokens": 0,
                    "train_valid_tokens": 0,
                },
                source,
            )
            export_weights(source, folder / f"demo-{seed}.weights.pt", expected_source_sha256=file_sha256(source))
    rows = {
        "toy_choices": (
            "multiple_choice",
            [dict(query=f"Locker {i} label", choices=["A", "B"], gold=i % 2) for i in range(3)],
        ),
        "toy_schema": (
            "schema",
            [dict(context_options=[f"Tray {i} L", f"Tray {i} R"], continuation="!", gold=i % 2) for i in range(3)],
        ),
        "toy_completion": ("language_modeling", [dict(context=f"Dial {i}", continuation=f"{i}.") for i in range(3)]),
    }
    specs = []
    for label, (kind, items) in rows.items():
        payload = "".join(json.dumps(item) + "\n" for item in items).encode()
        filename = label + ".jsonl"
        (folder / filename).write_bytes(payload)
        baseline = 0 if kind == "language_modeling" else 0.5
        specs.append(
            dict(
                label=label,
                task_type=kind,
                num_fewshot=1,
                delimiter=" ",
                data_path=filename,
                raw_count=len(items),
                selected_count=len(items),
                data_sha256=hashlib.sha256(payload).hexdigest(),
                baseline_v1=baseline,
                baseline_v2=baseline,
            )
        )
    (folder / "roster.json").write_text(
        json.dumps({"schema": "nanodllm.paper_tasks.roster.v1", "tasks": specs}, indent=2) + "\n"
    )
    prompts = [{"id": "lamp", "text": "The lamp blinked twice."}, {"id": "dial", "text": "Turn the dial to "}]
    (folder / "prompts.jsonl").write_text("".join(json.dumps(p) + "\n" for p in prompts))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    create(p.parse_args().directory)
