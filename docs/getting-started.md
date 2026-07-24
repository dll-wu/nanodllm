# Getting Started

This guide takes a fresh source checkout through a portable CPU run, then shows
how to point the same trainer at a real Parquet corpus.

## 1. Create An Environment

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install a PyTorch build for the platform, then install the project:

```bash
python -m pip install -e .
```

For tests and packaging:

```bash
python -m pip install -e ".[dev]"
```

## 2. Prepare Portable Smoke Data

```bash
python scripts/prepare_tiny_corpus.py
```

The script converts the checked-in original text fixture into
`data/tiny_corpus.parquet`. Re-running it with unchanged input produces the same
logical rows.

## 3. Run The CPU Smoke

```bash
nanodllm-pretrain --config configs/smoke_byte.json
```

This uses the byte tokenizer, float32, no compilation, a tiny model, and a short
training horizon. It validates parsing, Parquet loading, packing, corruption,
forward/backward, optimization, structured logs, and checkpoint output. It is
not a quality experiment.

Useful bounded overrides:

```bash
nanodllm-pretrain \
  --config configs/smoke_byte.json \
  --run-dir runs/my-smoke \
  --set train.num_steps=2 \
  --set train.tensorboard=false
```

## 4. Inspect The Run

At minimum, check:

```bash
python -m json.tool runs/smoke_byte/resolved_config.json
python -m json.tool runs/smoke_byte/run_summary.json
python -m json.tool runs/smoke_byte/metrics.json
```

`train_log.jsonl` contains per-interval metrics. A successful checkpoint is
`model_latest.pt`.

## 5. Run The Analytic Audit

```bash
nanodllm-pretrain --config configs/mdlm_bayes_analytic_audit.json
```

This deterministic CPU-only experiment checks the synthetic data and absorbing
corruption calculation without instantiating or training a model. See
`docs/experiments/bayes-analytic-audit.md`.

## 6. Use Your Own Corpus

Create Parquet files with a string column named `text`, then override the smoke
recipe:

```bash
nanodllm-pretrain \
  --config configs/smoke_byte.json \
  --run-dir runs/my-corpus \
  --set 'data.train_data=["/absolute/path/to/data.parquet"]' \
  --set data.val_fraction=0.1 \
  --set train.num_steps=20
```

Directories and globs are also accepted. Start with the byte tokenizer while
checking the loader. Move to a Hugging Face tokenizer only after its special
tokens and byte table are verified.

## 7. ClimbMix

For a small public-data experiment, download one or more shards from
[`karpathy/climbmix-400b-shuffle`](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle)
and point `NANODLLM_DATA_DIR` at the directory.

The project 32k tokenizer requires the complete pinned corpus; follow
`docs/tokenizer-training.md`. Then:

```bash
export NANODLLM_DATA_DIR=/path/to/climbmix-400b-shuffle
export NANODLLM_TOKENIZER_DIR=/path/to/climbmix-32k

nanodllm-pretrain --config configs/climbmix_mdlm_small.json
```

Review every provisional field and run a GPU capacity probe before increasing
the model or horizon.

## 8. Resume

For the same run directory and resolved trajectory:

```bash
nanodllm-pretrain \
  --config configs/climbmix_mdlm_small.json \
  --set train.resume_from=latest
```

Resume intentionally rejects incompatible architecture, data, batch, world
size, and schedule changes.

## 9. Verify The Checkout

```bash
ruff check .
ruff format --check .
pytest -q
python -m build
python -m twine check dist/*
```

See `docs/hpc.md` before a distributed launch and
`docs/reproducibility.md` before comparing runs.
