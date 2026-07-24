# nano-dllm

Research-preview code for training masked discrete diffusion language models
(dLLMs) with PyTorch.

The default path is a bidirectional absorbing-mask MDLM with Parquet streaming,
byte or Hugging Face tokenizers, structured JSONL metrics, periodic evaluation,
and resumable checkpoints. The repository also exposes experimental corruption,
optimizer, objective, and sampling variants for controlled comparisons.

This is a source and methods preview. It does not include pretrained weights, a
released project tokenizer, or final benchmark claims. See
[`docs/research-status.md`](docs/research-status.md) before interpreting the
included scale-up recipes.

## Install

Python 3.10 or newer is required. Install a PyTorch build appropriate to the
platform, then install the package:

```bash
python -m pip install -e .
```

For tests, linting, and local package builds:

```bash
python -m pip install -e ".[dev]"
```

The project does not force a private package index or one CUDA wheel channel.
Use the official PyTorch installation selector for accelerator-specific wheels.

## Five-Minute CPU Check

Create a tiny deterministic Parquet corpus from the checked-in original text:

```bash
python scripts/prepare_tiny_corpus.py
```

Run the byte-tokenizer smoke recipe:

```bash
nanodllm-pretrain --config configs/smoke_byte.json
```

This checks the complete train path—config, Parquet loading, packing,
corruption, forward/backward, optimization, metrics, and checkpointing—but is
not a language-quality experiment.

Run the self-contained analytic corruption audit:

```bash
nanodllm-pretrain --config configs/mdlm_bayes_analytic_audit.json
```

## ClimbMix Experiments

The public scale-up source is
[`karpathy/climbmix-400b-shuffle`](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle),
an MIT-licensed Parquet dataset whose current full snapshot is approximately
600 GB.

The repository includes the exact script used to train the project-specific
32,768-token byte-BPE tokenizer:

```bash
uv run --script scripts/train_climbmix_tokenizer.py --help
```

The complete, revision-pinned download and tokenizer procedure is in
[`docs/tokenizer-training.md`](docs/tokenizer-training.md). It reserves
`shard_06542.parquet` for validation and excludes that shard from tokenizer
fitting.

Once the corpus and tokenizer exist locally:

```bash
export NANODLLM_DATA_DIR=/path/to/climbmix-400b-shuffle
export NANODLLM_TOKENIZER_DIR=/path/to/climbmix-32k

nanodllm-pretrain --config configs/climbmix_mdlm_small.json
```

For the provisional d20 shape on one eight-GPU node:

```bash
torchrun --standalone --nproc_per_node=8 pretrain.py \
  --config configs/climbmix_mdlm_d20.json \
  --set train.num_steps=1 \
  --set train.total_batch_size=16384 \
  --set runtime.compile=false \
  --set eval.enabled=false
```

This bounded eager-mode command is a capacity probe with one microbatch per
rank. The d20 config's unmodified 12-tokens-per-parameter horizon is a scale-up
template, not a locked optimum; use it only after validating the exact GPU
type, compilation path, and evaluation memory.

Inspect its parameter/FLOP accounting without allocating model tensors:

```bash
python scripts/architecture_fingerprint.py \
  --config configs/climbmix_mdlm_d20.json \
  --vocab-size 32768
```

## Use Another Parquet Corpus

The loader accepts a Parquet file, directory, or glob. The default text column
is `text`.

```bash
nanodllm-pretrain \
  --config configs/smoke_byte.json \
  --run-dir runs/my-corpus \
  --set 'data.train_data=["/absolute/path/to/data.parquet"]' \
  --set data.val_fraction=0.1 \
  --set train.num_steps=20
```

Documents are joined with EOS and packed as one rolling stream. Documents may
cross row boundaries; row boundaries do not imply document boundaries.

## Common Overrides

```bash
nanodllm-pretrain \
  --config configs/climbmix_mdlm_small.json \
  --run-dir runs/debug \
  --set train.num_steps=10 \
  --set train.device_batch_size=1 \
  --set runtime.compile=false \
  --set eval.enabled=false
```

`--set` may be repeated and accepts dotted dataclass fields. Values are parsed
as JSON where possible.

## Outputs And Resume

Runs write under `train.run_dir`:

- `resolved_config.json`;
- `architecture_fingerprint.json`;
- `data_identity.v1.json` when enabled;
- `run_summary.json`;
- `train_log.jsonl` and `eval_log.jsonl`;
- `tensorboard/` when enabled;
- `metrics.json`; and
- `model_latest.pt` plus the final `model.pt` compatibility alias.

Resume the canonical checkpoint with the same trajectory-defining settings:

```bash
nanodllm-pretrain \
  --config configs/climbmix_mdlm_small.json \
  --set train.resume_from=latest
```

The resume contract checks model, data identity, schedule, batch topology,
loader state, and RNG state rather than silently accepting an incompatible run.

## Verification

```bash
ruff check .
ruff format --check .
pytest -q
python -m build
python -m twine check dist/*
```

CPU CI cannot qualify CUDA, NCCL, BF16, compilation, or memory capacity. Run
[`scripts/gpu_smoke.py`](scripts/gpu_smoke.py) and the bounded real-data
preflights described in [`docs/hpc.md`](docs/hpc.md) on the actual allocation.

## Documentation

- [Getting started](docs/getting-started.md)
- [Tokenizer reproduction](docs/tokenizer-training.md)
- [HPC and DDP](docs/hpc.md)
- [Reproducibility](docs/reproducibility.md)
- [Research status and limitations](docs/research-status.md)
- [Architecture and operating references](docs/README.md)
- [Scientific design decisions](DESIGN_DECISIONS.md)

## Scope And Stability

The trainer is usable research code, but the Python API, config schema, metric
schema, and checkpoint contract may evolve before a 1.0 release. Experimental
switches are not all validated as meaningful combinations. Preserve resolved
configs and source identity with every retained run.

## License And Citation

Code is released under the MIT License. See [`LICENSE`](LICENSE),
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md), and
[`CITATION.cff`](CITATION.cff).
