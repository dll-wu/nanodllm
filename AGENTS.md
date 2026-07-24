# AGENTS.md

This file is the map, not the manual. Keep it short and send agents to the
repo-local source of truth in `docs/`; see
`docs/architecture/agent-legibility.md` for the repo's agent-legibility rules.

## Start Here

1. Read `README.md` for install and launch commands.
2. Read `docs/README.md` for the full documentation index.
3. Read `DESIGN_DECISIONS.md` before changing model, noise, optimizer, sampler,
   or scale-up defaults.

## Repo Purpose

This repository trains full-scale masked discrete diffusion language models. The
default path is an MDLM bidirectional transformer with parquet streaming,
Hugging Face or byte tokenizers, structured JSONL metrics, periodic evaluation,
and resumable checkpoints.

## Boundaries

- `nanodllm/cli.py` owns CLI parsing and config overrides.
- `pretrain.py` is only a source-tree wrapper around the packaged CLI.
- `nanodllm/config.py` owns dataclass config schema and validation.
- `nanodllm/data.py` owns parquet discovery, row-group streaming, tokenizers,
  DDP sharding, and sequence packing.
- `nanodllm/model.py`, `nanodllm/noise.py`, and `nanodllm/sampling.py` own core
  modeling behavior.
- `nanodllm/training.py` owns distributed setup, optimization, logging,
  evaluation scheduling, and checkpointing.
- `configs/` owns recipes; keep experiment toggles explicit in JSON.
- `docs/` owns durable repository knowledge.

## Working Rules

- Prefer small, config-driven changes over new entrypoints.
- Keep `AGENTS.md` short; add details to a focused file under `docs/`.
- Update docs in the same change that alters launch behavior, config semantics,
  run artifacts, or architecture.
- Keep generated corpora, checkpoints, results, and reports out of Git.
- Keep checked-in configs portable: use repository-relative example paths and
  require users to supply paths for large external datasets or tokenizers.
- Do not change scientific or scale-up defaults without updating
  `DESIGN_DECISIONS.md`.
- Add runtime dependencies to `pyproject.toml`, not ad hoc install notes.
- Keep Ruff clean. CI runs `ruff check .` and `ruff format --check .`.
- Run the test suite and the bounded CPU smoke before merging behavior changes.

## Useful Commands

```bash
python3 -m pip install -e ".[dev]"
python scripts/prepare_tiny_corpus.py
nanodllm-pretrain --config configs/smoke_byte.json
pytest -q
ruff check .
ruff format --check .
python -m build
twine check dist/*
```
