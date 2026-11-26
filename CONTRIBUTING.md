# Contributing

Thank you for helping improve nano-dllm. For research code,
reproducibility, explicit assumptions, and narrowly scoped changes are
especially valuable.

## Development setup

Use Python 3.11 or newer in an isolated environment:

```bash
python -m pip install -e ".[dev]"
python scripts/prepare_tiny_corpus.py
pytest -q
ruff check .
ruff format --check .
```

Run the bounded CPU training smoke after changing the model, data path,
objective, configuration, or training loop:

```bash
nanodllm-pretrain \
  --config recipes/tiny-block.json \
  --set train.num_steps=1 \
  --set train.run_dir=runs/contributor-smoke
```

Before proposing packaging changes, also verify both distribution formats:

```bash
python -m build
twine check --strict dist/*
```

## Proposing changes

- Keep each change focused and explain its motivation and expected scientific
  or operational effect.
- Add or update tests for behavior changes.
- Update the relevant file under `docs/` when configuration semantics,
  architecture, launch behavior, or run artifacts change.
- Update `DESIGN_DECISIONS.md` when changing a scientific default.
- Do not commit datasets, model weights, checkpoints, run logs, credentials,
  machine-specific paths, or unpublished experimental results.
- Report security concerns privately as described in `SECURITY.md`.

When reporting experimental results, include the config, random seeds, hardware
and software environment, dataset and tokenizer identity, evaluation method,
and enough commands to reproduce the result. Clearly distinguish measured
results from hypotheses.

Contributions are submitted under the repository's MIT license.
