# Packaging

The project uses a standard `pyproject.toml` build:

- distribution: `nano-dllm`;
- import package: `nanodllm`;
- console command: `nanodllm-pretrain`; and
- source-tree wrapper: `pretrain.py`.

This is a source release. Example configs and scripts remain repository assets;
they are not installed as package data. Always pass an explicit config path.

## Install

Create an isolated environment and install a PyTorch build appropriate to the
platform. Then install the project:

```bash
python -m pip install -e .
```

Development tools:

```bash
python -m pip install -e ".[dev]"
```

The package metadata does not force a private index or a particular CUDA wheel
channel. Follow the official PyTorch selector when a CUDA-specific wheel is
needed.

## Build

```bash
python -m build
python -m twine check dist/*
```

Inspect the wheel rather than assuming source-only files were included:

```bash
python -m zipfile --list dist/*.whl
```

The public preview is not published to PyPI. A local wheel should still install
cleanly and expose `nanodllm-pretrain --help`.

## Quality Checks

```bash
ruff check .
ruff format --check .
pytest -q
python -m build
python -m twine check dist/*
```

CI runs these checks on supported Python versions. GPU behavior still requires
an explicit accelerator smoke because hosted CPU CI cannot validate CUDA,
NCCL, BF16, compilation, or memory capacity.

## Dependency Policy

Runtime imports belong in `project.dependencies`; build, test, and lint tools
belong in the `dev` extra. Tokenizer fitting has heavier, pinned standalone
dependencies in the PEP 723 header of
`scripts/train_climbmix_tokenizer.py`, so normal trainer installs do not pull
them in.
