# Packaging

The `nano-dllm` distribution installs the public training, scoring and checkpoint-export commands, plus the CPU-only `paper.companion` module. The wheel places recipes, frozen profiles and reference manifests at the same relative paths as the checkout. Numerical inputs and TeX templates travel inside `paper.companion`. The source distribution also includes documentation, contributor materials, original tiny-text examples and tests.

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

Run `python scripts/check_release_dist.py dist` to compare the sdist and wheel inventories with the checkout. It also checks every source archive entry for local user and group metadata. The build sets numeric owner IDs to zero and leaves owner names empty. Test the installed wheel from another directory to ensure it can locate its paper assets.

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
