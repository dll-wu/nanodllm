# NanoDLLM

Start with [the quickstart](README.md), [the implementation map](docs/architecture/architecture.md) and [scientific decisions](DESIGN_DECISIONS.md).

- `nanodllm/runtime/` owns the shared training, model and decoding operators. Preserve the public Python imports and the `nanodllm-pretrain` config interface when changing them.
- `recipes/` owns editable comparisons and component replications. Keep training and scoring usable with a local corpus and without any paper assets.
- `paper/` owns frozen profiles, evaluation panels and numerical exhibits. Preserve their identities when changing code or layout. The [release notes](paper/release.md) document the supported evidence.
- Keep training data, model weights, run logs and local paths outside Git. Update the focused guide and regression tests when an interface changes.
- Run `ruff check .`, `ruff format --check .`, `python -m pytest -q` and the [CPU quickstart](README.md#train-and-sample-without-a-gpu). For a packaging change follow the [distribution checks](docs/architecture/packaging.md).
