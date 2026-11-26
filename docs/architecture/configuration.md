# Configuration

The trainer reads a JSON config into [runtime dataclasses](../../nanodllm/runtime/config.py). For a single CPU experiment, start with [the block recipe](../../recipes/tiny-block.json). For matched multi-stage runs, the [recipe launcher](../../recipes/README.md) combines shared defaults and per-stage overrides into that same JSON format.

```bash
nanodllm-pretrain \
  --config recipes/tiny-block.json \
  --set train.num_steps=2 \
  --set runtime.device_type=cpu
```

`--set` may be repeated. Values are parsed as JSON when possible, so booleans,
numbers, lists, objects, and quoted strings retain their types. `--run-dir PATH`
is a shortcut for `--set train.run_dir=PATH`.

## Earlier preview configs

- `smoke_byte.json`: portable end-to-end CPU check.
- `climbmix_mdlm_small.json`: bounded starting point for one accelerator.
- `climbmix_mdlm_d20.json`: provisional multi-accelerator scale-up shape.
- `mdlm_bayes_analytic_audit.json`: deterministic CPU-only analytic audit.

Read a config's `recipe.status` before launching it:

- `locked` means the recipe has no unresolved fields.
- `provisional` means its `provisional_fields` must be reviewed for the intended
  run.
- historical recipes are intentionally excluded from this public snapshot.

`recipe.locked_fields` and `recipe.provisional_fields` are validated dotted
paths into the same resolved config.

## Major Sections

`model` controls depth, hidden width, heads, context length, SwiGLU width, RoPE,
attention scale, initialization, optional heads, and experimental attention
layouts.

`data` controls:

- `tokenizer`: `byte` or `hf`;
- `tokenizer_path` for a local Hugging Face-compatible directory;
- train, exclusion, and validation Parquet selectors;
- text column and validation fraction;
- BOS/EOS and loss policy;
- packing augmentations;
- row-group shuffling; and
- data-identity strength.

Paths pass through environment-variable and user-home expansion. Prefer:

```json
{
  "train_data": ["${NANODLLM_DATA_DIR}"],
  "tokenizer_path": "${NANODLLM_TOKENIZER_DIR}"
}
```

Export those variables before launch. An undefined variable remains literal and
causes the normal missing-path validation to fail.

`noise` selects the corruption family and its time range. The public baseline is
`mdlm` with schedule weighting. Other families are experimental.

`optim` controls optimizer type, base and per-group learning rates, Adam
parameters, weight decay, warmup/warmdown, final LR fraction, and clipping.

`runtime` controls device type, dtype, seeds, compilation, TF32, cuDNN
benchmarking, numeric determinism, and GPU preflight.

`train` controls run directory, step/token budget, total and device batch size,
logging, TensorBoard, save/eval intervals, and resume.

`eval` controls NELBO, fixed-noise reconstruction, generation, samplers,
conditional diagnostics, and optional judge-model scoring.

## Batch Arithmetic

For world size `W`, device batch `B`, sequence length `L`, and gradient
accumulation `A`, the optimizer-step token batch is:

```text
W * B * L * A
```

The trainer resolves `A` from `train.total_batch_size` and rejects incompatible
values unless the recipe explicitly permits rounding. Record the resolved
configuration rather than relying on the source JSON alone.

## Validation

`validate_config` rejects unsupported combinations before training, including:

- missing data or tokenizer selections;
- incompatible model dimensions;
- invalid token/time/evaluation budgets;
- unsupported dtype, optimizer, noise, or sampler values;
- impossible batch arithmetic;
- auxiliary losses without their required heads;
- invalid checkpoint/evaluation intervals;
- train/validation overlap after canonical path resolution; and
- incompatible experimental switches.

When adding a field, add validation that explains unsupported combinations.
