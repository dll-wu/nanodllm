# Training

To follow an optimizer update, start at [`_run_training_steps`](../../nanodllm/runtime/training_step.py). Each microbatch contributes gradients through the [causal](../../nanodllm/runtime/training_causal_microbatch.py), [block](../../nanodllm/runtime/training_block_conversion_step.py), or [diffusion](../../nanodllm/runtime/training_diffusion_microbatch.py) objective. After accumulation, the trainer reduces the counters across ranks and [applies the optimizer update](../../nanodllm/runtime/training_step_update.py). Checkpointing captures the prefetched next batch so resumed runs consume the same examples.

The public `nanodllm.training.train` enters through [session initialization](../../nanodllm/runtime/training_loop.py). A config that enables the analytic Bayes audit invokes it after tokenizer construction. `nanodllm.telemetry` supplies the structured-file and TensorBoard writers.

## Launch Modes

CPU smoke:

```bash
python scripts/prepare_tiny_corpus.py
nanodllm-pretrain --config recipes/preview/smoke_byte.json
```

One GPU:

```bash
export NANODLLM_DATA_DIR=/path/to/climbmix
export NANODLLM_TOKENIZER_DIR=/path/to/climbmix-32k-tokenizer
nanodllm-pretrain --config recipes/preview/climbmix_mdlm_small.json
```

Single-node DDP:

```bash
torchrun --standalone --nproc_per_node=8 pretrain.py \
  --config recipes/preview/climbmix_mdlm_d20.json
```

`torchrun` sets rank, local rank, and world size. The trainer uses NCCL for CUDA
and Gloo for CPU process groups.

## Preflight

Before an expensive run:

1. validate the exact config and path overrides;
2. run `scripts/data_identity.py`;
3. run `scripts/gpu_smoke.py` on the allocated accelerator and dtype;
4. run a few real-data eager steps;
5. repeat with compilation enabled;
6. measure peak memory and steady-state tokens/second; and
7. test an intermediate checkpoint and resume.

A smoke on one GPU model or memory size does not qualify another.

## Budget Resolution

Training length may be explicit (`train.num_steps` or `train.target_tokens`) or
derived from `train.target_param_data_ratio`. Automatic total-batch and
learning-rate scaling are available but remain config-visible.

The optimizer-step token batch is:

```text
world_size * device_batch_size * max_seq_len * gradient_accumulation
```

The trainer records the chosen parameter-count basis, target tokens, resolved
steps, batch arithmetic, and learning-rate schedule in run metadata. Compare
resolved values, not just source configs.

## Optimizers

AdamW is the public baseline. Muon is available for explicit experiments and
uses AdamW for parameter groups that are not eligible for Muon. Optional
`optim.muon_lr` and `optim.adamw_lr` separate group learning rates; `-1`
inherits `optim.lr`.

Warmup may be an explicit step count or a fraction. Warmdown is a fraction of
the run, ending at `optim.final_lr_frac * base_lr`. Gradient clipping happens
after unscaling.

## Precision And Compilation

Supported dtypes are float32, float16, and bfloat16. Confirm BF16 support on the
actual GPU before choosing it. TF32 and cuDNN benchmarking are explicit runtime
settings.

`runtime.compile=true` compiles the model. Compilation can dominate a short run,
so compare steady-state throughput after warmup. A compiled smoke proves only
the exercised shapes and branches.

For closer replay, disable nondeterministic runtime features and enable the
repository's deterministic runtime settings. Exact replay also depends on
hardware, world size, package versions, and loader topology.

## Structured Outputs

Rank 0 writes under `train.run_dir`:

- `resolved_config.json`: the exact post-override config;
- `architecture_fingerprint.json`: resolved architecture and parameter
  inventory;
- `data_identity.v1.json`: input identity when enabled;
- `run_summary.json`: tokenizer, model, optimizer, budget, runtime, and batch
  metadata;
- `train_log.jsonl`: loss, LR, throughput, active tokens, gradients, and memory;
- `eval_log.jsonl`: periodic evaluation records;
- `tensorboard/`: mirrored scalar events when enabled;
- `metrics.json`: final summary;
- `model_latest.pt`: canonical resumable checkpoint; and
- `model.pt`: final compatibility alias when training completes.

When the package is running directly from its Git checkout, `metrics.json`
records the commit, whether tracked files are clean, and the count of untracked
paths. Installed wheels and source archives report Git provenance as
unavailable. These fields are informational: a missing or dirty checkout never
changes an otherwise successful run to `blocked`, and untracked path names are
not recorded.

JSONL is the stable machine-readable surface. Console output is for operators.
TensorBoard, garbage-collector state, and distributed process groups are
restored or closed on both success and exceptions.

## Checkpoints

`train.save_every=-1` saves only the final checkpoint. A positive value also
saves numbered intermediate checkpoints.

Resume the canonical checkpoint in the same run directory with:

```bash
nanodllm-pretrain \
  --config recipes/preview/climbmix_mdlm_small.json \
  --set train.resume_from=latest
```

The resume contract binds trajectory-defining config, derived schedule, model,
data identity, world size, accumulation, and topology. It stores optimizer,
scaler, RNG, and loader state and checks the replayed next batch. A mismatch
fails closed instead of silently changing the trajectory.

Copying a checkpoint without its resolved config and identity records weakens
reproducibility. Keep the complete compact run metadata even when large model
bytes are stored separately.

On successful finalization, rank 0 writes the canonical checkpoint and
compatibility alias before publishing `metrics.json`. A failed checkpoint
therefore cannot leave a new success snapshot that claims the model was saved.

## Evaluation Scheduling

Set `train.eval_every` to a positive step interval for periodic evaluation.
Evaluation is collective where required; every rank must follow the same
schedule. Final evaluation and checkpoint behavior are controlled separately by
the resolved config.

Cheap periodic checks should use smaller fixed budgets than endpoint
comparisons. Record both so a low-cost canary is not mistaken for a final
measurement.

## Failure Handling

The trainer rejects NaN/Inf losses and configured divergence conditions. For a
multi-process failure, preserve the first error, resolved config, run summary,
last structured metrics, and launcher environment. Do not promote a partially
written checkpoint without validating it through resume.
