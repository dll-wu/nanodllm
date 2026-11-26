# Training on your own text

First run the [CPU quickstart](../README.md#train-and-sample-without-a-gpu) to check training, cached generation and scoring. It uses a byte tokenizer and a local Parquet file with a string column named `text`. For a real experiment, keep validation documents in a separate file and build your scoring panel from that file.

## Change the data or training budget

Copy [the block config](../recipes/tiny-block.json), edit it, or use dotted overrides:

```sh
nanodllm-pretrain --config recipes/tiny-block.json --run-dir runs/my-block \
  --set 'data.train_data=["/path/to/train.parquet"]' \
  --set 'data.val_data=["/path/to/validation.parquet"]' \
  --set train.num_steps=20
```

The config controls model depth and width, attention, corruption, optimizer, batch, training length and runtime. The [four-layer comparison](../recipes/README.md) supplies larger settings and explains how to initialize a diffusion model from an AR parent. The [tokenizer guide](tokenizer-training.md) covers the 32k tokenizer used for the paper.

## Inspect a run

Open `resolved_config.json` for the executed settings, `train_log.jsonl` for the loss and learning rate, and `metrics.json` for final measurements. The trainer writes its final state to `model_latest.pt`. Export [portable weights](portable-weights.md) before exchanging a model.

To resume an interrupted run, repeat the original command with `--set train.resume_from=latest`. Keep the original training schedule and data, and use the same process count. To initialize a new experiment from a previous model, follow the recipe's weight initialization procedure instead. [Distributed training](hpc.md) · [Run accounting](reproducibility.md)

## Earlier preview configurations

The original full-sequence configs include the [small ClimbMix run](../recipes/preview/climbmix_mdlm_small.json), the [D20 run](../recipes/preview/climbmix_mdlm_d20.json), the [byte-tokenizer smoke test](../recipes/preview/smoke_byte.json), and the [analytic masking audit](experiments/bayes-analytic-audit.md). The same `nanodllm-pretrain` command reads them. The four original `configs/*.json` paths remain aliases in a source checkout. For exact replay of an old run, use its recorded source revision and environment.
