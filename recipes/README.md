# NanoDLLM training recipes

List the editable baselines with `nanodllm-recipes list`. To include the archived study profiles, use `list --include-paper`.

## Train a tiny block model on CPU

From the release checkout with NanoDLLM installed, run the following commands.

```sh
python scripts/prepare_tiny_corpus.py
nanodllm-pretrain --config recipes/tiny-block.json
```

The preparation script writes 32 documents to `data/tiny_corpus.parquet`. The model uses the byte tokenizer, two transformer layers and fixed 8-token blocks in a 128-token context. Two FP32 AdamW updates consume 256 allocated source positions. The run writes its final checkpoint to `runs/tiny-block/model.pt`, its resolved settings to `resolved_config.json` in the same directory, and per-step metrics to `train_log.jsonl`. To use another output directory, pass `--run-dir PATH`. [Training config](tiny-block.json)

## Compare scratch40 with AR10 + block30 at D4

The editable D4 catalog has three single-GPU stages. Both routes use seed 48, a four-layer, width-256 model, a 1,024-token context and 65,536 allocated positions per update. The block stages use causal-clean attention, mixed partition lengths of 8, 16, 32 and 64, and a reference block length of 32. The budget counts the transformer matrices and output head, giving 11,534,336 counted parameters with the 32k tokenizer. [Shared settings and stage overrides](profiles/d4-matched.json)

| Profile | Initialization | Updates | Allocated source positions |
| --- | --- | ---: | ---: |
| `d4-scratch40` | Fresh block model | 7,040 | 461,373,440 |
| `d4-ar10` | Fresh causal AR model | 1,760 | 115,343,360 |
| `d4-block30` | Model weights from the completed `d4-ar10` | 5,280 | 346,030,080 |

Supply a local corpus directory and tokenizer. The default held-out file is `shard_06542.parquet` inside the corpus and is excluded from training. These settings target one CUDA device with BF16 support. To inspect the config before training, use `materialize` in place of `launch`. [Corpus and tokenizer](../paper/data.md)

```sh
export CUBLAS_WORKSPACE_CONFIG=:4096:8
corpus=data/climbmix
tokenizer=data/climbmix-tokenizer

nanodllm-recipes list --group d4-matched
nanodllm-recipes launch d4-scratch40 --corpus "$corpus" --tokenizer "$tokenizer" --run-dir runs/d4-scratch40
nanodllm-recipes launch d4-ar10 --corpus "$corpus" --tokenizer "$tokenizer" --run-dir runs/d4-ar10

source=runs/d4-ar10/model.pt
source_sha=$(python -c 'from nanodllm.weights import file_sha256; import sys; print(file_sha256(sys.argv[1]))' "$source")
parent=runs/d4-ar10/model.weights.pt
parent_sha=$(nanodllm-export-weights --trusted-checkpoint "$source" --source-sha256 "$source_sha" \
  --tokenizer-dir "$tokenizer" --output "$parent")
nanodllm-recipes launch d4-block30 --corpus "$corpus" --tokenizer "$tokenizer" \
  --parent "$parent" --parent-sha256 "$parent_sha" --run-dir runs/d4-block30
```

The two routes each spend 461,373,440 positions. The continuation loads its parent's weights and starts a fresh optimizer, schedule and data stream. Its `initial_state.v1.json` records the parent digest. [Portable weight export](../docs/portable-weights.md)

To adapt the comparison, copy `profiles/d4-matched.json` and edit `defaults` for changes shared by the stages. Each profile's `overrides` replaces individual fields in those sections. Set `validation_shard` to the held-out filename for your corpus. If you change the vocabulary or architecture, recompute the token budgets for the new parameter count. Pass the same `--catalog` path for every stage. The launcher reads that catalog independently of the checkout's frozen paper profiles.

```sh
cp recipes/profiles/d4-matched.json my-d4.json
nanodllm-recipes list --catalog my-d4.json
nanodllm-recipes materialize d4-scratch40 --catalog my-d4.json --corpus "$corpus" \
  --tokenizer "$tokenizer" --run-dir runs/my-d4 --output-dir runs/resolved
```

For a single custom stage, `nanodllm-pretrain --config FILE` accepts a plain native JSON config such as `tiny-block.json`. The catalog launcher resolves profile defaults, rank counts and parent bindings before calling that same trainer.

## Try model-only conversion on CPU

This example exercises plain causal and diffusion training, best-fit packing, portable weight export and model-only initialization. It uses the project's original tiny text fixture, the byte tokenizer, a two-layer model and native AdamW. Scratch trains four updates. The two-stage route trains two AR and two MDLM updates. The published study uses the 32k tokenizer and the larger models above.

```sh
python scripts/prepare_tiny_corpus.py --output demo/tiny.parquet
nanodllm-recipes launch tiny-ar --corpus demo/tiny.parquet --run-dir demo/ar
nanodllm-recipes launch tiny-scratch --corpus demo/tiny.parquet --run-dir demo/scratch

source_sha=$(python -c 'import sys; from nanodllm.weights import file_sha256; print(file_sha256(sys.argv[1]))' demo/ar/model_latest.pt)
parent_sha=$(nanodllm-export-weights --trusted-checkpoint demo/ar/model_latest.pt \
  --source-sha256 "$source_sha" --output demo/ar.weights.pt)
nanodllm-recipes launch tiny-convert --corpus demo/tiny.parquet \
  --parent demo/ar.weights.pt --parent-sha256 "$parent_sha" --run-dir demo/convert
```

Each run writes a resolved config, initial-state receipt, training metrics and final checkpoint. In the conversion's `initial_state.v1.json`, `model_initialization_source.sha256` identifies the parent actually loaded.

The paired 20M component experiments have their own presets and scoring command. Exact paper replications use frozen catalogs with the archived checkpoint identities and input checks. [Component comparisons](components/README.md) · [Frozen paper profiles](../paper/training.md)
