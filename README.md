# NanoDLLM

NanoDLLM is a PyTorch codebase for experimenting with small diffusion language models. Train a masked diffusion model on whole sequences or successive blocks, start it from scratch or from an autoregressive (AR) checkpoint, and compare the resulting models on the same inputs. The block decoder reuses completed blocks' attention keys and values as the continuation grows.

## Train and sample on a laptop

From a source checkout with Python 3.11 or newer and a suitable PyTorch installation:

```sh
python -m pip install -e .
python scripts/prepare_tiny_corpus.py
nanodllm-pretrain --config recipes/tiny-block.json
nanodllm-generate --checkpoint runs/tiny-block/model_latest.pt --training-checkpoint \
  --prompt 'The lamp' --capacity 24 --seed 7
```

This trains a two-layer model for two CPU updates with 8-token blocks. The last command samples up to three blocks after your prompt and prints the text and forward-pass count. The tiny corpus makes this a check of the pipeline. Use the training-checkpoint flag only for a checkpoint you trust. [Editable config](recipes/tiny-block.json) · [Portable weights](docs/portable-weights.md)

Score next-token predictions from that same checkpoint:

```sh
nanodllm-eval prepare block --parquet data/tiny_corpus.parquet \
  --block-size 8 --count 4 --output runs/tiny-block/panel.pt
nanodllm-eval score --panel runs/tiny-block/panel.pt \
  --checkpoint runs/tiny-block/model_latest.pt --trusted-training-checkpoint \
  --output runs/tiny-block/scores.json
```

This example uses the training fixture for convenience. For an experiment, build the panel once from held-out documents and reuse it across checkpoints. [Evaluation guide](docs/evaluation.md)

## Run an experiment

The first matched comparison asks how to spend a fixed training budget. Train a four-layer block model from scratch for 40 tokens per counted parameter, or spend the first 10 on an AR parent and the remaining 30 on block diffusion. Both routes use the same trainer and evaluator. The recipe gives the settings, parent handoff and commands for both arms. [Training recipes](recipes/README.md) · [Published method replications](recipes/components/README.md)

| If you want to… | Start here |
| --- | --- |
| Change the mask, model, or decoder | [Core implementation map](docs/architecture/architecture.md) |
| Train on your own corpus | [Getting started](docs/getting-started.md) |
| Evaluate tasks, stopping, repetition, or decoding speed | [Tasks and generation](docs/generation-and-tasks.md) |
| Rebuild the published figures and tables, or repeat an exact training run | [Paper companion](paper/README.md) |

[MIT license](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md) · [Citation](CITATION.cff)
