# NanoDLLM

NanoDLLM is a PyTorch research codebase for training and evaluating small masked diffusion language models. The paper compares training from scratch with initializing from an autoregressive (AR) parent. It follows both full-sequence diffusion and block diffusion from 20M to 477M parameters. Block models can generate beyond their training block length by appending blocks and caching completed prefix keys and values.

| What do you want to do? | Start here |
| --- | --- |
| Rebuild the figures and tables on a laptop | [Paper results](docs/paper-results.md) |
| Train an AR parent, a diffusion model from scratch, or an AR-initialized diffusion model | [Training recipes](docs/paper-recipes.md) |
| Prepare the corpus and identify the tokenizer | [Paper data](docs/paper-data.md) |
| Compare published diffusion components | [Component experiments](docs/paper-components.md) |
| Evaluate a checkpoint or compare two runs | [Corpus scoring](docs/paper-evaluation.md) and [tasks and generation](docs/paper-tasks-generation.md) |
| Find checkpoint identities and export local weights | [Checkpoint inventory](docs/checkpoints.md) and [portable weights](docs/portable-weights.md) |
| Understand release scope and validation | [Release notes](docs/release.md) |

## Reproduce the paper displays

From a source checkout, with Python 3.11 or newer and NumPy installed, run:

```bash
python -m paper_results --check
python -m paper_results --output paper-output
```

This recomputes the displays from the bundled numerical observations and checks them against the paper edition. It runs on CPU without training data or checkpoints. The output includes TeX and machine-readable statistics. The results guide identifies which summaries can be recomputed from document losses and which start from retained aggregates.

## Train on CPU

Install a PyTorch build appropriate for your platform and install the package. From the checkout:

```bash
python -m pip install -e .
python scripts/prepare_tiny_corpus.py
nanodllm-paper-recipes launch tiny-ar --corpus data/tiny_corpus.parquet --run-dir runs/tiny-ar
```

The example trains on the original text fixture using the built-in byte tokenizer. The paper profiles use a 32,768-token tokenizer and a held-out ClimbMix shard. The block series uses best-fit packing. The full-sequence scaling series preserves document tails. List the profiles and their parent dependencies before selecting a run:

```bash
nanodllm-paper-recipes list
```

For a fully specified JSON config, use `nanodllm-paper-train --config FILE`. The earlier preview's `nanodllm-pretrain` and `pretrain.py` entry points retain their original trainer and config conventions. See [getting started](docs/getting-started.md) for that path.

## Reading the comparisons

The paper reports separate full-sequence and block comparisons. In the block experiment, the selected recipe spends 10 tokens per counted parameter on the AR parent and 30 on diffusion continuation. Its scratch comparator spends all 40 on diffusion. The denominator includes the transformer matrices and output head and excludes input embeddings. Budgets count allocated tokens, including padding. A lower reconstruction bound, stronger next-token predictions, higher task accuracy and fluent sampled text measure different behaviors. The released panels preserve each metric, conditioning convention and token budget.

Code is licensed under the [MIT License](LICENSE). See [third-party notices](THIRD_PARTY_NOTICES.md), [software citation metadata](CITATION.cff), and the [documentation index](docs/README.md).
