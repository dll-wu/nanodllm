# Reproducing the NanoDLLM paper

## Verify the figures and tables on CPU

From the repository root, with Python 3.11+ and NumPy, run

```sh
python -m paper.companion --check
```

This hashes ten retained JSON snapshots, recomputes the statistics and compares 67 TeX files against the edition manifest. They cover seven numbered figures, thirteen numbered tables and supplementary exhibits. The check runs offline using the losses, counts and aggregates from the original experiments. For the TeX files and unrounded `statistics.json`, run `python -m paper.companion --output build/paper-results`. The results guide explains which intervals can be recomputed from paired observations and which summaries use historical aggregates. [Results guide](results.md) · [SHA-256 manifest](companion/manifest.json)

## Train an exact paper profile

Use the frozen profiles to repeat a training stage with its recorded model, optimizer, token budget, seeds and rank count. Supply the pinned ClimbMix shards and tokenizer locally. For example, with the NanoDLLM commands installed and a CUDA BF16 device available, the following trains the 59M full-sequence scratch arm. [Corpus and tokenizer setup](data.md) · [Frozen profiles](profiles/)

```sh
nanodllm-recipes list --group full-adamw --checkpoints
nanodllm-recipes launch full-d8-scratch \
  --corpus data/climbmix --tokenizer data/climbmix-tokenizer \
  --run-dir runs/full-d8-scratch
```

Use `materialize` in place of `launch` to inspect the resolved config and its digest before training. For a conversion, train the listed AR parent first or supply its verified checkpoint. A new fit produces its own checkpoint and measurements. The training guide gives the parent arguments and rank requirements for each series. [Training guide](training.md) · [Checkpoint identities](checkpoints.md)

For evaluation, the corpus panel identifies the held-out inputs, and the two task manifests specify the task roster and generation panel. The release notes record the source revisions and checks for this local candidate. [Corpus panel](panels/corpus-v1.json) · [Task roster](tasks/core21.json) · [Generation panel](tasks/generation32.json) · [Release notes](release.md)
