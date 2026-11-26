# Documentation

For the paper release, begin with [training profiles](paper-recipes.md), [data](paper-data.md), [corpus evaluation](paper-evaluation.md), [tasks and generation](paper-tasks-generation.md), [component comparisons](paper-components.md), [checkpoint inventory](checkpoints.md) and [paper results](paper-results.md). [Release notes](release.md) describe source provenance and local validation.

For the original preview trainer, start with the repository `README.md`, then open the guide that matches the
task:

- [`getting-started.md`](getting-started.md): install, create portable smoke
  data, run training, inspect outputs, and adapt a recipe to your own Parquet
  corpus.
- [`tokenizer-training.md`](tokenizer-training.md): reproduce the 32k ClimbMix
  tokenizer.
- [`hpc.md`](hpc.md): scheduler-neutral single-node and multi-GPU launches.
- [`reproducibility.md`](reproducibility.md): data identity, seeds, configs,
  checkpoints, and comparison hygiene.
- [`research-status.md`](research-status.md): validated functionality,
  preliminary choices, limitations, and non-claims.
- [`experiments/bayes-analytic-audit.md`](experiments/bayes-analytic-audit.md):
  a self-contained CPU audit of the synthetic MDLM corruption path.

Implementation references:

- [`architecture/architecture.md`](architecture/architecture.md)
- [`architecture/configuration.md`](architecture/configuration.md)
- [`architecture/data-and-tokenization.md`](architecture/data-and-tokenization.md)
- [`architecture/training.md`](architecture/training.md)
- [`architecture/evaluation-and-sampling.md`](architecture/evaluation-and-sampling.md)
- [`architecture/packaging.md`](architecture/packaging.md)

`DESIGN_DECISIONS.md` owns the scientific baseline and must be read before
changing model, corruption, optimizer, sampler, or scale-up defaults.
