# Research Preview Status

This release is intended for code inspection, reproduction, and new
experiments. It is not a production training stack or a model release.

## Included And Exercised

The repository includes:

- an installable Python package and console entrypoint;
- portable byte-tokenizer CPU smoke data and config;
- Parquet streaming, rolling document packing, and DDP row-group sharding;
- bidirectional transformer, MDLM objective, and iterative sampling;
- AdamW and experimental Muon paths;
- structured logs, TensorBoard, architecture/data identities, checkpoints, and
  resume contracts;
- fixed-noise, shared-time NELBO, generation, and conditional diagnostics;
- a reproducible ClimbMix 32k tokenizer trainer; and
- unit/integration tests plus source-distribution and wheel checks.

The release checks demonstrate the exercised code paths in their test
environments. They do not qualify every GPU, compiler backend, config
combination, or multi-node topology.

## Public Scientific Baseline

`DESIGN_DECISIONS.md` defines a plausible MDLM baseline and a provisional d20
scale-up shape. These are starting hypotheses. This snapshot does not include
the private execution evidence that originally informed every choice, so
researchers should reproduce the relevant comparisons before treating a choice
as settled.

The self-contained Bayes analytic audit is public and reproducible, but it
covers a finite synthetic data/corruption calculation only. It says nothing
about a trained model's calibration or language quality.

## Not Included

- pretrained weights or generated model samples;
- a public benchmark leaderboard;
- a released copy of the project-trained 32k tokenizer;
- the 600 GB ClimbMix corpus;
- institution-specific scheduler, storage, or deployment automation;
- internal experiment ledgers and private artifact identifiers; or
- a claim of state-of-the-art quality, compute efficiency, or convergence.

## Known Limitations

- Full ClimbMix reproduction has substantial storage, download, tokenizer-fit,
  and accelerator requirements.
- The config schema includes experimental switches beyond the recommended
  public recipes. Not every cross-product is scientifically meaningful.
- Generation from masked language models can collapse into repetitive or
  high-frequency-token attractors even when reconstruction losses improve.
  Always inspect samples and collapse diagnostics.
- Judge-model perplexity and self-reconstruction are imperfect proxies for
  generation quality.
- Single-node DDP receives more testing than multi-node operation.
- Exact deterministic replay is environment-dependent.
- There is no stable public Python API promise yet; configs, metrics, and
  checkpoints may evolve before a 1.0 release.

## Data Responsibility

ClimbMix is an external dataset. Review its dataset card, license, provenance,
and content policy before use. This repository does not redistribute it.
Researchers are responsible for legal, ethical, privacy, filtering, and
downstream-use review appropriate to their jurisdiction and institution.

## Reporting Results

When publishing a result, include:

- code identity;
- full resolved config;
- tokenizer and dataset identity;
- token and compute budget;
- hardware/runtime details;
- evaluation plan and uncertainty;
- decoded generation examples and collapse metrics; and
- limitations and failed or inconclusive comparisons.

Please distinguish implementation verification, a result on one setup, and a
general scientific claim.
