# Design Decisions

This file records the choices that define the public research-preview baseline.
It is a design log, not a claim that the recipe is optimal. Changes to the
model, corruption process, optimizer, sampler, or scale-up defaults should
update this file and include a reproducible comparison plan.

## Release Baseline

The default research path is a bidirectional masked discrete diffusion language
model (MDLM):

- a decoder-shaped transformer with bidirectional self-attention;
- RoPE position encoding and QK normalization;
- SwiGLU feed-forward blocks;
- an untied token embedding and output projection;
- absorbing MASK corruption;
- masked-token cross entropy with the MDLM schedule weight;
- AdamW optimization; and
- iterative mask-to-token sampling.

The built-in byte tokenizer is the portable smoke-test path. Full experiments
use a Hugging Face-compatible tokenizer directory with explicit BOS, EOS, MASK,
and PAD tokens.

## Public Recipes

- `configs/smoke_byte.json` is a tiny CPU recipe used for installation and
  end-to-end checks.
- `configs/climbmix_mdlm_small.json` is a bounded single-accelerator starting
  point. Its batch size, learning rate, and horizon are examples to calibrate
  on the actual hardware, not published optima.
- `configs/climbmix_mdlm_d20.json` preserves the intended 20-layer scale-up
  shape. It is deliberately labeled provisional and should not be launched
  without resolving data paths, memory, global batch, horizon, and evaluation
  budget.
- `configs/mdlm_bayes_analytic_audit.json` is a self-contained CPU audit of the
  synthetic corruption and empirical Bayes calculation. It does not train a
  model.

Every normal run writes an architecture fingerprint. A change in the resolved
architecture, initialization contract, attention scale, or parameter inventory
changes that fingerprint.

## Data And Tokenizer

The scale-up corpus is
[`karpathy/climbmix-400b-shuffle`](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle).
For a complete-corpus experiment, reserve `shard_06542.parquet` for validation
and exclude it from both model training and tokenizer fitting.

The project-specific 32,768-token byte-BPE tokenizer is reproducible from the
checked-in `scripts/train_climbmix_tokenizer.py`. It reserves twelve control
tokens inside the vocabulary and also exports an ordinary-text backend so
literal strings that resemble control tokens remain ordinary corpus text.
See `docs/tokenizer-training.md` for the pinned dataset revision and complete
procedure.

Dataset and tokenizer paths remain explicit config values. Public configs use
environment-variable expansion instead of assuming a filesystem layout.

## Architecture

The d20 scale-up shape uses 20 layers, hidden width 1280, 10 attention heads,
context length 2048, and a 4x SwiGLU hidden width. Its conventional attention
scale is `1 / sqrt(head_dim)`.

The implementation supports additional model and objective families for
research comparisons, including uniform corruption, BD3LM scaffolding, Muon,
auxiliary heads, and several diagnostic losses. They are opt-in experimental
surfaces. Their presence in the config schema does not make them supported
defaults.

Absorbing MDLM does not receive an explicit scalar time input in the baseline.
The visible MASK pattern carries corruption-level information. This reasoning
does not automatically transfer to uniform replacement corruption, where
corrupted tokens are not marked; uniform-corruption results must therefore be
interpreted as separate empirical experiments.

## Optimization And Scale

AdamW is the portable baseline. Optimizer epsilon, betas, weight decay, gradient
clipping, warmup, warmdown, and final learning-rate fraction are always recorded
in the resolved config.

The trainer can derive a token horizon from a target parameter-to-data ratio and
can scale total batch and learning rate, but automatic scaling is a convenience,
not a substitute for a hardware and stability study. Public scale-up recipes
keep uncertain fields visibly provisional.

## Evaluation

The primary absorbing-MDLM validation quantity is a shared-sequence-time,
schedule-weighted objective reported in bits per byte. Fixed-noise
reconstruction CE and generation diagnostics are separate measurements.

Generation metrics are canaries for collapse, repetition, missing EOS, or
blocked-token leakage. Self-reconstruction scores are not external quality
judgments. Auxiliary judge-model scoring is optional and should record the exact
model identifier and revision used.

Comparisons should use common evaluation seeds, matched token budgets, and
confidence intervals appropriate to the decision. A result from one model size,
data slice, or sampler should not be promoted into a universal default.

## Research Status

This preview publishes code and procedures, not a final model, checkpoint, or
benchmark claim. Earlier internal experiments motivated several implementation
choices, but their private execution records are intentionally not part of this
snapshot. Treat those choices as hypotheses to reproduce. Publicly supportable
claims and current limitations are listed in `docs/research-status.md`.
