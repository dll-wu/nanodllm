# Architecture

The repository is a thin executable shell around the importable `nanodllm`
package:

```text
CLI -> resolved config -> tokenizer/data stream -> corruption -> transformer
    -> loss/optimizer -> evaluation -> logs/checkpoints
```

## Module Ownership

- `nanodllm.cli` parses command-line arguments and dotted config overrides.
- `nanodllm.config` owns dataclass schemas, JSON loading, validation, and
  serialization.
- `nanodllm.data` discovers Parquet files, streams row groups, adapts
  tokenizers, packs document streams, and shards row groups across DDP ranks.
- `nanodllm.data_identity` records the exact dataset, tokenizer, packing, and
  rank-assignment inputs to a run.
- `nanodllm.model` implements the transformer, RoPE, QK normalization,
  attention masks, initialization, parameter accounting, and architecture
  fingerprints.
- `nanodllm.noise` implements absorbing MDLM and experimental corruption
  processes.
- `nanodllm.mdlm_objective` contains the schedule-weighted reference objective.
- `nanodllm.sampling` implements iterative denoising samplers.
- `nanodllm.eval` computes reconstruction, NELBO, generation, and diagnostic
  metrics.
- `nanodllm.training` owns distributed initialization, budgets, optimization,
  logging, checkpointing, resume, and evaluation scheduling.

`pretrain.py` only invokes `nanodllm.cli.main`. CLI behavior belongs in the
package.

## Training Flow

1. Load a JSON config and apply each `--set section.field=value` override.
2. Validate the complete resolved config before allocating the model.
3. Initialize a single process or the process group supplied by `torchrun`.
4. Resolve the tokenizer and deterministic Parquet manifest.
5. Build the model and write its architecture fingerprint.
6. Resolve the step/token budget, global batch, gradient accumulation, and
   learning-rate schedule.
7. Stream, tokenize, EOS-separate, and pack documents into fixed-length rows.
8. Corrupt only positions allowed by the batch loss mask.
9. Predict clean tokens, accumulate gradients, and step the optimizer.
10. Reduce counters across ranks; rank 0 writes metrics and checkpoints.

The loader emits `DLMDataBatch` objects containing clean token IDs, a loss mask,
an attention mask, optional position IDs, and packing metadata. This makes
padding and document-boundary behavior explicit at the model boundary.

## Model And Objective

The public baseline is a fully bidirectional transformer trained with absorbing
MASK corruption. It predicts the original token at corrupted, lossable
positions. MDLM applies the configured schedule weight in addition to
active-token normalization.

The model supports causal, block-structured, and uniform-corruption experiments,
but these are explicit research surfaces rather than hidden variants of the
baseline. BD3LM support provides structured attention and sampling scaffolding;
it is not an optimized KV-cache implementation.

The absorbing baseline has no separate scalar time input. MASK positions expose
information about the corruption level. Uniform replacement does not have that
property, so results from the uniform branch require separate empirical
validation.

## Core Invariants

- DDP ranks must receive disjoint row-group assignments for an epoch.
- Tokenizer special-token IDs and token-byte accounting are part of run
  identity.
- BOS may be visible without being a training target; EOS is normally visible
  and lossable.
- Row boundaries are not document boundaries. Documents are joined with EOS
  and may span packed rows.
- Resume must preserve trajectory-defining config, data assignment, optimizer,
  scaler, RNG, and loader state.
- Normal outputs live under `train.run_dir`. Ordinary training does not
  discover previous runs; replay and comparison modes may read a prior run only
  when it is selected explicitly in the config.
- Architecture or initialization changes must change the fingerprint.

## Extension Pattern

For a new behavior:

1. add a config field and validation;
2. thread it through the module that owns the behavior;
3. add a focused unit or integration test;
4. add a launchable recipe only if the combination is coherent;
5. update the owning document; and
6. update `DESIGN_DECISIONS.md` if the public scientific baseline changes.
