# Reproducibility

The trainer records enough state to make differences inspectable, but exact
replay still depends on hardware and software outside the checkpoint.

## Minimum Run Record

Retain:

- source commit or immutable source archive digest;
- original and `resolved_config.json`;
- `architecture_fingerprint.json`;
- `data_identity.v1.json`;
- tokenizer directory and `tokenizer_build.v1.json`, or their content hashes;
- `run_summary.json`, `train_log.jsonl`, `eval_log.jsonl`, and `metrics.json`;
- relevant checkpoints and their SHA-256 digests;
- Python, PyTorch, CUDA, driver, and dependency versions;
- GPU model/count, CPU allocation, and memory;
- launch command and non-secret runtime settings; and
- the interpretation or decision made from the run.

Never put credentials or unrestricted environment dumps into a run record.

## Config Identity

The source JSON is not the final recipe. CLI overrides, derived token budgets,
batch arithmetic, model vocabulary padding, and automatic learning-rate scaling
change the executed run. Compare `resolved_config.json` and `run_summary.json`.

The architecture fingerprint binds the resolved model and named-parameter
inventory. Use it to detect changes that a high-level recipe name can hide.

## Data Identity

Metadata identity records canonical paths, file membership, sizes, mtimes,
Parquet footer structure, tokenizer hashes, packing settings, and DDP
assignment. It detects ordinary replacements but is not a full corpus content
hash.

Use full-SHA identity when immutable byte-level evidence justifies reading the
entire corpus. On a 600 GB dataset this is a deliberate, expensive operation.

Explicitly reserve validation data. Do not rely on “the last shard” when only a
partial local copy is present.

## Tokenizer Identity

Tokenizer bytes determine vocabulary size, special IDs, model parameter count,
token budget, and bits-per-byte accounting. Record:

- all tokenizer file hashes;
- fitting corpus revision and selection;
- fitting stop rule and sample digest;
- vocabulary and reserved tokens;
- ordinary-text encoding parity;
- token-byte table; and
- script/dependency versions.

The checked-in tokenizer trainer emits these fields in
`tokenizer_build.v1.json`.

## Randomness

The config exposes model, training, shuffle, corruption, and evaluation seeds.
Keep them distinct in experiment plans:

- a model seed identifies initialization;
- a training seed identifies stochastic optimization and corruption;
- a shuffle seed identifies data order; and
- evaluation seeds identify common-noise plans.

For paired comparisons, share evaluation plans and nuisance seeds while changing
only the declared treatment. Use fresh seeds for confirmation to avoid selecting
on one trajectory.

## Distributed Replay

World size, device batch, accumulation, row-group assignment, prefetch state,
and per-rank RNG affect the trajectory. The resume contract binds them and
verifies the replayed next batch.

Changing world size is a new run, even if the nominal global batch is unchanged.
It may change reduction order, data order, and floating-point behavior.

## Numeric Determinism

Bitwise equality is not guaranteed across GPU architectures, CUDA/PyTorch
versions, compiler backends, or distributed topologies. For a stricter
same-environment replay:

- use the deterministic runtime settings;
- disable cuDNN benchmarking and TF32 where required;
- keep dtype, compilation, world size, and package versions fixed;
- verify the next batch after resume; and
- compare model and optimizer state at a short horizon before trusting a long
  continuation.

Document the reproducibility level achieved: logical recipe, matched metrics,
numerically close trajectory, or byte-identical state.

## Evaluation Comparisons

Use matched tokenizer bytes, validation membership, integration range, sequence
count, seed plans, and byte accounting. Report paired differences and
uncertainty where possible.

Choose a practically meaningful margin before observing the result. A confidence
interval overlapping both “equivalent” and “meaningfully different” regions is
inconclusive, not a win.

## Artifact Retention

Keep generated artifacts outside the source repository. A compact retained run
usually needs configs, identities, summaries, logs, metrics, samples, reports,
and checkpoint hashes. Store large checkpoints separately when necessary and
record their immutable location and digest.

Do not commit entire run directories, downloaded datasets, tokenizer build
outputs, TensorBoard events, or model weights to this source repository.
