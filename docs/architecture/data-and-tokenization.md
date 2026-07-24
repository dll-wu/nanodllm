# Data And Tokenization

`nanodllm.data` owns Parquet discovery, tokenization, sequence packing, and DDP
row-group assignment.

## Parquet Input

`data.train_data` and `data.val_data` accept:

- a Parquet file;
- a directory recursively scanned for `*.parquet`; or
- a glob such as `/data/corpus/**/*.parquet`.

Every selected file must contain `data.text_column`, which defaults to `text`.
Paths are expanded for environment variables and `~`, canonicalized, and sorted
for deterministic manifests.

If `val_data` is empty, validation row groups are split from the end of the
training manifest according to `data.val_fraction`. For a durable experiment,
an explicit validation selector is usually clearer.

`data.train_exclude_data` removes whole files after training selectors resolve.
Every exclusion must be inside the included selection, and exclusions may not
remove all training files. Canonical path checks prevent accidental
train/validation overlap.

## ClimbMix

The public scale-up source is
[`karpathy/climbmix-400b-shuffle`](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle).
It contains Parquet shards named `shard_00000.parquet` through
`shard_06542.parquet`.

For the complete recipe:

- select the corpus directory for training;
- exclude `shard_06542.parquet`;
- select that shard explicitly for validation; and
- use `val_fraction=0`.

The same shard is excluded when fitting the tokenizer. This keeps tokenizer and
model fitting from reading the validation reservation.

For a systems-only test, downloading one or a few shards and using a
deterministic row-group split is sufficient. Do not compare the resulting
quality metrics with a full-corpus run.

## Row Groups And DDP

The loader reads Parquet footers to build a row-group manifest, then loads only
the configured text column for the active row group. At epoch `e`, ranks receive
deterministic disjoint assignments. The assignment and its digest are included
in data identity.

Use enough row groups for the intended world size. A tiny single-row-group file
is suitable for one process, not an eight-rank test.

## Tokenizers

The built-in byte tokenizer is dependency-free and deterministic:

- byte IDs: `0..255`;
- BOS: `256`;
- MASK: `257`;
- EOS: `258`; and
- PAD: `259`.

Use it for smoke tests, loader experiments, and controls where vocabulary scale
is not the question.

The Hugging Face adapter loads a normal local `save_pretrained()` directory
without remote code. It requires MASK and EOS IDs. A validated
`token_bytes.json` is preferred for bits-per-byte accounting.

The ClimbMix 32k tokenizer can be reproduced with
`scripts/train_climbmix_tokenizer.py`; see `docs/tokenizer-training.md`.

## Packing

Documents are encoded without automatic special tokens, optionally prefixed
with BOS, suffixed with EOS, concatenated into a rolling stream, and sliced into
fixed `model.max_seq_len` rows. Documents may cross row boundaries. Row
boundaries do not synthesize EOS, and document tails are not discarded.

The emitted batch contains:

- `x_0`: clean token IDs;
- `loss_mask`: positions eligible for corruption and reconstruction loss;
- `attention_mask`: positions visible to the model;
- optional position and document-boundary tensors; and
- loader epoch metadata.

BOS is normally visible but not lossable. EOS is normally both visible and
lossable. Padding is neither.

## Portable Smoke Corpus

The repository includes original tiny text and a deterministic conversion
script:

```bash
python scripts/prepare_tiny_corpus.py
```

This writes the ignored `data/tiny_corpus.parquet` expected by
`configs/smoke_byte.json`. It is for plumbing checks only.

## Data Identity

Preview a run's identity with:

```bash
export NANODLLM_DATA_DIR=/path/to/climbmix-400b-shuffle
export NANODLLM_TOKENIZER_DIR=/path/to/climbmix-32k

python scripts/data_identity.py \
  --config configs/climbmix_mdlm_small.json \
  --world-size 1 \
  --output /tmp/nanodllm-data-identity.json
```

Pass the same `--set` overrides used at launch. Metadata mode records canonical
paths, sizes, nanosecond mtimes, Parquet footer structure, split membership,
tokenizer hashes, packing settings, and rank assignment. Full-SHA mode also
hashes every dataset byte and may be expensive on a large corpus.

Tokenizer files are always content-hashed because they are small and directly
define token identities.

## Byte Accounting

NELBO is reported in bits per UTF-8 byte. Byte-tokenizer content IDs count as
one byte; special tokens count as zero. For subword tokenizers,
`token_bytes.json` records the raw-byte length of every token. Special tokens
should remain zero-byte so scaffolding does not improve the metric.
