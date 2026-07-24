# Reproducing the ClimbMix Tokenizer

`scripts/train_climbmix_tokenizer.py` trains the 32,768-token byte-level BPE
used by the ClimbMix recipes and exports a standard Hugging Face fast
tokenizer. The fitting algorithm follows NanoChat's RustBPE recipe: regex
pretokenization, no Unicode normalization, a 10,000-code-point cap per
document, and the first complete document that crosses a 2-billion-code-point
sample threshold.

## Corpus Identity

Use
[`karpathy/climbmix-400b-shuffle`](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle)
at the immutable revision
`915333b4f8b8684f39aeaafea600fea6f43fb703`. The dataset card declares an MIT
license. At that revision the repository is approximately 600 GB and contains
6,543 Parquet files named consecutively from `shard_00000.parquet` through
`shard_06542.parquet`.

The tokenizer fit reads shards in numeric order and reserves
`shard_06542.parquet` for validation. That reservation is part of the recipe:
the last shard must not contribute text to BPE fitting.

Install the current Hugging Face CLI, choose storage with at least 650 GB free
for the corpus and download metadata, and download the pinned snapshot:

```bash
python3 -m pip install --upgrade huggingface_hub

hf download karpathy/climbmix-400b-shuffle \
  --repo-type dataset \
  --revision 915333b4f8b8684f39aeaafea600fea6f43fb703 \
  --local-dir /path/to/climbmix-400b-shuffle
```

The repository is public, so a Hugging Face token is not normally required.
`hf download ... --dry-run` can report the transfer before downloading it.
Keep the `.cache/huggingface` metadata created inside `--local-dir`; the
training script ignores non-Parquet files.

## Runtime Requirements

The script supports Python 3.10 through 3.13. Python 3.14 is excluded because
the pinned `tiktoken` release does not support it. The script's PEP 723 header
pins the build dependencies (`blobfile`, `pyarrow`, `rustbpe`, `tiktoken`,
`tokenizers`, and `transformers`), so the simplest runner is
[`uv`](https://docs.astral.sh/uv/):

```bash
uv run --script scripts/train_climbmix_tokenizer.py --help
```

`uv` creates an isolated environment and installs those exact versions on the
first invocation. A GPU is not used. RustBPE parallelizes work across CPU
cores; set `RAYON_NUM_THREADS` before the command when a scheduler assigns a
fixed CPU count.

The full corpus needs approximately 600 GB of persistent storage. Leave
additional room for Hugging Face download metadata and the tokenizer output.
The fit streams Parquet rows but BPE training still needs substantial host RAM;
use a multi-core machine with tens of GB of available memory and monitor the
first attempt. Runtime and peak memory vary significantly with CPU, storage,
and thread count. The upstream NanoChat report describes a 2-billion-character
RustBPE fit as roughly a minute on its reference machine; treat that as an
implementation reference, not a portable runtime promise.

## Full 32k Build

Run this command from the repository root after replacing the two path
placeholders. The output directory must not already exist.

```bash
uv run --script scripts/train_climbmix_tokenizer.py \
  --data-dir /path/to/climbmix-400b-shuffle \
  --output-dir /path/to/climbmix-tokenizer-32k \
  --vocab-size 32768 \
  --max-chars 2000000000 \
  --document-char-cap 10000 \
  --text-column text \
  --validation-shard shard_06542.parquet \
  --expected-first-shard 0 \
  --expected-last-shard 6542 \
  --dataset-revision 915333b4f8b8684f39aeaafea600fea6f43fb703 \
  --model-max-length 2048 \
  --parity-documents 256 \
  --nanochat-revision 92d63d4e8bb4df75c3b71618f31ddde2378b2bcd
```

The script refuses an incomplete or non-consecutive shard inventory, refuses
to overwrite an existing output directory, and checks that every consumed row
contains a string. It trains on the sorted training shards in stored row-group
and row order. Each document is truncated before it is counted, and the
complete document that first makes the cumulative code-point count greater
than `--max-chars` is included. These details matter when comparing two builds.

The total vocabulary contains 32,756 mergeable byte tokens followed by these
12 special tokens in order:

```text
<|bos|> <|eos|> <|mask|> <|pad|>
<|user_start|> <|user_end|>
<|assistant_start|> <|assistant_end|>
<|python_start|> <|python_end|>
<|output_start|> <|output_end|>
```

## Small End-to-End Check

The following check downloads about two shards instead of the full corpus. It
exercises Parquet streaming, validation reservation, RustBPE training, the
Hugging Face export, and parity validation. It deliberately trains a small
throwaway vocabulary and does **not** reproduce the release tokenizer.

```bash
hf download karpathy/climbmix-400b-shuffle \
  shard_06541.parquet shard_06542.parquet \
  --repo-type dataset \
  --revision 915333b4f8b8684f39aeaafea600fea6f43fb703 \
  --local-dir /path/to/climbmix-tokenizer-check

uv run --script scripts/train_climbmix_tokenizer.py \
  --data-dir /path/to/climbmix-tokenizer-check \
  --output-dir /path/to/climbmix-tokenizer-check-output \
  --vocab-size 2048 \
  --max-chars 1000000 \
  --document-char-cap 2000 \
  --validation-shard shard_06542.parquet \
  --expected-first-shard 6541 \
  --expected-last-shard 6542 \
  --parity-documents 32
```

Use a new output path when repeating the check, or remove an earlier throwaway
output intentionally before rerunning it.

## Outputs and Verification

The output directory is directly loadable by Transformers:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "/path/to/climbmix-tokenizer-32k",
    local_files_only=True,
    trust_remote_code=False,
)
assert len(tokenizer) == 32_768
assert tokenizer.mask_token_id == 32_758
```

Important files are:

- `tokenizer.json` and the normal Hugging Face configuration files: the model
  tokenizer, including special-token behavior.
- `tokenizer_ordinary.json`: a backend that always treats strings such as
  `<|mask|>` as ordinary text rather than activating control tokens.
- `token_bytes.json`: bytes per token for tokenizer-invariant bits-per-byte
  metrics; special tokens have length zero.
- `corpus_inventory.v1.json`: expected shard names and sizes, the declared
  pinned dataset identity, and a canonical inventory hash.
- `tokenizer_build.v1.json`: script hash, complete scientific arguments,
  consumed-shard and validation-shard hashes, sample hash and terminal row,
  dependency versions, parity results, and output artifact hashes.

The build records intentionally omit local absolute paths, executable paths,
hostnames, file mtimes, and operating-system fingerprints. This makes them
safe to compare across research environments without losing the content hashes
and parameters needed to audit the build. The corpus revision is declared
provenance; the consumed-file SHA-256 values and deterministic fit-sample hash
are the local byte-level checks. Compare those fields first if independent
builds disagree.

The fitting recipe was adapted from
[`karpathy/nanochat`](https://github.com/karpathy/nanochat) at commit
`92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`, under its MIT license.
