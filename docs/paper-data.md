# Corpus and tokenizer for paper training

Use the detokenized and document-shuffled Nemotron-ClimbMix redistribution at revision `915333b4f8b8684f39aeaafea600fea6f43fb703`. Prepare a local directory with the 6,543 consecutively numbered files `shard_00000.parquet` through `shard_06542.parquet` at its root. The recorded inventory totals 599,758,882,644 bytes. Each file supplies the `text` column. The last *named* shard, `shard_06542.parquet`, is reserved for validation and excluded explicitly from training and tokenizer fitting. The local directory should contain only this release's Parquet shards. The trainer sorts canonical file paths before shuffling row groups. Use regular files, hardlinks or symlinks whose resolved paths preserve numeric shard order. [Pinned corpus revision](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/tree/915333b4f8b8684f39aeaafea600fea6f43fb703)

```sh
python -m nanodllm.paper_recipes verify-data \
  --corpus data/climbmix --tokenizer data/climbmix-tokenizer
```

This checks the four training tokenizer files by SHA-256 and checks the recursive Parquet inventory by relative name and byte size. The portable shard-list digest is `e4c794c66628e2518d91992bc8960981cd187d6fb4ee4724ae2b2465d5350145`. Its input is the concatenation of the lexically sorted lines `filename SPACE decimal_size LF`. A matching inventory is a metadata check. Same-size changes to shard contents require independent file hashes to detect. Historical dataset snapshots contain site-specific file identities. The recipes rebuild local metadata using the same named split and trainer discovery order.

## Rebuild the 32k tokenizer

The release includes the fitting script. The local tokenizer evidence had no artifact-specific redistribution license, so this release supplies the recipe and verified hashes. With the pinned build environment already provisioned, the following command runs offline from the repository root. Choose a new output directory. [Tokenizer build requirements and corpus acquisition](tokenizer-training.md)

```sh
PYTHONHASHSEED=0 RAYON_NUM_THREADS=16 uv run --offline --script scripts/train_climbmix_tokenizer.py \
  --data-dir data/climbmix --output-dir data/climbmix-tokenizer \
  --vocab-size 32768 --max-chars 2000000000 --document-char-cap 10000 \
  --text-column text --validation-shard shard_06542.parquet \
  --expected-first-shard 0 --expected-last-shard 6542 --model-max-length 2048 \
  --dataset-revision 915333b4f8b8684f39aeaafea600fea6f43fb703 \
  --nanochat-revision 92d63d4e8bb4df75c3b71618f31ddde2378b2bcd --parity-documents 256
```

The reference build used Python 3.13.14, rustbpe 0.1.0, tiktoken 0.11.0, tokenizers 0.22.2, transformers 5.10.2, pyarrow 21.0.0 and blobfile 3.2.0. It traversed numbered training shards, row groups and rows in stored order. It capped each document at 10,000 Unicode code points and included the entire capped document that first crossed two billion code points. The fit consumed 758,523 documents, 2,000,005,430 code points and 2,002,901,302 UTF-8 bytes. It stopped in `shard_00008.parquet`, row group 78, row 762 (zero-based). The script's length-framed fit-sample digest is `44ecfaea974351c6f80871a88e7004592c1a8486abc37b2f590cfe6ccc232ce2`.

The byte-level BPE uses the recorded regex (SHA-256 `4513fed735f5b87ddc4efa87947e0e95130573cf663df9b458ab65e7abe017c4`) and preserves case, Unicode and whitespace. The first 32,756 IDs represent ordinary tokens. IDs 32756–32759 are BOS, EOS, MASK and PAD. The remaining IDs are user, assistant, Python and output start/end pairs, in that order. Corpus input uses the ordinary tokenizer backend, and the raw byte-length table assigns zero bytes to all 12 controls. The training context is 1,024 even though the tokenizer advertises a 2,048-token model maximum.

| Artifact | Bytes | Reference SHA-256 |
| --- | ---: | --- |
| `tokenizer.json` | 3,860,280 | `9d9c605ecd29573a761c0085b4eb8bf65f6647edb915c50ac7d540fa067de01e` |
| `tokenizer_config.json` | 465 | `2d2a3e592b34fffc229ba0c4356b84d8db4971c342737213ef823b9285328735` |
| `tokenizer_ordinary.json` | 1,512,007 | `5be8f649ad2fa26cae084a6ecc968502adb12454bca6c66ba2e9ca9dfecb6ac2` |
| `token_bytes.json` | 70,957 | `3c83f5c1250af0bd96bcb23121c7ca866aa0c94c7706f7053795a80a17f22a55` |
| `tiktoken/tokenizer.model` | 549,398 | `41cbb11fc11b7393de5475da2102ad801f9299e24f9f96f87a150b3690a88f49` |

The first four files determine the paper tokenizer content identity, `371365805013d973e14cd6f024b628e1dc8232e6afc860d025ede3f1438fe00d`. Two local copies matched the table. The original corpus inventory file had SHA-256 `d7fb34b3ac63e1f8716419d3dec0f3f241493e8a53840d6f0612c1eb2889817d` and the build receipt had SHA-256 `82a860e393fbb88d5c3c121cf9fcc3cbbdccc1ed1214201fd1c48aa38acc8f44`. Those last two files record local build metadata and will differ across machines.

## Packing and exposure

The full-sequence AdamW series uses continuous, document-segmented packing. It carries an unfinished document into the next row. The best-fit 20M development and block Muon series buffer 1,000 documents. They use the longest buffered document that fits. If none fits, they take the prefix of the shortest buffered document and discard its remaining suffix. Attention is segment-local and positions restart per segment. Natural EOS is a target, BOS stays visible and unscored, and a cut does not create an EOS. Read each profile's shuffle seed independently of the model and training seeds. Conversion starts a fresh stream from the child's settings. [Training profiles](paper-recipes.md)

