# Paper corpus evaluation

Use `python -m nanodllm.paper_eval` to prepare a panel once, score local paper checkpoints on it, and compute paired differences. The evaluator calls the native transformer in `nanodllm.paper_runtime`. A score report contains per-example nats and bytes, checkpoint and panel digests, precision, and the fixed plan identities. Prepared panels contain reconstructable corpus tokens and belong in local data storage.

## Reference inputs

Supply the paper's 32,768-token ClimbMix tokenizer, including its ordinary BPE and token byte table. The reference tokenizer content digest is `371365805013d973e14cd6f024b628e1dc8232e6afc860d025ede3f1438fe00d`. The evaluator checks the supplied tokenizer directory against the panel and the portable weights. [Tokenizer rebuild and file hashes](paper-data.md)

All four panels use the `text` column of `shard_06542.parquet` from Karpathy's document-shuffled ClimbMix redistribution. The reference shard is 91,699,537 bytes with SHA-256 `769fe59d108dfd2cfa186c63173b83fbfb90a7adb3519519ba6eaa6ca9889f94`. Obtain this single shard from the pinned dataset revision. Preparation verifies the shard, tokenizer and resulting panel. The versioned manifest records the selected block documents in scoring order using zero-based Parquet row addresses, document SHA-256, length stratum, and source continuation length. [Pinned validation shard](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/blob/915333b4f8b8684f39aeaafea600fea6f43fb703/shard_06542.parquet) · [Panel manifest](../evaluation/panels/corpus-v1.json)

| Panel | Reference sampling unit | Readout |
| --- | --- | --- |
| `packed` | First 1,024 complete validation rows, each 1,024 tokens | Exact shifted AR content BPB or four-plan MDLM content NELBO |
| `bos` | 1,680 true document starts in those rows | AR content BPB or four-plan MDLM content NELBO from BOS to the next segment or row boundary |
| `dev32` | 16 short and 16 long documents | First 32 next tokens on both canvases, plus native reconstruction of up to 128 gold tokens for block diffusion |
| `holdout256` | 128 short and 128 long documents | Same block readouts, on a separate frozen selection |

The packed stream strips surrounding whitespace, tokenizes ordinary content, and wraps each document with BOS and natural EOS. A document can continue into the next row. Each row-local segment resets positions to zero and receives its own attention mask. The BOS panel excludes continued fragments and two starts with zero content bytes. Its 571,658 content tokens account for 2,697,287 bytes.

For block panels the 192-token prefix includes BOS. The length stratum uses the original continuation including natural EOS. Short continuations have 32–128 tokens and long continuations have at least 512. The development selection scanned row groups 50–57, alternating eligible documents within each stratum between prediction and generation. The holdout selection shuffled all row addresses in groups 58–82 using Python seed 20261120 before screening. It excluded duplicate 64-body-token prefixes, the BOS corpus prefixes and both earlier prompt panels. Preparation reconstructs all 1,543 distinct eligible BOS prefixes and verifies their set digest. The 96 earlier opened prompt prefixes are a subset of that set. A further 64 hashes identify the development prediction and generation prefixes. The holdout must be disjoint from both sets. Replaying the selected addresses also preserves the original document IDs used to key the noise generator. A hash of all selected prefixes, continuations and byte denominators checks the reconstructed targets.

## Prepare, score and compare

The examples assume local inputs under `data/` and exported `.weights.pt` files under `checkpoints/`. The CLI discovers the manifest in the checkout or installed data directory. Set `--manifest` to the supplied JSON for a relocated copy. Choose `packed`, `bos`, `dev32` or `holdout256` for the first argument to `prepare`. [Checkpoint catalog and training profiles](checkpoints.md)

```bash
python -m nanodllm.paper_eval prepare holdout256 \
  --parquet data/shard_06542.parquet --tokenizer data/tokenizer \
  --output data/panels/holdout256.pt

python -m nanodllm.paper_eval score \
  --panel data/panels/holdout256.pt --checkpoint checkpoints/scratch.weights.pt \
  --tokenizer data/tokenizer --device cuda --output reports/scratch.json

python -m nanodllm.paper_eval score \
  --panel data/panels/holdout256.pt --checkpoint checkpoints/conversion.weights.pt \
  --tokenizer data/tokenizer --device cuda --output reports/conversion.json

python -m nanodllm.paper_eval compare \
  --candidate reports/conversion.json --reference reports/scratch.json \
  --output reports/conversion-minus-scratch.json
```

Scoring defaults to the restricted, weights-only loader and strict model-state loading. Pass `--checkpoint-sha256` to bind a specific checkpoint file. The separate `--trusted-training-checkpoint` flag allows full pickle deserialization for an original local trainer checkpoint you trust. For HF training checkpoints, keep the matching v1 or v2 data identity sidecar beside the checkpoint, or pass its location with `--data-identity`. The CPU path uses float32. The paper's packed protocol used CUDA bfloat16 autocast, device batch 16, and TF32 disabled. BOS scoring used float32 and device batch 4. Block prediction used float32, one prompt at a time, with four native draws per forward. Kernel, device, precision and batch shape can affect floating-point results.

## Score interpretation

All aggregate BPB scores divide summed nats by `ln(2) × summed gold bytes`. The byte table supplies each token's contribution. Special tokens contribute zero bytes. For AR, targets need a predecessor in the same segment. This excludes the first token of a continued packed row. For full-sequence diffusion, each row or BOS passage gets one permuted midpoint of the equally spaced time strata. Every eligible token, including natural EOS, can be corrupted. Masked content CE receives weight `1/t` and the denominator includes all content bytes. Both full-sequence panels use seeds 2026091101 through 2026091104. Scoring checks the reference time and corruption digests for each seed. The packed diffusion denominator is 4,918,646 bytes. Shifted AR scores 4,913,971 bytes on the same rows. The primary diffusion result is `nelbo/mean_plans`.

For each of the first 32 continuation tokens the scorer supplies the gold past. `common/legacy` appends a single MASK. `common/block_end` appends masks through the end of the current 32-token block. The two inputs coincide at the final block position. Both metrics sum content NLL and content bytes. A causal AR parent uses its last prefix logit for both readouts.

For `native/mean_eval_seeds`, each block conditions on the prefix and all earlier gold blocks. Each evaluation seed (2026110201 and 2026110202) generates 16 permuted midpoint times per block and Bernoulli token masks. Noise is keyed by document ID, seed and block offset before batching. The native numerator includes loss on a masked natural EOS, even though EOS contributes zero bytes. The denominator covers the full gold continuation up to 128 tokens. Averaging the two seed estimates gives the reported native reconstruction BPB.

`compare` reports candidate minus reference BPB, with linear percentile endpoints at 2.5% and 97.5%. It checks panel identity, row order, strata, byte counts and compatible readouts. Each draw reuses the same rows for both checkpoints and retains the fixed corruption plans. Holdout and D4 development comparisons use 2,000 Python bootstrap draws with seed 2026111201. D8 development uses 5,000 NumPy draws with seed 20261116 in short–long order. D12 and D20 development use 5,000 NumPy draws with seed 20261118 in long–short order. For full-sequence comparisons the tool also offers a conditional 2,000-draw interval over packed rows or BOS passages. These intervals describe example sampling at fixed fitted weights and fixed evaluator noise.

## Use another corpus

Supply a local Parquet file with a string `text` column and add `--custom` to `prepare`. Omit `--tokenizer` for the built-in byte tokenizer when evaluating a matching byte checkpoint. For example, this prepares a four-row smoke panel for a model with at least 32 positions of context.

```bash
python -m nanodllm.paper_eval prepare packed --custom \
  --parquet data/validation.parquet --count 4 --width 32 --output data/panels/tiny.pt
python -m nanodllm.paper_eval score --panel data/panels/tiny.pt \
  --checkpoint checkpoints/tiny.weights.pt --batch-size 1 --output reports/tiny.json
```

For custom packed or BOS panels, `--count` and `--width` select the number and width of complete source rows. A packed scoring batch must divide the row count. For custom block panels, `--per-stratum` controls the number of documents. Development takes the first eligible unique prefixes, up to 16 per stratum. Custom holdout reserves the first 16 of each stratum and selects from subsequent distinct prefixes. These custom splits are disjoint when prepared from the same ordered file with the same tokenizer. Reports identify custom runs with `reference: false`.
