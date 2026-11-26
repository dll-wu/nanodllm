# Paper checkpoint catalog

The release provides training profiles and checkpoint identities. The inspected evidence contains completion receipts for 203 profile entries, representing 200 distinct final trainer files. Three profiles reuse an endpoint from another comparison. The archived identities below come from those receipts. The inspected release materials contain no original paper weight files or public download locations. A user can train the profiles locally and export the resulting models. [Training recipes](training.md)

```sh
nanodllm-recipes list --include-paper --checkpoints
nanodllm-recipes list --group block-muon --checkpoints
```

The first command lists the editable recipes and all 203 paper profiles. Only the paper profiles have an archived checkpoint SHA-256 and byte size. The second command selects the block Muon group. Both commands show the parent and rank count. An exported inference file has its own digest and records the digest of its source trainer file.

## Models for the main comparisons

| Profiles | Intended comparison | Reported result and scope |
| --- | --- | --- |
| `block-d20-{parent,scratch,convert}-s50` | 477M plain AR10, block40 and AR10 + block30 | Held-out next-token BPB was 0.917 for scratch and 0.910 for conversion. Native reconstruction BPB was 0.882 and 0.868. Mean 21-task accuracy was 30.29% and 30.55%. One trained pair. |
| `block-d8-{parent,scratch,convert,convert-alt}-s{50,51,52}` | 59M rate and seed comparison | The .00175 conversion improved held-out next-token BPB by 0.055–0.060 relative to the matching scratch model. The .0035 alternative brought reconstruction closer to scratch. Three separately identified AR parents. |
| `full-d20-{parent,scratch,convert}` | 477M full-sequence AdamW at the largest transferred size | AR20 causal content BPB was 0.834. Scratch40 and AR20 + MDLM20 corpus NELBO were 0.966 and 0.953 BPB. One seed per endpoint. |
| `dev-ar10-mdlm30-s47`, `dev-37m-ar10-mdlm30-s47` and their seed-48 counterparts | Plain 20M and nanochat-derived 37M full-sequence backbones | Document-start NELBO was 1.355 and 1.331 BPB at seed 47, and 1.359 and 1.330 at seed 48. Parameter count and both stages of training change together. |

All historical profiles use ClimbMix pretraining, the paper tokenizer and a 1,024-token training context. The scores refer to the paper evaluation panels. The corpus and task measurements do not establish instruction-following or deployment performance. [Data and tokenizer identity](data.md)

| Selected original trainer file | Bytes | SHA-256 |
| --- | ---: | --- |
| `block-d20-scratch-s50` | 3,902,536,487 | `5a6c9457ed33a5a1acc06449a9082a0a12616eb42e396db225752c5f9b3b04f3` |
| `block-d20-convert-s50` | 3,902,536,615 | `a6b2b987b89c5d9b88c7e469e5a1c75a5fd09991f486bc666d3dadfaa60c6e42` |
| `block-d20-parent-s50` | 3,902,535,847 | `041119563bed2f9d677eae91504aff67db8571f8354a5ade9196b3ccfc7fee37` |
| `block-d8-scratch-s50` | 503,655,049 | `e36380285b11bf11fdc4e5005231f2d4b0e28297191bc20d9b39c17449d2f158` |
| `block-d8-convert-s50` | 503,655,241 | `dadcfc3272f0424d90593e76e6e14bf3aa26bda408026c587f7516445d36ae1c` |
| `block-d8-convert-alt-s50` | 503,655,241 | `8d3ad12a92093d7705ed0b97d4a6b41b5eff9418dbc621deca2741d35f6546b5` |
| `full-d20-scratch` | 5,474,279,235 | `0aec20fa908ddd47c6938d4c5bf0b2cc2bc164c45b6e88801ebe37921add2754` |
| `full-d20-convert` | 5,474,279,299 | `947b1e822dfd96b04b83b7d1c75eac06fb5b36c815241d4c656b83671ed6e602` |
| `dev-ar10-mdlm30-s47` | 176,347,601 | `85b9df11c6107c723d8504f8cead30127895ca5dc0e10d283d770cd947062519` |
| `dev-37m-ar10-mdlm30-s47` | 277,021,070 | `905c74c187780ab760cf009f1b1c0cfb8c1e45a9320e4c84b5342ec074f195ca` |

## Loading and exchange

For scoring or sharing, export a `.weights.pt` file from a trusted native training checkpoint. Supply the verified training-file digest, the local tokenizer, and the run's data-identity JSON. With a byte-tokenizer demo, omit the tokenizer and data-identity arguments. The export retains named model tensors, architecture, relevant inference settings, source digest, tokenizer identity and counters.

```sh
nanodllm-export-weights \
  --trusted-checkpoint runs/parent/model_latest.pt --source-sha256 "$TRAINING_SHA256" \
  --tokenizer-dir data/climbmix-tokenizer --data-identity runs/parent/data_identity.v1.json \
  --output models/parent.weights.pt
```

The default paper evaluation loader uses the weights-only format and checks the state dictionary and tokenizer identity. When given an original training file, use the evaluator's explicit trusted-training option only for a file whose provenance you trust. A tensor export supports inference and eligible model-state conversions. Exact training resume also needs the original optimizer, progress, RNG and loader state.
