# Training the paper models

Use `nanodllm-recipes` (or `python -m recipes.launch` from the checkout) to resolve a frozen paper profile. A profile selects one training stage, including its model, objective, optimizer, seeds, batch shape, schedule and rank count. The catalogs under `paper/profiles/` contain section defaults and per-profile overrides. The CLI combines these with the pinned `nanodllm.runtime.config` schema. It leaves `optim.nanochat_checkout` empty so the runtime uses its bundled, hash-checked optimizer.

```sh
export CUBLAS_WORKSPACE_CONFIG=:4096:8
nanodllm-recipes list --include-paper
nanodllm-recipes list --group block-muon --checkpoints
nanodllm-recipes materialize full-d8-scratch \
  --corpus data/climbmix --tokenizer data/climbmix-tokenizer \
  --run-dir runs/full-d8-scratch --output-dir resolved
```

The JSON receipt gives the config path, SHA-256 of the exact serialized config, rank count and parent profile. Local paths become absolute, so relocating inputs changes this file digest. Historical checkpoint hashes in the catalog refer to the original trainer files. The parameterization contract digest in the resolved config binds the model's named parameters and optimizer groups. [Checkpoint catalog](checkpoints.md)

Replace `materialize` with `launch` to verify the tokenizer and shard inventory and start the native trainer. Supply an empty run directory for each new stage. The launcher uses the original local rank count, invoking `torch.distributed.run` for multi-GPU profiles. The historical profiles require CUDA BF16 and sufficient memory for the recorded device batch. Direct launches with an already materialized JSON use `nanodllm-pretrain --config FILE` under the same rank topology. [Corpus and tokenizer setup](data.md)

For the CPU block example and editable D4 comparison, see the [reusable recipes](../recipes/README.md).

## Full-sequence AdamW

The selected `full-adamw` profiles are named `full-d{depth}-parent`, `full-d{depth}-scratch` and `full-d{depth}-convert`. Parent training is causal AR20. Scratch trains MDLM40. Conversion trains MDLM20 after AR20. TPP denotes allocated source positions per counted parameter. The denominator includes the transformer matrices and output head and excludes input embeddings. The context is 1,024 tokens and the microbatch is 16 rows per rank. Training seed, model seed and runtime seed are 42.

| Depth | Total params | Width | Root LR, parent / scratch / convert | Batch tokens, parent / diffusion | Ranks, parent / diffusion |
| --- | ---: | ---: | --- | ---: | ---: |
| 4 | 19.92M | 256 | .0136 / .0024 / .00192 | 65,536 / 65,536 | 1 / 1 |
| 8 | 58.72M | 512 | .013463313 / .005034282 / .002013713 | 131,072 / 262,144 | 1 / 1 |
| 10 | 91.10M | 640 | .00952 / .003893460 / .001557384 | 262,144 / 262,144 | 1 / 8 |
| 12 | 135.27M | 768 | .007597561 / .003107227 / .001242891 | 262,144 / 262,144 | 8 / 8 |
| 16 | 268.44M | 1,024 | .005201694 / .002127373 / .000850949 | 262,144 / 262,144 | 8 / 8 |
| 20 | 477.10M | 1,280 | .003821596 / .001562944 / .000625178 | 262,144 / 262,144 | 8 / 8 |

The JSON retains full-precision learning rates. At D4 the plain backbone uses ordinary parameterization. Larger shapes use the recorded AR or MDLM CompleteP rule. D8 uses the AR checkpoint actually imported by the selected child. D10 and D12 select the coupled AdamW centers, with distinct momentum coefficients and decay for scratch and conversion. D16 and D20 transfer those settings to larger shapes. The `full-bracket` group supplies the additional measured rates and fixed-AdamW comparisons at D10/D12. The selected AR stages warm down over 30% of the schedule and the diffusion stages over 10%. Each stage uses its own 5% warmup.

## Four-size block Muon series

Use `block-d{depth}-{parent,scratch,convert}-s{seed}`. The selected parent is plain AR10. Scratch is block40 and conversion is block30 after AR10. Both diffusion arms use causal-clean attention, a reference block length of 32 and mixed training partitions with lengths 8, 16, 32 and 64. The trained partition seed and the evaluation reference length have separate config fields.

| Depth | Seeds | Root matrix LR, parent / scratch / convert | Batch tokens, parent / diffusion | Ranks, parent / diffusion |
| --- | --- | --- | ---: | ---: |
| 4 | 48 | .014 / .007 / .007 | 65,536 / 65,536 | 1 / 1 |
| 8 | 50, 51, 52 | .014 / .007 / .00175 | 65,536 / 262,144 | 1 / 1 |
| 12 | 50 | .014 / .007 / .00175 | 65,536 / 262,144 | 4 / 8 |
| 20 | 50 | .014 / .007 / .00175 | 65,536 / 262,144 | 8 / 8 |

The D20 parent uses eight rows per device. The other parents and all diffusion arms use 16. Every stage preserves its embedding and head AdamW rates, matrix optimizer scope, model compilation flag and stage-specific shuffle seed. D4 and D8 block children use eager networks. D12 and D20 children use compiled networks. The two D20 diffusion configs start at update zero. Their historical runs resumed with the same settings and eight ranks, and both profiles cover the full accepted training budget.

At 59M, `block-d8-convert-alt-s50`, `-s51` and `-s52` use .0035. The two conversion rates share a parent **within each seed**. The original training receipts and pre-update parameter hashes identify three different parents across seeds. This corrects the shared-parent wording in the manuscript caption. The `block-bracket` group includes the other measured discovery settings.

## 20M full-sequence development and 37M backbone

The `development` group covers the best-fit optimizer comparison, allocation and continuation-rate search, schedule and batch screens, fresh-seed confirmations, and their AR parents. The historical 36-cell AR and 36-cell MDLM batch/rate/horizon grids use the `development-adamw-grid` group. Each grid entry preserves the recorded training objective and cooldown, including reused runs.

| Seed-47 development endpoint | Profile | Stage budget |
| --- | --- | --- |
| Native AdamW | `dev-mdlm-native-s47-center` | Scratch20 |
| Role-specific AdamW | `dev-mdlm-adamw-s47-bestfit-low` | Scratch20 |
| Matrix Muon | `dev-mdlm-muon-s47-bestfit-center` | Scratch20 |
| Longer Muon scratch | `dev-ar0-mdlm40-s47` | Scratch40 |
| Best plain allocation | `dev-ar10-mdlm30-s47` | AR10 + MDLM30 |
| Nanochat-derived backbone | `dev-37m-ar10-mdlm30-s47` | AR10 + MDLM30 |

The allocation screen includes 0/40, 5/35, 10/30, 20/20, 30/10 and 35/5. Rates at seed 47 are .014 for AR and .007 for MDLM, with extra conversion-rate tests at 10/30 and 20/20. Seeds 48/49 confirm allocation. Seeds 47/48 compare the plain and nanochat-derived backbones. The 36,700,242-parameter variant has two heads and two additional token-value tables. Its parent uses the same variant. The recorded stage counts give 115,343,360 AR positions and 346,030,080 MDLM positions, with 11,534,384 counted parameters for the variant. The plain model has 19,922,944 total and 11,534,336 counted parameters.

## Initializing a conversion

Use `list` to identify the parent, then train that profile or use its identified checkpoint. The child copies model state and starts its own optimizer, scaler, training RNG, schedule, counters and data stream. A parent can revisit training text in the child stage. The source-token budget includes both stages.

For an exported parent, bind the exact SHA-256 of the `.weights.pt` file. To materialize the config before the file is present, also supply `--parent-bytes`. For a locally reproduced parent, the expected digest and size refer to the new file. A local reproduction can produce different bytes from the archived checkpoint.

```sh
nanodllm-recipes materialize block-d8-convert-s51 \
  --corpus data/climbmix --tokenizer data/climbmix-tokenizer \
  --parent models/block-d8-parent-s51.weights.pt --parent-sha256 "$PARENT_SHA256" \
  --run-dir runs/block-d8-convert-s51
```

For an original native training file, `--parent` alone selects the catalog's expected SHA-256 and byte size. Launching with such a file also requires `--trusted-training-parent` because trainer checkpoints contain Python pickle state. The trainer checks the digest on the same bytes it loads. Use checkpoints from the parent profile listed for that child.
