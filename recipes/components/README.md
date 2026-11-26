# Reproducing the 20M component comparisons

The four studies use training seeds **48 and 49**. Train each conversion arm from the same completed AR10 checkpoint within its seed. Train the two AR10 parents independently. The evaluator reports teacher-forced next-token bits per byte (BPB) for both the primary block-end canvas and the one-MASK sensitivity check. The numerical companion can reproduce the published reductions without training a model. [Component measurements](../../paper/companion/data/components.json) · [Paired per-document losses](../../paper/companion/data/component-losses.json)

| Study file | Control arm | Candidate arm | Updates per seed after initialization | Published candidate minus control BPB, paired document 95% interval |
| --- | --- | --- | --- | --- |
| `bd3.json` | `fixed` | `calibrated` | 3,520 each from scratch | −0.052 [−0.067, −0.038] |
| `llada1.json` | `mdlm` | `masking` | 3,520 each from scratch | +0.009 [−0.006, +0.025] |
| `llada2.json` | `fixed` | `curriculum` | 4,928 each from AR10, then 352 each at BL32 | +0.029 [+0.021, +0.037] |
| `fastv2.json` | `token-causal` | `full` | 5,280 each from AR10 | −0.046 [−0.064, −0.030] |
| `fastv2.json` | `block-causal` | same `full` checkpoint | 5,280 each from AR10 | −0.019 [−0.033, −0.007] |
| `fastv2.json` | same `block-causal` checkpoint | `matched` | 5,280 control, 1,817 candidate from AR10 | −0.011 [−0.0236, +0.0005] |

## Training

The plain transformer has four layers, width 256, four heads, a 1,024-wide squared-ReLU MLP, a 1,024-token source context and a 32,768-token vocabulary. Its 19,922,944 parameters include 11,534,336 counted for tokens per parameter (TPP). Each update consumes 65,536 allocated source positions on one rank, in four microbatches of 16 rows. One TPP is exactly 176 updates. Best-fit packing resets positions and isolates documents. BOS is protected and natural EOS is eligible. The 10% continuation augmentation protects a sampled prefix of 12.5–75% of the row. These choices apply across arms. [Shared config](base.json)

The reference runs use BF16, Muon on body matrices and AdamW on embeddings and the output head. The matrix LR is 0.007 for diffusion and 0.014 for AR10. All schedules warm up over 5% of their own horizon. Cooldown occupies the final 30%, or 10% for BD3. Model and training RNGs start from the selected seed. Component corruption uses seed + 100003, and BD3 calibration uses seed + 200003. The data shuffle seeds are 319180585 for AR10 and BD3, 319180586 for other main phases, and 319180587 for both tails. Inspect all role-specific rates, noise policies and absolute budgets before running.

Use the public ClimbMix shard layout and the matching local tokenizer. The launcher checks the 6,543 shard names and sizes and the tokenizer input hashes. The validation shard is excluded from training. The original machine-specific inventory is replaced by local discovery. A matching filename and size alone does not establish byte identity for every training shard. [Tokenizer preparation](../../docs/tokenizer-training.md)

```sh
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export NANODLLM_DATA_DIR=/path/to/climbmix
export NANODLLM_TOKENIZER_DIR=/path/to/climbmix-tokenizer
nanodllm-components show recipes/components/fastv2.json matched --seed 48
```

For a complete scratch pair, use a new output directory per arm and seed.

```sh
for seed in 48 49
do
  for arm in fixed calibrated
  do
    nanodllm-components train recipes/components/bd3.json "$arm" \
      --seed "$seed" --output "runs/components/bd3-$arm-$seed"
  done
done
```

For a conversion pair, use the AR10 preset (1,760 updates, 115,343,360 positions). This example trains the fixed and curriculum arms at seed 48. Repeat with seed 49. Use the same parent for all four Fast-v2 arms at that seed. The launcher verifies the selected AR checkpoint's hash, seed, completed budget and tokenizer. `--trusted-ar-parent` opts into loading a full local training checkpoint.

```sh
seed=48
nanodllm-components train recipes/components/ar10.json parent \
  --seed "$seed" --output "runs/components/ar10-$seed"
parent="runs/components/ar10-$seed/main/model.pt"
parent_sha=$(python -c 'from nanodllm.weights import file_sha256; import sys; print(file_sha256(sys.argv[1]))' "$parent")
for arm in fixed curriculum
do
  nanodllm-components train recipes/components/llada2.json "$arm" \
    --seed "$seed" --output "runs/components/llada2-$arm-$seed" \
    --trusted-ar-parent "$parent" --parent-sha256 "$parent_sha"
done
```

`main/model.pt` and `main/component.json` identify a complete endpoint. In the curriculum comparison, `tail/model.pt` and `tail/component.json` identify the endpoint used in the paper. The component runner starts a fresh optimizer and schedule for each phase, records its resolved config, data identity, source checkpoint hash and accumulated work, and writes an exportable checkpoint at phase completion. Interrupted component phases require a fresh run directory and a restart of the schedule. AR10 and the native token-causal control run through the native trainer.

## What each arm changes

- **BD3.** Both scratch arms use BL32, block-causal clean attention, stratified block mask rates and inverse-time weighted CE with MASK excluded from the softmax. Both begin on the full range. At zero-based updates 352, 704, …, 3168 the calibrated arm tests [0, 1], [0, 0.5], [0.3, 0.8] and [0.5, 1] on the same eight frozen training microbatches. It selects the lowest mean of the eight within-batch sample variances of block losses. Each score uses the weighted token-loss sum divided by 32, including partial blocks. Tuning costs 32 additional forwards per search. The clipped objective keeps the inverse-time weight without correcting for interval width. At seed 48 calibration changes primary BPB from 1.359 to 1.303, but one-MASK BPB goes from 1.392 to 1.614. Seed 49 also reverses. [Noise and canvas measurements](../../paper/companion/data/components.json)
- **LLaDA 1.** The scratch MDLM control stratifies masking rates across rows on the full range and excludes MASK from the softmax. The candidate samples IID row rates on [0.001, 1], includes MASK in the training vocabulary, and crops the entire microbatch with 1% probability to a length drawn uniformly from 1 through 1,024. Cropping inserts no EOS. Both arms divide the inverse-time weighted masked CE sum by all eligible positions, including visible tokens. The loader's independent crop is disabled in both arms. [Masking preset](llada1.json)
- **LLaDA 2.0.** Both arms use block-causal clean attention and independent per-block times on [0.05, 1]. The fixed control uses BL32 throughout. The 28-TPP curriculum devotes 492 updates to the ladder 1, 4, 32, 64, 1024, holds at 1024 for 3,944 updates, then spends 492 updates on 1024, 512, 256, 128, 64, 32. Integer boundaries floor cumulative equal shares within each ramp. At full context a single noisy stream suffices. Each arm imports its own main endpoint into a fresh 2-TPP BL32 tail, quarters all three LR roots, and restarts on the same tail shuffle seed. The compressed plateaus and phase budgets adapt the published method to 20M. [Curriculum and tail preset](llada2.json)
- **Fast-v2.** The candidate rounds each source fragment up to a BL32 boundary with visible MASK padding, allocates extra physical rows as needed, draws per-block rates on [0.001, 1], and runs both complementary masks. Each eligible target uses unweighted CE at its preceding position. Padding has no direct loss. A lossable first token without a same-document predecessor can be corrupted but is excluded from projection and normalization. The historical control uses the native fixed-BL32 trainer, token-causal clean attention, and its [0.001, 1] noise settings. The revised control uses the component fixed-block trainer, block-causal clean attention and [0.05, 1] noise. Full Fast-v2 and the historical control use 1% per-row length augmentation. Revised fixed and shortened Fast-v2 disable this loader augmentation. Comparing the controls therefore measures a bundle. [Fast-v2 preset](fastv2.json)

At equal source tokens, both full conversions process 346,030,080 allocated source positions. Alignment, both views and the number of vocabulary projections raise Fast-v2's estimated matrix work per source token by about 2.9×. For physical two-stream transformer positions (X) and projected targets (A), the estimate is `3 × [(2 × 3,145,728 + 4 × 4 × 256 × 2,048) × X + 2 × 256 × 32,768 × A]`. The 1,817-update schedule was fixed from prior work counters. It consumes 119,078,912 source positions and finishes its own warmup and cooldown. Estimated child FLOPs at seeds 48/49 were 3.9204/3.9197 × 10¹⁶ for the revised control and 3.9530/3.9525 × 10¹⁶ for shortened Fast-v2. The shared parent cancels within seed. Wall time, optimizer work, pointwise operations and evaluation are outside this estimate. [Realized compute records](../../paper/companion/data/components.json)

## Scoring and comparing endpoints

The paper uses FP32 inference on 32 documents, each with a 192-token prefix, scoring 32 gold targets (1,024 targets and 4,950 content bytes overall). The dev32 manifest identifies the original rows by address and content hash. Its short and long strata each have 16 documents. Selection excluded first-64-body-token overlaps with the 1,680-passage development corpus. The compute follow-up reused the opened panel. [Public panel manifest](../../paper/panels/corpus-v1.json)

Export each endpoint using the tokenizer used in training. The component evaluator loads portable weights with the tensor-only loader and binds their recorded source hash to the endpoint receipt. Use the final tail for both LLaDA 2.0 arms. The example below scores both BD3 arms at both seeds, then rebuilds their paired contrast.

```sh
for seed in 48 49
do
  for arm in fixed calibrated
  do
    stage="runs/components/bd3-$arm-$seed/main"
    sha=$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["source_sha256"])' "$stage/component.json")
    nanodllm-export-weights --trusted-checkpoint "$stage/model.pt" --source-sha256 "$sha" \
      --tokenizer-dir "$NANODLLM_TOKENIZER_DIR" --output "$stage/model.weights.pt"
    nanodllm-components eval --weights "$stage/model.weights.pt" \
      --endpoint "$stage/component.json" --tokenizer-dir "$NANODLLM_TOKENIZER_DIR" \
      --parquet "$NANODLLM_DATA_DIR/shard_06542.parquet" --device cpu > "$stage/scores.json"
  done
done
nanodllm-components compare \
  --candidate runs/components/bd3-calibrated-48/main/scores.json runs/components/bd3-calibrated-49/main/scores.json \
  --control runs/components/bd3-fixed-48/main/scores.json runs/components/bd3-fixed-49/main/scores.json
```

The primary score sums content NLL over documents and divides by total content bytes times ln 2. Previously scored gold tokens enter the prefix, while the next unknown and its future remain masked. The secondary conditional NELBO uses up to 128 continuation tokens, 16 permuted midpoint draws per evaluation seed (2026110201 and 2026110202), and batches of four. It uses full noise support even for calibrated BD3. Full-sequence arms reconstruct the whole suffix together, while block arms condition each block on earlier gold blocks. Fast-v2 uses shifted, unweighted CE and has no reported native bound. The comparison resamples the same documents 10,000 times across both fitted seeds. Its interval describes document uncertainty conditional on those fits. [Paired reduction](../../paper/companion/intervals.py)

## SFT, preference estimation and editing

The 48/49 comparisons above concern pretraining and model conversion. The additional LLaDA-family evidence comes from smaller diagnostic screens with different parents, supervision and evaluation panels.

| Diagnostic | What was measured | Limit of the evidence |
| --- | --- | --- |
| LLaDA 1.5 | A frozen seed-42 F2 endpoint and its midpoint reference scored 32 fixed chosen/corrupted pairs. At a 32-draw budget, mean scalar preference-loss variance was 22.0433 for nested sampling, 9.1297 for time-diverse sampling and 0.0390 when policy and reference shared noise for each response. | The experiment changed no weights. Its derivative probe covered one output-head coordinate. Preference-trained quality, full-gradient variance, SFT performance and training speed require other evidence. |
| LLaDA 2.1 | A three-round positional editor filled MASKs and replaced visible ordinary tokens. Its E0 comparison redrafted from all MASKs. The seed-42 editing continuation added 2 TPP to a B2 parent. | E0 received different information about the draft. Grounded-format repairs transferred weakly to ordinary text. The initial local screen recorded 0/64 exact outputs and 0/16 clean drafts preserved for E1. These counts measure exact repair on a fixed canvas with a synthetic EOS. |
| LLaDA 2.2 | E3 predicted keep/substitute/delete/insert actions using LCS labels and block-local splices. E2 used the same extended action vocabulary with positional repair. Both scored 0/64 exact and preserved 0/16 clean drafts in the initial screen. | The released decoder and paper specify some mechanics. The SFT noise mixture, round count, wildcard/tie handling, tail visibility, and loss normalization required reconstruction choices. Three rounds, rates in [0.3, 0.8], a 50% random-replacement split and hidden post-EOS padding describe this screen. Decoder parity alone cannot establish upstream SFT fidelity or general structural repair. |

The companion's `component-discovery-1.estimator` and `component-discovery-1.repair` records contain the measurements behind these diagnostic counts. The `estimator-screen` and `repair-screen` reductions retain the original sample sizes and controls. The initial exact-match screen and later grounded-format observations test different tasks. [Historical records](../../paper/companion/data/historical.json) · [Diagnostic tables](../../paper/companion/historical.py)

## Local verification

```sh
python -m pytest -q tests/test_paper_components*.py
python -m paper.companion --check
```

The CPU tests use a byte tokenizer, generated local Parquet, a small transformer and shortened schedules. They exercise both trainers, calibration, the shared tail, the shifted projection, export and scoring of all endpoints, and the six paired reductions against the compact evidence. Reproducing the training trajectories and paper scores requires the full local inputs and scientific budgets.
