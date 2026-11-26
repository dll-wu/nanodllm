# Scientific scope

The paper follows two transfer comparisons. The six-size full-sequence AdamW study compares scratch training for 40 tokens per counted parameter with AR 20 followed by masked diffusion 20. The four-size block Muon study compares scratch 40 with AR 10 followed by block diffusion 30. Separate 20M experiments screen published training components, and a 37M architectural variant appears in the development series. Read the estimates and uncertainty in the [numerical companion](../paper/results.md).

The two transfer comparisons change several settings across studies. Their results establish within-study contrasts and do not isolate the optimizer or the effect of block attention. Each evaluation measures a particular behavior under a named decoding or masking policy. The evidence for each component is limited to the budgets, model sizes and endpoints actually tested.

The [release notes](../paper/release.md) distinguish public implementation checks from historical experimental measurements. The [checkpoint inventory](../paper/checkpoints.md) identifies the files needed for checkpoint-level evaluation.
