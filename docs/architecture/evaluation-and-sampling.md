# Evaluation And Sampling

Evaluation is controlled by the `eval` section and scheduled by
`train.eval_every`. Keep endpoint budgets larger than periodic canaries and
record the exact plan with every comparison.

The [runtime evaluator](../../nanodllm/runtime/eval.py) exposes the scoring functions. To inspect a measurement, start with [per-sequence NELBO](../../nanodllm/runtime/eval_nelbo.py), [fixed-noise reconstruction](../../nanodllm/runtime/eval_reconstruction.py), [generated continuations](../../nanodllm/runtime/eval_conditional_generation.py), or [gold suffix ranks](../../nanodllm/runtime/eval_rank_suffix.py).

## MDLM Reference Objective

For absorbing MDLM, the evaluator samples one corruption time per validation
sequence, masks tokens independently at that time, and evaluates the
schedule-weighted clean-token negative log likelihood. Deterministic permuted
midpoint strata avoid evaluating the singular endpoint at exactly zero.

The primary output is bits per UTF-8 byte:

```text
sum(weighted token NLL) / sum(content bytes) / log(2)
```

Special tokens have zero byte weight. EOS may still be a prediction target.

Key settings:

- `eval.nelbo_tokens`;
- `eval.nelbo_t_min` and `eval.nelbo_t_max`;
- `eval.nelbo_seed`; and
- `eval.nelbo_time_sampling="stratified_per_sequence"`.

A full `[0, 1]` integration range is labeled as the schedule-weighted reference
bound only when the evaluator actually observed weighted targets across that
range. A narrower range is a truncated reference integral, not the same
quantity.

`val_nelbo_bpb_se` is a sequence-level delete-one ratio-jackknife diagnostic.
It includes variation across data, corruption, and time strata; it does not
measure quadrature bias or within-stratum variance. Calibrate evaluation budget
with repeated seeds and grid-size sensitivity before treating small differences
as meaningful.

## Fixed-Noise Reconstruction

Fixed-noise CE answers a different question: how well does the model reconstruct
at selected corruption levels? It is useful for identifying whether failures
are concentrated at low, medium, or high masking.

Report both all-lossable-token and content-only values. Do not substitute
fixed-noise CE for the integrated MDLM objective.

Optional reconstruction canaries log bounded decoded examples at configured
times. These are debugging artifacts, not samples from the model.

## Generation

MDLM generation starts from MASK tokens and iteratively commits predictions.
The baseline sampler is `eval.sampler="default"`.

Important settings:

- `eval.gen_samples`, `gen_len`, and `sample_steps`;
- `eval.sample_token_selection`: `argmax` or `multinomial`;
- `eval.gen_temperature`;
- `eval.sample_scoring_mode` and `sample_jitter_scale`;
- `eval.sample_trace`; and
- `eval.gen_score_mode`.

`gen_score_mode` may be:

- `none`: surface metrics and decoded samples only;
- `self`: corrupt generated text and measure reconstruction with the same
  model; or
- `judge`: load `eval.judge_model_id` and report an auxiliary model score.

Self scores measure consistency, not external quality. Judge scores inherit the
judge model's tokenizer, domain, and calibration limitations.

## Collapse Diagnostics

Generation outputs include EOS rate, empty rate, exact duplicates, distinct
n-grams, repeated 4-grams, token concentration, token entropy, content bytes,
and blocked-token rate.

Sampler traces include candidate entropy, top-1 confidence, commits per step,
predicted top-token concentration, remasking, and remaining MASK counts.

Interpret these together with decoded samples. No single diversity or
perplexity number establishes generation quality. Deterministic argmax can
produce deceptively consistent but collapsed text; stochastic selection can
increase diversity while reducing local quality.

## Conditional Diagnostics

The config schema also exposes conditional reconstruction, prefix continuation,
rank, leave-one-out, and rollout diagnostics. These are research tools for
separating representation failure from free-running sampler failure. They may
be expensive and should be enabled with bounded sample counts first.

## Experimental Samplers

`fhs`, `remdm`, uniform, mask-uniform, and BD3LM sampling paths are available for
explicit comparisons. They do not replace the default MDLM sampler merely
because they improve one diagnostic. Hold checkpoint, seed, length, steps, token
selection, and evaluation budget fixed when comparing samplers.

## Comparison Checklist

Before declaring a winner:

1. bind both runs to the same validation file membership and tokenizer;
2. use common evaluation seeds and sequence counts;
3. verify identical byte accounting and time range;
4. report paired differences when plans are shared;
5. predeclare a meaningful margin;
6. inspect generation collapse metrics and samples; and
7. repeat borderline outcomes with fresh plans.
