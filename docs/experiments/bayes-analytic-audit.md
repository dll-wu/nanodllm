# MDLM Bayes Analytic Audit

`recipes/preview/mdlm_bayes_analytic_audit.json` is a self-contained CPU audit of the
synthetic EOS-canary data, absorbing-MDLM corruption path, and the Bayes
decision under a finite empirical toy prior.

It does not instantiate a model, run optimization, evaluate a checkpoint, or
show that a trained model is calibrated.

Run:

```bash
nanodllm-pretrain --config recipes/preview/mdlm_bayes_analytic_audit.json
```

## Calculation

The clean candidate set contains eight byte-tokenized rows. Documents of
different lengths repeat twice, and EOS is appended. The prior is uniform over
these rows.

For a corrupted observation, posterior support retains clean candidates with:

- the same attention mask;
- the same loss mask; and
- identical values at every visible position.

At a masked position, the posterior token probability is the fraction of
supported candidates carrying each token. The Bayes prediction is the most
frequent supported token.

The audit evaluates deterministic corruption draws at fixed noise levels and
records all-token, EOS, and non-EOS accuracy; posterior support; EOS probability
and margin; entropy; token concentration; corruption integrity; and
independent-masking diagnostics.

## Interpretation

Because each document length is unique in the toy candidate set, the visible
attention mask identifies the underlying length. EOS can therefore be
identifiable even when content tokens are mostly masked.

This demonstrates an invariant of this finite toy distribution: its
model-visible length mask does not make EOS intrinsically ambiguous. It is not
evidence that a learned MDLM uses the information, and it does not establish
language-model or sampler quality.

## Acceptance

The authoritative success fields are:

- `strict_mdlm_all_token_bayes_analytic_status="ok"`;
- `strict_mdlm_all_token_bayes_analytic_error=""`; and
- `strict_mdlm_all_token_bayes_analytic_scope="empirical_toy_prior_data_and_corruption_only"`.

The run writes its resolved config, final metrics, and audit telemetry beneath
the configured run directory. The compatibility `val_nelbo_bpb` field is a
placeholder in this audit and must not be interpreted as a validation result.
