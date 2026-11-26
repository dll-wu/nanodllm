import hashlib
import json
import math
from fractions import Fraction
from itertools import combinations
from math import factorial

import pytest
import torch
from nanodllm.mdlm_objective import (
    expand_mdlm_time,
    schedule_weighted_mdlm_objective,
    schedule_weighted_mdlm_terms,
)


def _nonempty_mask_sets(num_tokens: int):
    positions = range(num_tokens)
    for size in range(1, num_tokens + 1):
        yield from combinations(positions, size)


def _shared_time_mask_set_coefficient(masked_count: int, num_tokens: int) -> Fraction:
    # Integral_0^1 t^(k-1) (1-t)^(n-k) dt = Beta(k, n-k+1).
    return Fraction(factorial(masked_count - 1) * factorial(num_tokens - masked_count), factorial(num_tokens))


def _shared_time_expectation(loss_by_mask: dict[tuple[int, ...], tuple[Fraction, ...]]) -> Fraction:
    num_tokens = len(next(iter(loss_by_mask.values())))
    total = Fraction(0)
    for masked in _nonempty_mask_sets(num_tokens):
        coefficient = _shared_time_mask_set_coefficient(len(masked), num_tokens)
        total += coefficient * sum(loss_by_mask[masked][position] for position in masked)
    return total / num_tokens


def _shared_time_midpoint_grid_expectation(
    loss_by_mask: dict[tuple[int, ...], tuple[Fraction, ...]],
    midpoint_count: int,
    *,
    schedule_weighted: bool = True,
) -> Fraction:
    num_tokens = len(next(iter(loss_by_mask.values())))
    total = Fraction(0)
    for midpoint_index in range(midpoint_count):
        t = Fraction(2 * midpoint_index + 1, 2 * midpoint_count)
        for masked in _nonempty_mask_sets(num_tokens):
            probability = t ** len(masked) * (1 - t) ** (num_tokens - len(masked))
            coefficient = probability / t if schedule_weighted else probability
            total += coefficient * sum(loss_by_mask[masked][position] for position in masked)
    return total / (midpoint_count * num_tokens)


def _independent_token_time_expectation(loss_by_mask: dict[tuple[int, ...], tuple[Fraction, ...]]) -> Fraction:
    num_tokens = len(next(iter(loss_by_mask.values())))
    # For a target i in S, integrating M_i/t_i gives 1; every other token is
    # masked or unmasked with marginal integral 1/2.
    coefficient = Fraction(1, 2 ** (num_tokens - 1))
    total = Fraction(0)
    for masked in _nonempty_mask_sets(num_tokens):
        total += coefficient * sum(loss_by_mask[masked][position] for position in masked)
    return total / num_tokens


def test_expand_mdlm_time_handles_scalar_per_sequence_and_per_token_shapes():
    reference = torch.zeros(2, 3, dtype=torch.float64)

    assert torch.equal(expand_mdlm_time(0.25, reference), torch.full_like(reference, 0.25))
    assert torch.equal(
        expand_mdlm_time(torch.tensor([0.25, 0.75]), reference),
        torch.tensor([[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]], dtype=torch.float64),
    )
    per_token = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float64)
    assert torch.equal(expand_mdlm_time(per_token, reference), per_token)

    with pytest.raises(ValueError, match="per-sequence"):
        expand_mdlm_time(torch.tensor([0.1, 0.2, 0.3]), reference)
    with pytest.raises(ValueError, match="per-token"):
        expand_mdlm_time(torch.ones(2, 1), reference)


@pytest.mark.parametrize(
    ("t", "expected"),
    [
        (0.5, [[2.0, 0.0, 6.0], [8.0, 10.0, 0.0]]),
        (torch.tensor([0.5, 0.25]), [[2.0, 0.0, 6.0], [16.0, 20.0, 0.0]]),
        (
            torch.tensor([[0.5, 0.25, 1.0], [0.25, 0.5, 0.75]]),
            [[2.0, 0.0, 3.0], [16.0, 10.0, 0.0]],
        ),
    ],
)
def test_schedule_weighted_terms_and_lossable_token_normalization(t, expected):
    token_nll = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    corruption_weight = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    lossable_mask = torch.tensor([[True, True, True], [True, True, False]])

    terms = schedule_weighted_mdlm_terms(
        token_nll,
        corruption_weight,
        t,
        lossable_mask=lossable_mask,
    )
    expected_terms = torch.tensor(expected)
    assert torch.equal(terms, expected_terms)
    assert schedule_weighted_mdlm_objective(
        token_nll,
        corruption_weight,
        t,
        lossable_mask=lossable_mask,
    ) == pytest.approx(expected_terms.sum().item() / 5)


def test_schedule_weighted_objective_rejects_nonpositive_active_time_without_clipping():
    nll = torch.ones(1, 2)
    corruption_weight = torch.tensor([[1.0, 0.0]])

    with pytest.raises(ValueError, match="strictly positive"):
        schedule_weighted_mdlm_terms(nll, corruption_weight, torch.tensor([[0.0, 0.0]]))

    # An inactive zero does not create a 0/0 term.
    terms = schedule_weighted_mdlm_terms(nll, corruption_weight, torch.tensor([[0.5, 0.0]]))
    assert torch.equal(terms, torch.tensor([[2.0, 0.0]]))


def test_shared_scalar_time_has_beta_mask_set_coefficients_for_three_tokens():
    coefficients = {masked: _shared_time_mask_set_coefficient(len(masked), 3) for masked in _nonempty_mask_sets(3)}

    assert {coefficients[masked] for masked in combinations(range(3), 1)} == {Fraction(1, 3)}
    assert {coefficients[masked] for masked in combinations(range(3), 2)} == {Fraction(1, 6)}
    assert coefficients[(0, 1, 2)] == Fraction(1, 3)


def test_constant_context_loss_has_same_scalar_and_independent_time_expectation():
    # If every target has the same CE regardless of which other tokens are
    # masked, both estimators reduce to that CE after per-token normalization.
    loss_by_mask = {masked: (Fraction(2),) * 3 for masked in _nonempty_mask_sets(3)}

    assert _shared_time_expectation(loss_by_mask) == Fraction(2)
    assert _independent_token_time_expectation(loss_by_mask) == Fraction(2)


def test_context_dependent_loss_makes_per_token_time_a_different_objective():
    # A model can be better when exactly one token is masked because it sees
    # more context. Shared scalar time gives singleton mask sets coefficient
    # 1/3, whereas independent token times gives every target/mask-set pair
    # coefficient 1/4. The expectations therefore differ exactly.
    loss_by_mask = {
        masked: (Fraction(1) if len(masked) == 1 else Fraction(2),) * 3 for masked in _nonempty_mask_sets(3)
    }

    shared_time = _shared_time_expectation(loss_by_mask)
    independent_token_time = _independent_token_time_expectation(loss_by_mask)

    assert shared_time == Fraction(5, 3)
    assert independent_token_time == Fraction(7, 4)
    assert shared_time != independent_token_time


def test_evaluate_nelbo_matches_exhaustive_mask_expectation_on_fixed_midpoint_grid(monkeypatch):
    from nanodllm.data import DLMDataBatch
    from nanodllm.eval import evaluate_nelbo
    from nanodllm.noise import MDLMNoise

    num_tokens = 3
    vocab_size = 4
    mask_token_id = 3
    midpoint_times = (Fraction(1, 4), Fraction(3, 4))
    loss_by_mask = {
        masked: (Fraction({1: 1, 2: 2, 3: 4}[len(masked)]),) * num_tokens for masked in _nonempty_mask_sets(num_tokens)
    }

    time_rows: list[float] = []
    active_mask_rows: list[torch.Tensor] = []
    uniform_rows: list[torch.Tensor] = []
    all_mask_sets = [(), *_nonempty_mask_sets(num_tokens)]
    for t in midpoint_times:
        rows_at_t = 0
        for masked in all_mask_sets:
            multiplicity = t.numerator ** len(masked) * (t.denominator - t.numerator) ** (num_tokens - len(masked))
            active_mask = torch.zeros(num_tokens, dtype=torch.bool)
            if masked:
                active_mask[list(masked)] = True
            uniforms = torch.full((num_tokens,), float((1 + t) / 2), dtype=torch.float32)
            uniforms[active_mask] = float(t / 2)
            for _ in range(multiplicity):
                time_rows.append(float(t))
                active_mask_rows.append(active_mask.clone())
                uniform_rows.append(uniforms.clone())
            rows_at_t += multiplicity
        assert rows_at_t == t.denominator**num_tokens

    exhaustive_times = torch.tensor(time_rows, dtype=torch.float32)
    exhaustive_active_masks = torch.stack(active_mask_rows)
    exhaustive_uniforms = torch.stack(uniform_rows)
    batch_size = len(time_rows)
    assert batch_size == 128

    clean_row = torch.arange(num_tokens, dtype=torch.long)
    x_0 = clean_row.repeat(batch_size, 1)
    attention_mask = torch.ones_like(x_0, dtype=torch.bool)
    batch = DLMDataBatch(x_0=x_0, loss_mask=attention_mask, attention_mask=attention_mask)

    def fake_loader(*_args, **_kwargs):
        while True:
            yield batch

    class _Tokenizer:
        def get_vocab_size(self):
            return vocab_size

    class _MaskCountToyModel(torch.nn.Module):
        def forward(self, z_t, attention_mask=None):
            del attention_mask
            masked_count = (z_t == mask_token_id).sum(dim=1)
            nll_by_masked_count = torch.tensor([1.0, 1.0, 2.0, 4.0], dtype=torch.float64)
            correct_nll = nll_by_masked_count[masked_count]
            correct_probability = torch.exp(-correct_nll)
            other_probability = (1.0 - correct_probability) / (vocab_size - 1)
            logits = other_probability.log()[:, None, None].expand(-1, num_tokens, vocab_size).clone()
            target_ids = clean_row.view(1, num_tokens, 1).expand(z_t.shape[0], -1, -1)
            logits.scatter_(
                2,
                target_ids,
                correct_probability.log()[:, None, None].expand(-1, num_tokens, 1),
            )
            return logits

    noise = MDLMNoise(mask_token_id)

    def noise_fn(x_eval, loss_mask):
        z_t, corruption_weight = noise.corrupt(
            x_eval,
            exhaustive_times,
            maskable=loss_mask,
            uniforms=exhaustive_uniforms,
        )
        assert torch.equal(corruption_weight.bool(), exhaustive_active_masks)
        return z_t, exhaustive_times, corruption_weight, None, {"corruption_uniforms": exhaustive_uniforms}

    monkeypatch.setattr("nanodllm.runtime.data.make_dataloader", fake_loader)
    monkeypatch.setattr(
        "nanodllm.runtime.data.build_token_bytes",
        lambda _tokenizer: torch.tensor([1, 1, 1, 0], dtype=torch.int32),
    )

    nelbo, masked_ce, weight_mean, schedule_weighted, diagnostics = evaluate_nelbo(
        _MaskCountToyModel(),
        None,
        _Tokenizer(),
        noise_fn,
        batch_size=batch_size,
        device=torch.device("cpu"),
        seq_len=num_tokens,
        eval_tokens=batch_size * num_tokens,
        schedule_weight=True,
        schedule_weight_scale=1.0,
        return_diagnostics=True,
    )

    finite_grid_nats = _shared_time_midpoint_grid_expectation(loss_by_mask, midpoint_count=2)
    finite_grid_masked_nats = _shared_time_midpoint_grid_expectation(
        loss_by_mask,
        midpoint_count=2,
        schedule_weighted=False,
    )
    exact_integral_nats = _shared_time_expectation(loss_by_mask)

    assert finite_grid_nats == Fraction(37, 16)
    assert finite_grid_masked_nats == Fraction(43, 32)
    assert exact_integral_nats == Fraction(7, 3)
    assert exact_integral_nats - finite_grid_nats == Fraction(1, 48)
    assert nelbo * math.log(2) == pytest.approx(float(finite_grid_nats), abs=1e-6)
    assert masked_ce * math.log(2) == pytest.approx(float(finite_grid_masked_nats), abs=1e-6)
    assert weight_mean == pytest.approx(2.0, abs=1e-6)
    assert schedule_weighted is True
    assert diagnostics["sequence_count"] == 128
    assert diagnostics["total_bytes"] == 384
    assert diagnostics["masked_target_count"] == 192
    assert diagnostics["schedule_weight_count"] == 192
    assert all(record["schema_version"] == 2 for record in diagnostics["sequence_records"])
    for root_name in (
        "input_identity_root_sha256",
        "corruption_uniform_root_sha256",
        "corruption_mask_root_sha256",
        "evaluation_sample_root_sha256",
    ):
        assert len(diagnostics[root_name]) == 64


def test_exhaustive_midpoint_oracle_converges_to_beta_integral_without_claiming_finite_grid_exactness():
    num_tokens = 3
    loss_by_mask = {
        masked: (Fraction({1: 1, 2: 2, 3: 4}[len(masked)]),) * num_tokens for masked in _nonempty_mask_sets(num_tokens)
    }

    exact_integral = _shared_time_expectation(loss_by_mask)
    midpoint_values = [
        _shared_time_midpoint_grid_expectation(loss_by_mask, midpoint_count) for midpoint_count in (2, 4, 8)
    ]
    midpoint_errors = [exact_integral - value for value in midpoint_values]

    assert exact_integral == Fraction(7, 3)
    assert midpoint_values == [Fraction(37, 16), Fraction(149, 64), Fraction(597, 256)]
    assert midpoint_errors == [Fraction(1, 48), Fraction(1, 192), Fraction(1, 768)]
    assert midpoint_errors[0] == 4 * midpoint_errors[1] == 16 * midpoint_errors[2]


def test_evaluate_nelbo_expands_per_sequence_time_and_reports_sequence_jackknife_se(monkeypatch):
    from nanodllm.eval import evaluate_nelbo

    vocab_size = 4
    sequence = torch.tensor([[1, 2, 3], [2, 3, 1]])
    attention_mask = torch.ones_like(sequence, dtype=torch.bool)
    corruption_weight = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    class _Batch:
        pass

    batch = _Batch()
    batch.x_0 = sequence
    batch.loss_mask = attention_mask
    batch.attention_mask = attention_mask

    def fake_loader(*_args, **_kwargs):
        while True:
            yield batch

    class _Tokenizer:
        def get_vocab_size(self):
            return vocab_size

    times = iter((torch.tensor([0.25, 0.5]), torch.tensor([0.5, 0.5])))
    uniforms = torch.tensor([[0.1, 0.6, 0.7], [0.2, 0.8, 0.9]])

    def noise_fn(x_0, _loss_mask):
        return x_0.clone(), next(times), corruption_weight, None, {"corruption_uniforms": uniforms}

    logits = torch.zeros(2, 3, vocab_size)

    def model_logits_fn(_model, _z_t, _x_0, _attention_mask, **_kwargs):
        return logits

    monkeypatch.setattr("nanodllm.runtime.data.make_dataloader", fake_loader)
    monkeypatch.setattr("nanodllm.runtime.data.build_token_bytes", lambda _tokenizer: torch.ones(vocab_size))

    nelbo, masked_ce, weight_mean, schedule_weighted, diagnostics = evaluate_nelbo(
        object(),
        None,
        _Tokenizer(),
        noise_fn,
        batch_size=2,
        device=torch.device("cpu"),
        seq_len=3,
        eval_tokens=12,
        model_logits_fn=model_logits_fn,
        schedule_weight=True,
        return_diagnostics=True,
    )

    # Uniform logits have CE log(4) = 2 bits. The two batches' weighted
    # numerators are respectively 6*CE and 4*CE over six bytes per batch.
    assert nelbo == pytest.approx(Fraction(5, 3))
    assert masked_ce == pytest.approx(Fraction(2, 3))
    assert weight_mean == pytest.approx(2.5)
    assert schedule_weighted is True
    assert diagnostics["nelbo_bpb_se"] == pytest.approx(1 / 3)
    assert diagnostics["masked_ce_bpb_se"] == pytest.approx(0.0, abs=1e-12)
    assert diagnostics["se_method"] == "delete_one_sequence_jackknife_ratio_conditional_on_fixed_time_grid"
    assert diagnostics["block_count"] == 4
    assert diagnostics["sequence_count"] == 4
    assert diagnostics["total_bytes"] == 12
    assert diagnostics["masked_target_count"] == 4
    assert diagnostics["schedule_weight_count"] == 4
    assert [record["sampled_t"] for record in diagnostics["sequence_records"]] == [0.25, 0.5, 0.5, 0.5]
    assert [record["byte_denominator"] for record in diagnostics["sequence_records"]] == [3.0] * 4
    assert [record["masked_target_count"] for record in diagnostics["sequence_records"]] == [1] * 4
    assert all(record["schema_version"] == 2 for record in diagnostics["sequence_records"])
    assert all(len(record["evaluation_sample_sha256"]) == 64 for record in diagnostics["sequence_records"])


def test_evaluate_nelbo_rejects_inexact_token_budget():
    from nanodllm.eval import evaluate_nelbo

    with pytest.raises(ValueError, match="positive exact multiple"):
        evaluate_nelbo(
            object(),
            None,
            object(),
            lambda *_args: (),
            batch_size=2,
            device=torch.device("cpu"),
            seq_len=3,
            eval_tokens=11,
        )


@pytest.mark.parametrize(
    ("t_min", "t_max", "expected_times", "is_full"),
    [
        (0.0, 1.0, [0.125, 0.375, 0.625, 0.875], True),
        (0.1, 0.9, [0.2, 0.4, 0.6, 0.8], False),
    ],
)
def test_run_eval_wires_shared_time_range_correction_and_bound_labels(
    monkeypatch,
    tmp_path,
    t_min,
    t_max,
    expected_times,
    is_full,
):
    from contextlib import nullcontext

    from nanodllm.config import (
        DataConfig,
        EvalConfig,
        ExperimentConfig,
        ModelConfig,
        NoiseConfig,
        RuntimeConfig,
        TrainConfig,
    )
    from nanodllm.data import ByteTokenizer, DLMDataBatch
    from nanodllm.noise import MDLMNoise
    from nanodllm.runtime.training_artifacts import _ordered_nelbo_digest_root
    from nanodllm.runtime.training_evaluation import _run_eval

    tokenizer = ByteTokenizer()
    batch = DLMDataBatch(
        x_0=torch.tensor([[65, 66, 67], [68, 69, 70]]),
        loss_mask=torch.ones(2, 3, dtype=torch.bool),
        attention_mask=torch.ones(2, 3, dtype=torch.bool),
    )

    def fake_val_loader(*_args, **_kwargs):
        while True:
            yield batch

    captured_times = []
    captured_scale = []

    def fake_evaluate_nelbo(_model, _data, _tokenizer, noise_fn, batch_size, **kwargs):
        assert batch_size == 2
        captured_scale.append(float(kwargs["schedule_weight_scale"]))
        for _ in range(2):
            _z_t, times, _weights, noise_level, noise_audit = noise_fn(batch.x_0, batch.loss_mask)
            assert times.shape == (2,)
            assert noise_level is None
            assert noise_audit["corruption_uniforms"].shape == batch.x_0.shape
            captured_times.extend(float(value) for value in times.tolist())
        row_target_counts = (1, 2, 1, 2)
        sequence_records = [
            {
                "schema_version": 2,
                "sequence_index": index,
                "sampled_t": sampled_t,
                "sampled_t_min": sampled_t,
                "sampled_t_max": sampled_t,
                "nelbo_nats_numerator": 1.25 * math.log(2) * 3,
                "masked_nats_numerator": 0.75 * math.log(2) * 3,
                "byte_denominator": 3.0,
                "lossable_content_target_count": 3,
                "masked_target_count": row_target_counts[index],
                "schedule_weight_sum": 2.0 * row_target_counts[index],
                "schedule_weight_count": row_target_counts[index],
                "input_identity_sha256": hashlib.sha256(f"input:{index}".encode()).hexdigest(),
                "corruption_uniform_sha256": hashlib.sha256(f"uniform:{index}".encode()).hexdigest(),
                "corruption_mask_sha256": hashlib.sha256(f"mask:{index}".encode()).hexdigest(),
                "evaluation_sample_sha256": hashlib.sha256(f"sample:{index}".encode()).hexdigest(),
            }
            for index, sampled_t in enumerate(expected_times)
        ]
        identity_roots = {
            "input_identity_root_sha256": _ordered_nelbo_digest_root(
                b"nanodllm-nelbo-input-identity-root-v2", sequence_records, "input_identity_sha256"
            ),
            "corruption_uniform_root_sha256": _ordered_nelbo_digest_root(
                b"nanodllm-nelbo-corruption-uniform-root-v1",
                sequence_records,
                "corruption_uniform_sha256",
            ),
            "corruption_mask_root_sha256": _ordered_nelbo_digest_root(
                b"nanodllm-nelbo-corruption-mask-root-v2", sequence_records, "corruption_mask_sha256"
            ),
            "evaluation_sample_root_sha256": _ordered_nelbo_digest_root(
                b"nanodllm-nelbo-evaluation-sample-root-v2", sequence_records, "evaluation_sample_sha256"
            ),
        }
        return (
            1.25,
            0.75,
            2.0,
            True,
            {
                "nelbo_bpb_se": 0.01,
                "masked_ce_bpb_se": 0.02,
                "se_method": "fixed-grid-test",
                "se_unit": "sequence",
                "block_count": 4,
                "sequence_count": 4,
                "total_bytes": 12,
                "masked_target_count": 6,
                "schedule_weight_count": 6,
                **identity_roots,
                "sequence_records": sequence_records,
            },
        )

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def forward(self, z_t, attention_mask=None, noise_level=None):
            del attention_mask, noise_level
            return torch.zeros(*z_t.shape, tokenizer.get_vocab_size(), device=z_t.device) + self.anchor

    config = ExperimentConfig(
        model=ModelConfig(depth=1, n_embd=8, n_head=2, max_seq_len=3),
        data=DataConfig(
            tokenizer="byte",
            train_data=["unused.parquet"],
            val_fraction=0.0,
            random_length_ratio=0.0,
            continuation_ratio=0.0,
        ),
        noise=NoiseConfig(type="mdlm", schedule_weight=True, complement_masking=False),
        runtime=RuntimeConfig(device_type="cpu", dtype="float32", compile=False),
        train=TrainConfig(
            run_dir=str(tmp_path),
            device_batch_size=2,
            total_batch_size=6,
            tensorboard=False,
        ),
        eval=EvalConfig(
            nelbo_tokens=12,
            nelbo_t_min=t_min,
            nelbo_t_max=t_max,
            nelbo_seed=17,
            fixed_ce_batches=1,
            reconstruction_eval=False,
            conditional_rank_eval=False,
            conditional_rank_rollout_eval=False,
            conditional_gen_eval=False,
            gen_eval=False,
        ),
    )
    monkeypatch.setattr("nanodllm.runtime.training_evaluation.build_val_loader", fake_val_loader)
    monkeypatch.setattr("nanodllm.runtime.training_evaluation.evaluate_nelbo", fake_evaluate_nelbo)

    model = TinyModel().train()
    metrics = _run_eval(
        model,
        config,
        tokenizer,
        MDLMNoise(tokenizer.get_mask_token_id()),
        nullcontext(),
        torch.device("cpu"),
        step=4,
        include_gen=False,
    )

    assert sorted(captured_times) == pytest.approx(expected_times)
    assert captured_scale == pytest.approx([t_max - t_min])
    assert metrics["val_nelbo_time_sampling"] == "stratified_per_sequence"
    assert metrics["val_nelbo_is_schedule_weighted_bound"] is is_full
    assert metrics["val_nelbo_is_truncated_reference_integral"] is (not is_full)
    assert metrics["val_nelbo_bpb_se"] == pytest.approx(0.01)
    assert metrics["val_nelbo_sequence_count"] == 4
    assert metrics["val_nelbo_jackknife_sequence_count"] == 4
    assert metrics["val_nelbo_jackknife_unit"] == "sequence"
    assert metrics["val_nelbo_sequence_records_count"] == 4
    assert metrics["val_nelbo_sequence_records_schema_version"] == 2
    artifact_path = tmp_path / metrics["val_nelbo_sequence_records_artifact"]
    artifact_bytes = artifact_path.read_bytes()
    assert hashlib.sha256(artifact_bytes).hexdigest() == metrics["val_nelbo_sequence_records_sha256"]
    records = [json.loads(line) for line in artifact_bytes.decode("utf-8").splitlines()]
    assert records[0]["evaluation_step"] == 4
    assert records[0]["nelbo_seed"] == 17
    assert model.training is True
