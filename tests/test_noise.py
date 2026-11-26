import pytest
import torch
from nanodllm.config import DataConfig, EvalConfig, ExperimentConfig, NoiseConfig, TrainConfig, validate_config
from nanodllm.noise import (
    MaskUniformNoise,
    MDLMNoise,
    UniformNoise,
    build_noise,
    sample_positive_uniform_scalar,
    sample_positive_uniform_tensor,
)


def test_full_support_training_time_sampling_rejects_only_exact_zero(monkeypatch):
    draws = iter([0.0, 0.25])
    monkeypatch.setattr("nanodllm.runtime.noise.random.uniform", lambda _low, _high: next(draws))

    assert sample_positive_uniform_scalar(0.0, 1.0) == pytest.approx(0.25)
    values = sample_positive_uniform_tensor((4096,), device="cpu", t_min=0.0, t_max=1.0)
    assert torch.all(values > 0.0)
    assert torch.all(values < 1.0)
    assert sample_positive_uniform_scalar(1.0, 1.0) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="fixed t=0"):
        sample_positive_uniform_scalar(0.0, 0.0)


def test_zero_mask_outcomes_are_not_forced_to_one_mask():
    x = torch.arange(8).view(2, 4)
    uniforms = torch.full(x.shape, 0.5)

    z, w = MDLMNoise(mask_token_id=99).corrupt(x, torch.tensor([0.1, 0.9]), uniforms=uniforms)

    assert torch.equal(z[0], x[0])
    assert not w[0].bool().any()
    assert torch.equal(z[1], torch.full_like(z[1], 99))
    assert w[1].bool().all()


def test_schedule_weighted_mdlm_rejects_a_fixed_zero_time():
    config = ExperimentConfig(
        data=DataConfig(tokenizer="byte", train_data=["dummy.parquet"]),
        noise=NoiseConfig(type="mdlm", t_min=0.0, t_max=0.0, schedule_weight=True),
    )

    with pytest.raises(ValueError, match="t_max > 0"):
        validate_config(config)


def test_mdlm_noise_respects_maskable():
    x = torch.arange(10).view(1, 10)
    maskable = torch.zeros_like(x, dtype=torch.bool)
    maskable[:, 5:] = True
    z, w = MDLMNoise(mask_token_id=99).corrupt(x, 1.0, maskable=maskable)
    assert torch.equal(z[:, :5], x[:, :5])
    assert torch.equal(z[:, 5:], torch.full_like(z[:, 5:], 99))
    assert torch.equal(w.bool(), maskable)


def test_mdlm_noise_accepts_per_row_t():
    x = torch.arange(8).view(2, 4)
    z, w = MDLMNoise(mask_token_id=99).corrupt(x, torch.tensor([0.0, 1.0]))

    assert torch.equal(z[0], x[0])
    assert torch.equal(z[1], torch.full_like(z[1], 99))
    assert not w[0].bool().any()
    assert w[1].bool().all()


def test_mdlm_noise_accepts_explicit_uniforms_and_validates_them():
    x = torch.arange(8).view(2, 4)
    maskable = torch.tensor([[True, True, False, True], [True, True, True, True]])
    uniforms = torch.tensor([[0.1, 0.3, 0.0, 0.7], [0.6, 0.2, 0.8, 0.4]])

    z, w = MDLMNoise(mask_token_id=99).corrupt(
        x,
        torch.tensor([0.25, 0.5]),
        maskable=maskable,
        uniforms=uniforms,
    )

    expected_mask = torch.tensor([[True, False, False, False], [False, True, False, True]])
    assert torch.equal(w.bool(), expected_mask)
    assert torch.equal(z, x.masked_fill(expected_mask, 99))
    with pytest.raises(ValueError, match="match the token tensor shape"):
        MDLMNoise(mask_token_id=99).corrupt(x, 0.5, uniforms=torch.zeros(1, 4))
    with pytest.raises(TypeError, match="floating point"):
        MDLMNoise(mask_token_id=99).corrupt(x, 0.5, uniforms=torch.zeros_like(x))


def test_mdlm_per_token_t_respects_maskable_and_returns_token_levels():
    x = torch.arange(10).view(1, 10)
    maskable = torch.zeros_like(x, dtype=torch.bool)
    maskable[:, 5:] = True
    z, w, t_values = MDLMNoise(mask_token_id=99).corrupt_per_token_t(
        x,
        1.0,
        1.0,
        maskable=maskable,
    )

    assert t_values.shape == x.shape
    assert torch.equal(t_values, torch.ones_like(t_values))
    assert torch.equal(z[:, :5], x[:, :5])
    assert torch.equal(z[:, 5:], torch.full_like(z[:, 5:], 99))
    assert torch.equal(w.bool(), maskable)


def test_complement_noise_respects_maskable():
    x = torch.arange(20).view(2, 10)
    maskable = torch.zeros_like(x, dtype=torch.bool)
    maskable[:, 5:] = True
    z, w = MDLMNoise(mask_token_id=99).corrupt_with_complement(x, 0.5, maskable=maskable)
    assert torch.equal(z[:, :5], x[:, :5])
    assert not w[:, :5].bool().any()
    assert torch.equal(w.bool() & ~maskable, torch.zeros_like(maskable))


def test_uniform_noise_respects_maskable():
    x = torch.arange(10).view(1, 10)
    maskable = torch.zeros_like(x, dtype=torch.bool)
    maskable[:, 5:] = True
    z, w = UniformNoise(mask_token_id=99, vocab_size=128).corrupt(x, 1.0, maskable=maskable)
    assert torch.equal(z[:, :5], x[:, :5])
    assert not torch.equal(z[:, 5:], x[:, 5:])
    assert torch.equal(w.bool(), maskable)


def test_mask_uniform_noise_respects_maskable_and_mix_extremes():
    x = torch.arange(10).view(1, 10)
    maskable = torch.zeros_like(x, dtype=torch.bool)
    maskable[:, 5:] = True

    z_mask, w_mask = MaskUniformNoise(mask_token_id=99, vocab_size=128, uniform_mix_prob=0.0).corrupt(
        x,
        1.0,
        maskable=maskable,
    )
    assert torch.equal(z_mask[:, :5], x[:, :5])
    assert torch.equal(z_mask[:, 5:], torch.full_like(z_mask[:, 5:], 99))
    assert torch.equal(w_mask.bool(), maskable)

    z_uniform, w_uniform = MaskUniformNoise(mask_token_id=99, vocab_size=128, uniform_mix_prob=1.0).corrupt(
        x,
        1.0,
        maskable=maskable,
    )
    assert torch.equal(z_uniform[:, :5], x[:, :5])
    assert not (z_uniform[:, 5:] == 99).any()
    assert torch.equal(w_uniform.bool(), maskable)


def test_mdlm_suffix_noise_respects_maskable():
    x = torch.arange(20).view(2, 10)
    maskable = torch.zeros_like(x, dtype=torch.bool)
    maskable[0, 3:] = True
    maskable[1, 5:] = True
    z, w = MDLMNoise(mask_token_id=99).corrupt_suffix(x, 0.5, maskable=maskable)
    # Row 0: maskable positions 3..9 (7 tokens), round(3.5)=4 masked, cutoff at position 6
    assert torch.equal(z[0, :6], x[0, :6])
    assert torch.equal(z[0, 6:], torch.full_like(z[0, 6:], 99))
    # Row 1: maskable positions 5..9 (5 tokens), round(2.5)=2 masked, cutoff at position 8
    assert torch.equal(z[1, :8], x[1, :8])
    assert torch.equal(z[1, 8:], torch.full_like(z[1, 8:], 99))
    # Weights should only be on maskable positions in the suffix
    assert torch.equal(w.bool(), maskable & (z == 99))


def test_mdlm_suffix_noise_accepts_per_row_t():
    x = torch.arange(20).view(2, 10)
    maskable = torch.ones_like(x, dtype=torch.bool)
    z, w = MDLMNoise(mask_token_id=99).corrupt_suffix(x, torch.tensor([0.2, 0.8]), maskable=maskable)

    assert torch.equal(z[0, :8], x[0, :8])
    assert torch.equal(z[0, 8:], torch.full_like(z[0, 8:], 99))
    assert torch.equal(z[1, :2], x[1, :2])
    assert torch.equal(z[1, 2:], torch.full_like(z[1, 2:], 99))
    assert int(w[0].sum().item()) == 2
    assert int(w[1].sum().item()) == 8


def test_build_noise_supports_mask_uniform():
    noise = build_noise(NoiseConfig(type="mask_uniform", mask_uniform_mix_prob=0.3), mask_token_id=4, vocab_size=64)

    assert isinstance(noise, MaskUniformNoise)
    assert noise.uniform_mix_prob == 0.3


def test_mdlm_rejects_time_conditioning():
    cfg = ExperimentConfig(
        data=DataConfig(tokenizer="byte", train_data=["dummy.parquet"]),
        noise=NoiseConfig(type="mdlm", time_conditioning=True),
    )

    with pytest.raises(ValueError, match="only allowed for non-MDLM"):
        validate_config(cfg)


def test_per_token_noise_requires_strict_mdlm():
    cfg = ExperimentConfig(
        data=DataConfig(tokenizer="byte", train_data=["dummy.parquet"]),
        noise=NoiseConfig(type="mdlm", per_token_noise_levels=True, complement_masking=True),
    )

    with pytest.raises(ValueError, match="complement_masking=false"):
        validate_config(cfg)

    cfg = ExperimentConfig(
        data=DataConfig(tokenizer="byte", train_data=["dummy.parquet"]),
        noise=NoiseConfig(type="uniform", per_token_noise_levels=True, complement_masking=False),
    )

    with pytest.raises(ValueError, match="noise.type='mdlm'"):
        validate_config(cfg)

    cfg = ExperimentConfig(
        data=DataConfig(tokenizer="byte", train_data=["dummy.parquet"]),
        noise=NoiseConfig(type="mdlm", per_token_noise_levels=True, complement_masking=False, schedule_weight=False),
    )

    with pytest.raises(ValueError, match="schedule_weight=true"):
        validate_config(cfg)


def test_soft_input_curriculum_requires_time_conditioned_uniform():
    cfg = ExperimentConfig(
        data=DataConfig(tokenizer="byte", train_data=["dummy.parquet"]),
        noise=NoiseConfig(type="mdlm", soft_input_curriculum=True),
    )

    with pytest.raises(ValueError, match="noise.type='uniform'"):
        validate_config(cfg)

    cfg = ExperimentConfig(
        data=DataConfig(tokenizer="byte", train_data=["dummy.parquet"]),
        noise=NoiseConfig(type="uniform", time_conditioning=False, soft_input_curriculum=True),
    )

    with pytest.raises(ValueError, match="time_conditioning=true"):
        validate_config(cfg)


def test_prior_correction_is_eval_only():
    cfg = ExperimentConfig(
        data=DataConfig(tokenizer="byte", train_data=["dummy.parquet"]),
        eval=EvalConfig(gen_prior_correction_strength=0.25, gen_prior_correction_batches=1),
    )

    with pytest.raises(ValueError, match="only supported for train.eval_only"):
        validate_config(cfg)


def test_prior_correction_requires_frequency_batches():
    cfg = ExperimentConfig(
        data=DataConfig(tokenizer="byte", train_data=["dummy.parquet"]),
        train=TrainConfig(eval_only=True, resume_from="model.pt"),
        eval=EvalConfig(gen_prior_correction_strength=0.25, gen_prior_correction_batches=0),
    )

    with pytest.raises(ValueError, match="must be positive when prior correction is enabled"):
        validate_config(cfg)


def test_sample_token_selection_validation():
    cfg = ExperimentConfig(
        data=DataConfig(tokenizer="byte", train_data=["dummy.parquet"]),
        eval=EvalConfig(sample_token_selection="top2"),
    )

    with pytest.raises(ValueError, match="sample_token_selection"):
        validate_config(cfg)
