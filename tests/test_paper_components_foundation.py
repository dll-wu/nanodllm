from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest
import torch
from nanodllm.paper_components_foundation import (
    backward_microbatch,
    foundation_loss,
    initialize_policy,
    prepare_update,
    subs_cross_entropy,
)
from nanodllm.paper_components_noise import loss_variance_score, mask_rates
from nanodllm.paper_runtime.data import DLMDataBatch
from nanodllm.paper_runtime.training_bd3lm import bd3lm_attention_mask


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            causal=False, time_conditioning=False, output_factorization="flat", mask_token_id=6
        )
        self.logits = torch.nn.Parameter(torch.arange(7, dtype=torch.float32) / 10)
        self.calls = []

    def forward(self, tokens, *, logits_indices, attention_mask, position_ids):
        self.calls.append((tokens.detach().clone(), attention_mask.detach().clone(), position_ids))
        return self.logits.expand(logits_indices.numel(), -1)


def batch(rows=2, length=8, segmented=False):
    tokens = torch.arange(rows * length).reshape(rows, length).remainder(6)
    loss_mask = torch.ones_like(tokens, dtype=torch.bool)
    loss_mask[:, 0] = False
    positions = torch.arange(length).expand(rows, -1).clone()
    boundary = None
    if segmented:
        positions[:, length // 2 :] -= length // 2
        segments = torch.arange(length) // (length // 2)
        boundary = (segments[:, None] == segments[None, :]).expand(rows, -1, -1).clone()
    return DLMDataBatch(
        tokens, loss_mask, torch.ones_like(loss_mask), position_ids=positions, document_boundary_mask=boundary
    )


def rng(seed=7):
    return torch.Generator().manual_seed(seed)


def test_subs_normalizes_over_nonmask_vocabulary_and_blocks_mask_gradient():
    logits = torch.tensor([[1.0, 2.0, 10.0]], requires_grad=True)
    targets = torch.tensor([1])
    subs = subs_cross_entropy(logits, targets, mask_token_id=2, subs=True)
    llada = subs_cross_entropy(logits, targets, mask_token_id=2, subs=False)
    assert subs.item() == pytest.approx(torch.nn.functional.cross_entropy(logits[:, :2], targets).item())
    assert llada.item() > subs.item()
    subs.sum().backward()
    assert logits.grad[0, 2] == 0
    assert logits.grad[0, :2].abs().sum() > 0
    with pytest.raises(ValueError, match="clean MASK"):
        subs_cross_entropy(logits, torch.tensor([2]), mask_token_id=2, subs=True)


def test_stratified_rates_and_open_zero_endpoint():
    values = mask_rates(200, low=0, high=1, stratified=True, device="cpu", generator=rng())
    assert (values > 0).all()
    assert torch.equal((values * 200).long(), torch.arange(200))
    clipped = mask_rates(200, low=0.3, high=0.8, stratified=True, device="cpu", generator=rng())
    torch.testing.assert_close(clipped, 0.3 + 0.5 * values)
    with pytest.raises(ValueError):
        mask_rates(1, low=0, high=0, stratified=True, device="cpu", generator=rng())


def test_mdlm_loss_is_inverse_time_sum_over_all_eligible_not_active_tokens():
    data = batch()
    model = TinyModel()
    model.logits.data.zero_()
    reference_rng = rng()
    times = mask_rates(2, low=0, high=1, stratified=True, device="cpu", generator=reference_rng)[:, None]
    selected = (torch.rand(data.x_0.shape, generator=reference_rng) < times) & data.loss_mask
    expected = (selected / times).sum() * math.log(6) / data.loss_mask.sum()
    result = foundation_loss(model, data, arm="F1", generator=rng(), policy={})
    torch.testing.assert_close(result.loss, expected)
    assert result.telemetry["source_positions"] == 16
    assert result.telemetry["active_tokens"] == int(selected.sum())
    assert not (result.token_losses[~selected] != 0).any()
    assert torch.equal(model.calls[-1][0][:, 0], data.x_0[:, 0])


def test_llada_variable_length_hides_removed_tokens_without_new_targets():
    data = batch(rows=32)
    model = TinyModel()
    result = foundation_loss(model, data, arm="F2", generator=rng(), policy={"variable_length_probability": 1})
    tokens, visibility, _ = model.calls[-1]
    valid = visibility.diagonal(dim1=1, dim2=2)
    crop_length = tokens.shape[1]
    assert result.telemetry["variable_length_rows"] == 32
    assert result.telemetry["source_positions"] == data.x_0.numel()
    assert result.telemetry["valid_tokens"] < data.x_0.numel()
    assert (valid.sum(dim=1) >= 1).all()
    assert not (visibility & ~valid[:, None, :]).any()
    assert not (visibility & ~valid[:, :, None]).any()
    assert crop_length < data.x_0.shape[1]
    assert valid.all()
    assert result.telemetry["processed_positions"] == 32 * crop_length
    assert result.telemetry["lossable_tokens"] == int(data.loss_mask[:, :crop_length].sum())
    assert result.telemetry["noise_low"] == 0.001
    result.loss.backward()
    assert model.logits.grad[6] > 0  # LLaDA uses the ordinary full softmax.
    assert (data.attention_mask).all()  # No mutation of caller-owned batches.


def test_native_bd3_topology_is_block_causal_clean_and_isolates_documents():
    data = batch(segmented=True)
    model = TinyModel()
    result = foundation_loss(model, data, arm="F3", generator=rng(), policy={"block_size": 2})
    tokens, mask, positions = model.calls[-1]
    expected = bd3lm_attention_mask(data.document_boundary_mask, 2, position_ids=data.position_ids)
    assert torch.equal(mask, expected)
    assert tokens.shape == (2, 16)
    assert torch.equal(positions[:, :8], positions[:, 8:])
    assert mask[0, 8, 9]  # Same-block future clean key, unlike causal-clean conversion.
    assert not mask[0, 0, 8:10].any()  # No same-block gold leakage to noisy queries.
    assert not mask[0, :4, 12:].any()  # No other-document clean visibility.
    assert result.telemetry["processed_positions"] == 32
    reach = mask[0].float()
    for _ in range(3):
        reach = ((reach @ reach) > 0).float()
    assert not reach[:2, 8:].any()  # No indirect same/future-block gold path either.


def test_f3_f4_initial_losses_are_identical_and_f4_rejects_skipped_tuning():
    model, data = TinyModel(), batch()
    policy = {"block_size": 2, "noise_update_interval": 2}
    initialize_policy(policy)
    f3 = foundation_loss(model, data, arm="F3", generator=rng(), policy=policy)
    f4 = foundation_loss(model, data, arm="F4", generator=rng(), policy=policy)
    torch.testing.assert_close(f3.loss, f4.loss, rtol=0, atol=0)
    with pytest.raises(ValueError, match="selection is due"):
        backward_microbatch(model, data, arm="F4", step=2, num_steps=5, generator=rng(), policy=policy)


def test_variance_score_is_mean_within_batch_block_variance_not_batchmean_variance():
    observations = [torch.tensor([0.0, 2.0]), torch.tensor([10.0, 12.0])]
    assert loss_variance_score(observations) == 2.0
    with pytest.raises(ValueError, match="two independent"):
        loss_variance_score(observations[:1])


def test_noise_tuning_updates_json_state_without_gradient_or_mode_changes():
    model = TinyModel()
    model.logits.grad = torch.ones_like(model.logits)
    weights = model.logits.detach().clone()
    tuning_rng = rng()
    before_rng = tuning_rng.get_state().clone()
    policy = {"block_size": 2, "noise_update_interval": 2}
    args = dict(model=model, num_steps=5, generator=tuning_rng, policy=policy, calibration_batches=[batch(), batch()])
    report = prepare_update(step=0, **args)
    assert report["calibration_forwards"] == 0
    assert torch.equal(before_rng, tuning_rng.get_state())
    report = prepare_update(step=2, **args)
    assert report["calibration_forwards"] == 8
    assert report["calibration_processed_positions"] == 8 * 32
    assert model.training
    assert torch.equal(weights, model.logits)
    assert torch.equal(model.logits.grad, torch.ones_like(model.logits))
    assert policy["method_state"]["tuning_count"] == 1
    assert json.loads(json.dumps(policy["method_state"])) == policy["method_state"]
    assert all(math.isfinite(value) for value in report.values())
