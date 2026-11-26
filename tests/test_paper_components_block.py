import math

import pytest
import torch
from nanodllm.paper_components_block import (
    backward_microbatch,
    build_block_view,
    curriculum_stages,
    resolve_block_size,
)
from nanodllm.paper_components_fastv2 import (
    align_mask_padding,
    build_fastv2_view,
    forward_block_logits,
)
from nanodllm.paper_runtime.data import DLMDataBatch
from nanodllm.paper_runtime.model import DiffusionConfig, DiffusionTransformer


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def model(length=8):
    torch.manual_seed(19)
    result = DiffusionTransformer(
        DiffusionConfig(
            sequence_len=length,
            vocab_size=16,
            n_layer=2,
            n_head=2,
            n_embd=16,
            mask_token_id=15,
            residual_output_projection_init="scaled_normal",
        ),
        compute_dtype=torch.float32,
        pad_vocab_to=1,
    )
    result.init_weights()
    return result


def batch(*, segments=(8,)):
    tokens = torch.arange(1, sum(segments) + 1)[None, :]
    positions = torch.tensor([[i for n in segments for i in range(n)]])
    loss = positions != 0
    return DLMDataBatch(tokens, loss, torch.ones_like(loss), position_ids=positions)


def generator(seed=17):
    return torch.Generator().manual_seed(seed)


def logits(model, view):
    return model(
        view.tokens, logits_indices=view.logit_indices, attention_mask=view.visibility, position_ids=view.positions
    )


def test_default_curriculum_is_exhaustive_balanced_and_decreases_without_legacy_change():
    stages = curriculum_stages(3520, context_length=1024, policy={})
    assert stages[0].start == 0 and stages[-1].stop == 3520
    assert all(a.stop == b.start for a, b in zip(stages, stages[1:], strict=False))
    assert [s.block_size for s in stages] == [1, 4, 32, 64, 1024, 1024, 1024, 512, 256, 128, 64, 32]
    assert [sum(s.stop - s.start for s in stages if s.phase == phase) for phase in range(3)] == [352, 2816, 352]
    for phase in (0, 2):
        lengths = [s.stop - s.start for s in stages if s.phase == phase]
        assert max(lengths) - min(lengths) <= 1
    for s in stages:
        for step in (s.start, s.stop - 1):
            assert resolve_block_size(arm="B2", step=step, num_steps=3520, context_length=1024, policy={}) == (
                s.block_size,
                s.phase,
            )


def test_b1_b2_same_block_have_identical_loss_gradients_and_rng():
    first, second = model(), model()
    policy = {"block_size": 2, "warmup_blocks": [1, 2, 8], "decay_blocks": [8, 4, 2]}
    state = torch.random.get_rng_state().clone()
    results = [
        backward_microbatch(m, batch(), arm=arm, step=3, num_steps=60, generator=generator(), policy=policy)
        for m, arm in ((first, "B1"), (second, "B2"))
    ]
    assert results[0] == results[1]
    assert torch.equal(state, torch.random.get_rng_state())
    for p, q in zip(first.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)


def test_full_context_phase_has_global_noisy_visibility_and_no_clean_half():
    source = batch()
    active = source.loss_mask
    noisy = source.x_0.masked_fill(active, 15)
    view = build_block_view(source, noisy, active, torch.ones_like(noisy).float(), block_size=8)
    assert view.tokens.shape == (1, 8)
    assert view.visibility.all()
    m = model()
    expected = m(noisy, attention_mask=source.attention_mask, position_ids=source.position_ids)[active]
    torch.testing.assert_close(logits(m, view), expected)
    result = backward_microbatch(
        m,
        source,
        arm="B2",
        step=20,
        num_steps=60,
        generator=generator(),
        policy={"block_size": 2, "warmup_blocks": [1, 2, 8], "decay_blocks": [8, 4, 2]},
    )
    assert result["block_size"] == 8
    assert result["forward_token_positions"] == source.x_0.numel()
    assert result["full_context_single_stream"] == 1


def test_clipped_fragment_exclusion_preserves_source_and_corruption_support():
    source = batch(segments=(3, 5))
    source.loss_mask[:] = True  # Real source-row clips may start without BOS.
    aligned = align_mask_padding(source, block_size=2, mask_token_id=15, exclude_unpredictable_first_targets=True)
    assert int(aligned.batch.loss_mask.sum()) == 8
    assert sorted(aligned.source_indices[aligned.source_indices >= 0].tolist()) == list(range(8))
    view = build_fastv2_view(
        aligned.batch,
        aligned.batch.loss_mask,
        block_size=2,
        mask_token_id=15,
        exclude_unpredictable_first_targets=True,
    )
    assert view.tokens[0, 0] == view.tokens[1, 0] == 15
    assert view.targets.tolist() == [2, 3, 5, 6, 7, 8]
    m = model()
    result = backward_microbatch(
        m,
        source,
        arm="B3",
        step=0,
        num_steps=3520,
        generator=generator(),
        policy={"block_size": 2, "t_min": 1.0, "exclude_unpredictable_first_targets": True},
    )
    assert result["source_positions"] == result["lossable_tokens"] == result["corrupted_tokens"] == 8
    assert result["objective_target_tokens"] == result["active_tokens"] == 6
    assert result["excluded_unpredictable_first_targets"] == 2
    assert result["loss"] == pytest.approx(result["raw_ce"])
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_fastv2_backward_counts_source_once_and_expanded_work_separately():
    source = batch(segments=(3, 5))
    m = model()
    result = backward_microbatch(
        m, source, arm="B3", step=0, num_steps=3520, generator=generator(), policy={"block_size": 2}
    )
    assert result["source_positions"] == 8
    assert result["active_tokens"] == result["lossable_tokens"] == 6
    assert result["view_count"] == 2
    assert result["aligned_rows"] == 2
    assert result["forward_calls"] == result["backward_calls"] == 4
    assert result["forward_token_positions"] == result["backward_token_positions"] == 64
    assert result["loss"] == pytest.approx(result["raw_ce"])
    assert all(math.isfinite(value) for value in result.values())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_uncached_shifted_readout_matches_training_with_clean_preceding_block():
    m = model(4)
    source = batch(segments=(4,))
    active = torch.tensor([[False, False, True, True]])
    view = build_fastv2_view(source, active, block_size=2, mask_token_id=15)
    expected = logits(m, view).view(1, 2, -1)
    actual = forward_block_logits(m, source.x_0[:, :2], torch.full((1, 2), 15), policy={"block_size": 2})
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    actual.sum().backward()
    assert m.wte.weight.grad.abs().sum() > 0


def test_shifted_readout_block_start_does_not_see_current_block_but_interior_does():
    m = model()
    prefix = torch.tensor([[1, 2]])
    a = forward_block_logits(m, prefix, torch.tensor([[15, 15]]), policy={"block_size": 2})
    b = forward_block_logits(m, prefix, torch.tensor([[15, 7]]), policy={"block_size": 2})
    torch.testing.assert_close(a[:, 0], b[:, 0], rtol=0, atol=0)
    assert not torch.allclose(a[:, 1], b[:, 1])
