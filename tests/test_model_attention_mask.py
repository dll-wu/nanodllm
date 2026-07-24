import pytest
import torch
from nanodllm.model import DiffusionConfig, DiffusionTransformer, bd3lm_attention_mask, duplicated_position_ids


def test_model_accepts_attention_mask():
    cfg = DiffusionConfig(sequence_len=8, vocab_size=32, n_layer=2, n_head=2, n_embd=16, mask_token_id=1)
    model = DiffusionTransformer(cfg, compute_dtype=torch.float32)
    z = torch.randint(0, 32, (2, 8))
    attention_mask = torch.ones((2, 8), dtype=torch.bool)
    attention_mask[0, 6:] = False
    logits = model(z, attention_mask=attention_mask)
    assert logits.shape == (2, 8, 32)


def test_model_accepts_noise_level_when_configured():
    cfg = DiffusionConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_embd=16,
        mask_token_id=1,
        time_conditioning=True,
    )
    model = DiffusionTransformer(cfg, compute_dtype=torch.float32)
    z = torch.randint(0, 32, (2, 8))

    logits = model(z, noise_level=torch.tensor([0.25, 0.75]))
    inferred_logits = model(z)

    assert logits.shape == (2, 8, 32)
    assert inferred_logits.shape == (2, 8, 32)


def test_model_accepts_soft_input_embeddings():
    cfg = DiffusionConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_embd=16,
        mask_token_id=1,
        time_conditioning=True,
    )
    model = DiffusionTransformer(cfg, compute_dtype=torch.float32)
    z = torch.randint(0, 32, (2, 8))
    input_embeddings = model.wte(z).detach()
    input_embeddings[:, 0, :] = 0.0

    logits = model(z, input_embeddings=input_embeddings, noise_level=torch.tensor([0.25, 0.75]))

    assert logits.shape == (2, 8, 32)


def test_model_length_head_returns_sequence_position_logits():
    cfg = DiffusionConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_embd=16,
        mask_token_id=1,
        length_head=True,
    )
    model = DiffusionTransformer(cfg, compute_dtype=torch.float32)
    z = torch.randint(0, 32, (2, 8))
    attention_mask = torch.ones((2, 8), dtype=torch.bool)
    attention_mask[0, 6:] = False

    logits, length_logits = model(z, attention_mask=attention_mask, return_length_logits=True)

    assert logits.shape == (2, 8, 32)
    assert length_logits.shape == (2, 8)


def test_masked_suffix_keys_do_not_change_valid_prefix_outputs():
    torch.manual_seed(0)
    cfg = DiffusionConfig(sequence_len=8, vocab_size=32, n_layer=2, n_head=2, n_embd=16, mask_token_id=1)
    model = DiffusionTransformer(cfg, compute_dtype=torch.float32).eval()
    z1 = torch.randint(0, 32, (1, 8))
    z2 = z1.clone()
    z2[:, 6:] = torch.randint(0, 32, (1, 2))
    attention_mask = torch.ones((1, 8), dtype=torch.bool)
    attention_mask[:, 6:] = False

    y1 = model(z1, attention_mask=attention_mask)
    y2 = model(z2, attention_mask=attention_mask)
    assert torch.allclose(y1[:, :6], y2[:, :6], atol=1e-5)


def test_causal_model_outputs_do_not_depend_on_future_tokens():
    torch.manual_seed(0)
    cfg = DiffusionConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_embd=16,
        mask_token_id=1,
        causal=True,
    )
    model = DiffusionTransformer(cfg, compute_dtype=torch.float32).eval()
    z1 = torch.randint(0, 32, (1, 8))
    z2 = z1.clone()
    z2[:, 5:] = torch.randint(0, 32, (1, 3))

    y1 = model(z1)
    y2 = model(z2)

    assert torch.allclose(y1[:, :5], y2[:, :5], atol=1e-5)


def test_residual_output_projection_default_init_is_zero():
    torch.manual_seed(0)
    cfg = DiffusionConfig(sequence_len=8, vocab_size=32, n_layer=2, n_head=2, n_embd=16, mask_token_id=1)
    model = DiffusionTransformer(cfg, compute_dtype=torch.float32)

    for block in model.blocks:
        assert torch.count_nonzero(block.attn.c_proj.weight) == 0
        assert torch.count_nonzero(block.mlp.down.weight) == 0


def test_residual_output_projection_scaled_normal_init_uses_block_sigma():
    torch.manual_seed(0)
    cfg = DiffusionConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=2,
        n_head=2,
        n_embd=32,
        mask_token_id=1,
        residual_output_projection_init="scaled_normal",
    )
    model = DiffusionTransformer(cfg, compute_dtype=torch.float32)

    attn_std = float(model.blocks[0].attn.c_proj.weight.std().item())
    mlp_std = float(model.blocks[0].mlp.down.weight.std().item())
    assert torch.count_nonzero(model.blocks[0].attn.c_proj.weight) > 0
    assert torch.count_nonzero(model.blocks[0].mlp.down.weight) > 0
    assert 0.005 < attn_std < 0.02
    assert 0.005 < mlp_std < 0.02


def test_embedding_head_init_multiplier_only_scales_embedding_and_lm_head():
    cfg = DiffusionConfig(sequence_len=4, vocab_size=64, n_layer=1, n_head=4, n_embd=32, mask_token_id=1)
    scaled_cfg = DiffusionConfig(
        sequence_len=4,
        vocab_size=64,
        n_layer=1,
        n_head=4,
        n_embd=32,
        mask_token_id=1,
        embedding_head_init_std_multiplier=2.0,
    )

    torch.manual_seed(123)
    base_model = DiffusionTransformer(cfg, compute_dtype=torch.float32)
    torch.manual_seed(123)
    scaled_model = DiffusionTransformer(scaled_cfg, compute_dtype=torch.float32)

    assert torch.allclose(scaled_model.wte.weight, base_model.wte.weight * 2.0)
    assert torch.allclose(scaled_model.lm_head.weight, base_model.lm_head.weight * 2.0)
    assert torch.allclose(scaled_model.blocks[0].attn.c_q.weight, base_model.blocks[0].attn.c_q.weight)


def test_bd3lm_attention_mask_components():
    attention_mask = torch.ones((1, 8), dtype=torch.bool)
    attention_mask[0, 7] = False
    mask = bd3lm_attention_mask(attention_mask, block_size=2)

    assert mask.shape == (1, 16, 16)
    assert bool(mask[0, 0, 1])  # noised block diagonal
    assert not bool(mask[0, 0, 8])  # no same-block clean leakage into x_t
    assert bool(mask[0, 2, 8])  # noised block can see prior clean block
    assert not bool(mask[0, 2, 10])  # noised block cannot see same clean block
    assert not bool(mask[0, 2, 0])  # noised block cannot see prior noised block
    assert bool(mask[0, 10, 8])  # clean half is block-causal
    assert bool(mask[0, 10, 10])
    assert not bool(mask[0, 10, 12])
    assert not bool(mask[:, :, 7].any())
    assert not bool(mask[:, :, 15].any())


def test_model_accepts_duplicated_position_ids_for_bd3lm_concat():
    cfg = DiffusionConfig(sequence_len=8, vocab_size=32, n_layer=2, n_head=2, n_embd=16, mask_token_id=1)
    model = DiffusionTransformer(cfg, compute_dtype=torch.float32)
    z = torch.randint(0, 32, (2, 16))
    attention_mask = torch.ones((2, 8), dtype=torch.bool)
    bd3lm_mask = bd3lm_attention_mask(attention_mask, block_size=4)
    position_ids = duplicated_position_ids(8)

    logits = model(z, attention_mask=bd3lm_mask, position_ids=position_ids)

    assert logits.shape == (2, 16, 32)


@pytest.mark.parametrize("invalid_position", [-1, 8])
def test_model_eager_rejects_out_of_range_position_ids(invalid_position):
    cfg = DiffusionConfig(sequence_len=8, vocab_size=32, n_layer=1, n_head=2, n_embd=16, mask_token_id=1)
    model = DiffusionTransformer(cfg, compute_dtype=torch.float32).eval()
    z = torch.randint(0, 32, (1, 8))
    position_ids = torch.arange(8)
    position_ids[3] = invalid_position

    with pytest.raises(ValueError, match="valid RoPE positions"):
        model(z, position_ids=position_ids)


def test_compiled_model_captures_position_validation_in_full_graph():
    cfg = DiffusionConfig(sequence_len=8, vocab_size=32, n_layer=1, n_head=2, n_embd=16, mask_token_id=1)
    model = DiffusionTransformer(cfg, compute_dtype=torch.float32).eval()
    compiled_model = torch.compile(model, backend="eager", fullgraph=True)
    z = torch.randint(0, 32, (1, 8))

    logits = compiled_model(z, position_ids=torch.arange(8))

    assert logits.shape == (1, 8, 32)
    for invalid_position in (-1, 8):
        position_ids = torch.arange(8)
        position_ids[3] = invalid_position
        with pytest.raises((IndexError, RuntimeError), match="index out of range|out of bounds"):
            compiled_model(z, position_ids=position_ids)


def test_hierarchical_output_logprobs_and_ce_are_exact():
    cfg = DiffusionConfig(
        sequence_len=4,
        vocab_size=12,
        n_layer=1,
        n_head=2,
        n_embd=8,
        mask_token_id=1,
        output_factorization="hierarchical_frequency_decile",
    )
    token_class_ids = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3], dtype=torch.long)
    model = DiffusionTransformer(cfg, compute_dtype=torch.float32, token_class_ids=token_class_ids).eval()
    z = torch.randint(0, 12, (2, 4))
    targets = torch.randint(0, 12, (2, 4))

    class_logits, token_logits = model(z, return_hierarchical_logits=True)
    log_probs = model.hierarchical_token_log_probs(class_logits, token_logits)
    ce, class_ce, within_class_ce = model.hierarchical_ce(class_logits, token_logits, targets)

    dense_ce = torch.nn.functional.cross_entropy(
        log_probs.reshape(-1, log_probs.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).view_as(targets)

    assert log_probs.shape == (2, 4, 12)
    assert torch.allclose(torch.logsumexp(log_probs, dim=-1), torch.zeros_like(log_probs[..., 0]), atol=1e-5)
    assert torch.allclose(ce, dense_ce, atol=1e-5)
    assert torch.allclose(ce, class_ce + within_class_ce, atol=1e-6)
