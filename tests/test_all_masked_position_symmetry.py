import torch
from nanodllm.model import DiffusionConfig, DiffusionTransformer

MASK_ID = 1
SEQ_LEN = 16


def _build_model(
    n_layer: int,
    seed: int = 0,
    mask_position_embedding: bool = False,
) -> DiffusionTransformer:
    cfg = DiffusionConfig(
        sequence_len=SEQ_LEN,
        vocab_size=32,
        n_layer=n_layer,
        n_head=2,
        n_embd=16,
        mask_token_id=MASK_ID,
        mask_position_embedding=mask_position_embedding,
    )
    model = DiffusionTransformer(cfg, compute_dtype=torch.float32)
    # The default init zeroes residual output projections, which would make the
    # symmetry-breaking test vacuous (attention contributes nothing at init).
    # The invariance claim is weight-independent, so test with generic weights.
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.randn(param.shape, generator=generator) * 0.2)
    return model.eval()


def test_all_masked_input_yields_position_constant_logits():
    # RoPE rotates only queries/keys; values carry no position signal. With a
    # position-constant input every attention output is a convex combination of
    # identical value vectors, so logits must be identical at every position,
    # at any depth, for any weights.
    for n_layer in (1, 4):
        model = _build_model(n_layer)
        z = torch.full((2, SEQ_LEN), MASK_ID, dtype=torch.long)
        with torch.no_grad():
            logits = model(z)
        reference = logits[:, :1, :]
        assert torch.allclose(logits, reference.expand_as(logits), atol=1e-4), (
            f"all-masked logits differ across positions at depth {n_layer}: "
            f"max diff {(logits - reference).abs().max().item()}"
        )


def test_mask_position_embedding_breaks_all_masked_position_symmetry():
    model = _build_model(n_layer=1, mask_position_embedding=True)
    z = torch.full((1, SEQ_LEN), MASK_ID, dtype=torch.long)
    with torch.no_grad():
        logits = model(z)
    spread = (logits[:, 1:, :] - logits[:, :1, :]).abs().max().item()
    assert spread > 1e-3, f"mask-position embeddings did not break all-mask symmetry (spread {spread})"


def test_mask_position_embedding_leaves_visible_tokens_unchanged():
    model = _build_model(n_layer=2, mask_position_embedding=True)
    z = torch.arange(SEQ_LEN, dtype=torch.long).view(1, SEQ_LEN) + 2
    with torch.no_grad():
        logits_before = model(z)
        model.mask_position_embedding.weight.mul_(100.0)
        logits_after = model(z)
    assert torch.allclose(logits_before, logits_after, atol=1e-5), (
        "mask-position embeddings changed an input with no mask tokens: "
        f"max diff {(logits_before - logits_after).abs().max().item()}"
    )


def test_all_masked_logits_do_not_encode_row_length():
    # Attention weights sum to 1 over the attended set, so the shared output is
    # also independent of how many (identical) tokens are attended.
    model = _build_model(n_layer=2)
    z = torch.full((2, SEQ_LEN), MASK_ID, dtype=torch.long)
    attention_mask = torch.ones((2, SEQ_LEN), dtype=torch.bool)
    attention_mask[1, 10:] = False
    with torch.no_grad():
        logits = model(z, attention_mask=attention_mask)
    assert torch.allclose(logits[0, 0], logits[1, 0], atol=1e-4), (
        f"all-masked logits depend on attended row length: max diff {(logits[0, 0] - logits[1, 0]).abs().max().item()}"
    )


def test_single_visible_token_breaks_position_symmetry():
    # One non-mask token makes one value vector differ, so masked positions
    # receive position-dependent mixtures and the degeneracy disappears.
    model = _build_model(n_layer=1)
    z = torch.full((1, SEQ_LEN), MASK_ID, dtype=torch.long)
    z[0, 3] = 7
    with torch.no_grad():
        logits = model(z)
    masked_positions = [i for i in range(SEQ_LEN) if i != 3]
    masked_logits = logits[0, masked_positions]
    spread = (masked_logits - masked_logits[:1]).abs().max().item()
    assert spread > 1e-3, f"visible token did not break position symmetry (spread {spread})"
