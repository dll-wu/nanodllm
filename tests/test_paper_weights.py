import json
from dataclasses import asdict, replace

import pytest
import torch
from nanodllm.compute import compute_ledger
from nanodllm.evaluation.checkpoint import tokenizer_identity
from nanodllm.runtime.config import ModelConfig
from nanodllm.runtime.data import ByteTokenizer
from nanodllm.runtime.model import DiffusionConfig, DiffusionTransformer
from nanodllm.weights import export_weights, file_sha256, load_weights, portable_parent


def _parent(tmp_path):
    architecture = DiffusionConfig(
        sequence_len=8, vocab_size=260, n_layer=1, n_head=2, n_embd=16, mask_token_id=257, causal=True
    )
    settings = ModelConfig(depth=1, n_head=2, n_embd=16, max_seq_len=8, causal=True, pad_vocab_to=1)
    model = DiffusionTransformer(
        architecture, compute_dtype=torch.float32, rope_base=settings.rope_base, pad_vocab_to=1
    )
    model.init_weights()
    state = {
        "config": asdict(architecture),
        "train_config": {
            "model": asdict(settings),
            "data": {"tokenizer": "byte", "train_data": ["/example/private-corpus.parquet"]},
            "noise": {"type": "mdlm"},
            "train": {"run_dir": "/example/run"},
        },
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "rng_state": {},
        "step": 2,
        "train_tokens": 64,
        "train_valid_tokens": 63,
    }
    native = tmp_path / "model.pt"
    torch.save(state, native)
    return model.eval(), native


def test_export_roundtrip_and_parent_binding(tmp_path):
    parent, native = _parent(tmp_path)
    weights = tmp_path / "parent.weights.pt"
    output_digest = export_weights(native, weights, expected_source_sha256=file_sha256(native))
    loaded = load_weights(weights, expected_checkpoint_sha256=output_digest)
    inputs = torch.tensor([[256, 42, 82]])
    with torch.inference_mode():
        assert torch.equal(parent(inputs), loaded.model(inputs))
    payload = torch.load(weights, weights_only=True)
    assert payload["source_sha256"] == file_sha256(native)
    assert {"optimizer_state_dict", "rng_state", "train_config"}.isdisjoint(payload)
    assert "/example/" not in str({k: v for k, v in payload.items() if k != "model_state_dict"})
    with pytest.raises(FileExistsError):
        export_weights(native, weights, expected_source_sha256=file_sha256(native))
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        load_weights(weights, expected_checkpoint_sha256="0" * 64)
    with pytest.raises(ValueError, match="tokenizer identity"):
        portable_parent(payload, target_model=parent, expected_tokenizer_sha256="1" * 64)
    other_rope = DiffusionTransformer(
        replace(parent.config, causal=False), rope_base=20000, compute_dtype=torch.float32
    )
    with pytest.raises(ValueError, match="RoPE"):
        portable_parent(payload, target_model=other_rope, expected_tokenizer_sha256=tokenizer_identity(ByteTokenizer()))
    with pytest.raises(ValueError, match="expected a nanodllm.weights"):
        load_weights(native)


def test_compute_resumes_and_verified_two_stage_parent(tmp_path):
    _, parent = _parent(tmp_path)
    child = tmp_path / "child"
    child.mkdir()
    metrics = {
        "status": "ok",
        "num_params": 5000,
        "total_tokens": 64,
        "flops_used": 1000,
        "flops_accounting_method": "base_training_estimate",
        "world_size": 2,
        "runtime_device_type": "cuda",
        "training_seconds": 1.5,
    }
    param = {"target_param_count": 2000, "target_param_count_basis": "transformer_body_parameters"}
    for root, resumed, source in [
        (tmp_path, 0, None),
        (child, 1, {"sha256": file_sha256(parent), "source_train_tokens": 64}),
    ]:
        (root / "metrics.json").write_text(json.dumps({**metrics, "resume_consumed_microbatches": resumed}))
        (root / "run_summary.json").write_text(
            json.dumps({"parameter_accounting": param, "initial_state": {"model_initialization_source": source}})
        )
    assert compute_ledger(tmp_path)["totals"]["gpu_seconds"] == 3.0
    result = compute_ledger(child, parent_run=tmp_path)
    assert result["totals"] == {
        "processed_tokens": 128,
        "estimated_training_flops": 2000,
        "training_seconds": None,
        "gpu_seconds": None,
    }
    with pytest.raises(ValueError, match="supply the parent"):
        compute_ledger(child)


def test_completep_portable_parents_match_all_full_sequence_profiles(tmp_path):
    from types import SimpleNamespace

    from nanodllm.runtime.training_plan import _resolve_diffusion_config
    from recipes.launch import read_profiles, resolve

    profiles = read_profiles(group="full-adamw")
    assert any(profile.parent for profile in profiles.values())
    vocab = SimpleNamespace(get_vocab_size=lambda: 32768, get_mask_token_id=lambda: 32758)
    for name, profile in profiles.items():
        if profile.group != "full-adamw" or not profile.parent:
            continue
        common = {"corpus": tmp_path, "tokenizer": tmp_path, "run_dir": tmp_path}
        old = resolve(profiles, profile.parent, **common)
        new = resolve(profiles, name, parent=tmp_path / "weights.pt", **common)
        target = SimpleNamespace(config=_resolve_diffusion_config(new, vocab), rope_base=new.model.rope_base)
        payload = {
            "config": asdict(_resolve_diffusion_config(old, vocab)),
            "model": asdict(old.model),
            "schema": "nanodllm.weights.v1",
            "tokenizer_sha256": "f" * 64,
            "model_state_dict": {},
            "step": 1,
            "train_tokens": 1,
            "train_valid_tokens": 1,
        }
        assert (
            portable_parent(payload, target_model=target, expected_tokenizer_sha256="f" * 64)["config"]
            == payload["config"]
        )
