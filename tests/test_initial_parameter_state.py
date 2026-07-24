import hashlib
import json

import torch
import torch.nn as nn
from nanodllm.training import _pre_update_parameter_state_artifact


class _TinyTiedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(7, 4, dtype=torch.bfloat16)
        self.projection = nn.Linear(4, 3, bias=True)
        self.tied_alias = self.embedding


def _artifact(model: nn.Module, *, seed: int = 17) -> dict[str, object]:
    return _pre_update_parameter_state_artifact(
        model,
        architecture_fingerprint_sha256="a" * 64,
        model_seed=seed,
    )


def test_pre_update_parameter_state_artifact_is_exact_and_stable() -> None:
    torch.manual_seed(4)
    model = _TinyTiedModel()

    first = _artifact(model)
    second = _artifact(model)

    assert first == second
    assert first["schema"] == "nanodllm.initial_state.v1"
    assert first["initial_state_sha256"] == first["pre_update_parameter_state_sha256"]
    assert len(str(first["pre_update_parameter_state_sha256"])) == hashlib.sha256().digest_size * 2
    assert first["model_seed"] == 17
    assert (
        first["initialization_contract_sha256"]
        == hashlib.sha256(
            json.dumps(
                first["initialization_contract"],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
    )
    assert first["parameter_count"] == 3
    assert first["parameter_numel"] == sum(parameter.numel() for parameter in model.parameters())
    assert first["parameter_bytes"] == sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )


def test_pre_update_parameter_state_artifact_changes_with_parameter_bytes() -> None:
    torch.manual_seed(4)
    model = _TinyTiedModel()
    before = _artifact(model)

    with torch.no_grad():
        model.projection.bias[0] += 1
    after = _artifact(model)

    assert before["pre_update_parameter_state_sha256"] != after["pre_update_parameter_state_sha256"]


def test_pre_update_parameter_state_digest_excludes_seed_metadata() -> None:
    torch.manual_seed(4)
    model = _TinyTiedModel()

    seed_17 = _artifact(model, seed=17)
    seed_18 = _artifact(model, seed=18)

    assert seed_17["pre_update_parameter_state_sha256"] == seed_18["pre_update_parameter_state_sha256"]
    assert seed_17["model_seed"] != seed_18["model_seed"]
