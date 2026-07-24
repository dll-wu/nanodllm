import hashlib

import pytest
from nanodllm.config import DataConfig, EvalConfig, ExperimentConfig, TrainConfig, validate_config
from nanodllm.training import _source_checkpoint_identity


def _eval_only_identity_config() -> ExperimentConfig:
    return ExperimentConfig(
        data=DataConfig(train_data=["/tmp/train.parquet"]),
        train=TrainConfig(eval_only=True, resume_from="/tmp/model.pt"),
        eval=EvalConfig(
            source_checkpoint_identity_mode="sha256",
            source_checkpoint_expected_sha256="a" * 64,
            source_checkpoint_expected_size_bytes=3,
        ),
    )


def test_source_checkpoint_identity_hashes_the_exact_bytes(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"abc")

    digest, size_bytes = _source_checkpoint_identity(str(checkpoint))

    assert digest == hashlib.sha256(b"abc").hexdigest()
    assert size_bytes == 3


def test_source_checkpoint_identity_contract_is_explicit_and_eval_only():
    config = _eval_only_identity_config()
    validate_config(config)

    config.eval.source_checkpoint_expected_sha256 = "A" * 64
    with pytest.raises(ValueError, match="lowercase 64-hex"):
        validate_config(config)

    config = _eval_only_identity_config()
    config.train.eval_only = False
    with pytest.raises(ValueError, match="requires train.eval_only=true"):
        validate_config(config)

    config = _eval_only_identity_config()
    config.eval.source_checkpoint_identity_mode = "off"
    with pytest.raises(ValueError, match="require identity_mode='sha256'"):
        validate_config(config)
