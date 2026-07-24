import json
from pathlib import Path

import pyarrow.parquet as pq
from nanodllm.config import DataConfig, EvalConfig, load_config
from scripts.prepare_tiny_corpus import prepare_corpus

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPOSITORY_ROOT / "configs"


def test_public_defaults_do_not_assume_a_private_filesystem():
    assert DataConfig().tokenizer_path == "tokenizers/climbmix-32k"
    assert EvalConfig().judge_model_id == "Qwen/Qwen3-0.6B-Base"


def test_load_config_expands_environment_variables_recursively(tmp_path, monkeypatch):
    monkeypatch.setenv("NANODLLM_TEST_ROOT", str(tmp_path))
    config_path = tmp_path / "environment.json"
    config_path.write_text(
        json.dumps(
            {
                "data": {
                    "tokenizer": "byte",
                    "tokenizer_path": "",
                    "train_data": ["${NANODLLM_TEST_ROOT}/train.parquet"],
                },
                "train": {"run_dir": "${NANODLLM_TEST_ROOT}/run"},
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.data.train_data == [str(tmp_path / "train.parquet")]
    assert config.train.run_dir == str(tmp_path / "run")


def test_small_climbmix_recipe_supports_a_local_subset(monkeypatch, tmp_path):
    tokenizer_dir = tmp_path / "tokenizer"
    data_dir = tmp_path / "climbmix"
    monkeypatch.setenv("NANODLLM_TOKENIZER_DIR", str(tokenizer_dir))
    monkeypatch.setenv("NANODLLM_DATA_DIR", str(data_dir))

    config = load_config(CONFIG_DIR / "climbmix_mdlm_small.json")

    assert config.data.tokenizer_path == str(tokenizer_dir)
    assert config.data.train_data == [str(data_dir)]
    assert config.data.train_exclude_data == []
    assert config.data.val_data == []
    assert config.data.val_fraction == 0.02
    assert config.recipe.status == "provisional"


def test_d20_climbmix_recipe_reserves_the_validation_shard(monkeypatch, tmp_path):
    tokenizer_dir = tmp_path / "tokenizer"
    data_dir = tmp_path / "climbmix"
    monkeypatch.setenv("NANODLLM_TOKENIZER_DIR", str(tokenizer_dir))
    monkeypatch.setenv("NANODLLM_DATA_DIR", str(data_dir))

    config = load_config(CONFIG_DIR / "climbmix_mdlm_d20.json")
    validation_path = str(data_dir / "shard_06542.parquet")

    assert config.data.tokenizer_path == str(tokenizer_dir)
    assert config.data.train_data == [str(data_dir / "shard_*.parquet")]
    assert config.data.train_exclude_data == [validation_path]
    assert config.data.val_data == [validation_path]
    assert config.recipe.status == "provisional"


def test_d20_recipe_preserves_the_public_scale_up_shape():
    config = load_config(CONFIG_DIR / "climbmix_mdlm_d20.json")

    assert config.model.depth == 20
    assert config.model.n_embd == 1280
    assert config.model.n_head == 10
    assert config.model.max_seq_len == 2048
    assert config.model.mlp_hidden_dim == 5120
    assert config.model.attention_scale == 0.08838834764831843
    assert config.model.rope_base == 100000.0


def test_bayes_audit_is_self_contained_and_cpu_only():
    config = load_config(CONFIG_DIR / "mdlm_bayes_analytic_audit.json")

    assert config.recipe.name == "mdlm-bayes-analytic-audit"
    assert config.data.synthetic_dataset == "eos_overfit_canary"
    assert config.data.train_data == []
    assert config.runtime.device_type == "cpu"
    assert config.train.strict_mdlm_all_token_bayes_audit_only is True


def test_tiny_corpus_preparation_is_deterministic_and_matches_smoke_recipe(tmp_path):
    source = REPOSITORY_ROOT / "examples" / "tiny_corpus.txt"
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"

    first_result = prepare_corpus(source, first, row_group_size=4)
    second_result = prepare_corpus(source, second, row_group_size=4)

    assert first.read_bytes() == second.read_bytes()
    assert first_result["content_sha256"] == second_result["content_sha256"]
    assert first_result["parquet_sha256"] == second_result["parquet_sha256"]
    parquet_file = pq.ParquetFile(first)
    assert parquet_file.metadata.num_rows == 32
    assert parquet_file.num_row_groups == 8
    assert parquet_file.schema_arrow.names == ["text"]
    assert load_config(CONFIG_DIR / "smoke_byte.json").data.train_data == ["data/tiny_corpus.parquet"]
