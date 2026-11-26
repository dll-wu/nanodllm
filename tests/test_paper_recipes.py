import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch
from nanodllm.runtime.config import load_config
from nanodllm.runtime.model import DiffusionTransformer
from nanodllm.runtime.parameterization import bind_parameterization_contract, resolve_parameterization_contract
from nanodllm.runtime.training_plan import _resolve_diffusion_config
from recipes import launch as recipes
from recipes.data import VALIDATION_SHARD, file_sha256, verify_corpus


class PaperVocabulary:
    def get_vocab_size(self):
        return 32768

    def get_mask_token_id(self):
        return 32758


def test_catalog_model_contracts_and_lineage(tmp_path):
    profiles = recipes.read_profiles()
    assert Counter(p.group for p in profiles.values()) == {
        "full-adamw": 18,
        "block-muon": 21,
        "development": 59,
        "development-adamw-grid": 72,
        "full-bracket": 22,
        "block-bracket": 11,
        "tiny": 3,
        "d4-matched": 3,
    }
    for p in profiles.values():
        if not p.frozen:
            continue
        cfg = recipes.resolve(
            profiles,
            p.name,
            corpus=tmp_path,
            run_dir=tmp_path / "out",
            parent=tmp_path / "parent.pt" if p.parent else None,
            tokenizer=tmp_path / "tokenizer",
        )
        assert cfg.train.num_steps * cfg.train.total_batch_size == cfg.train.target_tokens, p.name
        assert cfg.train.total_batch_size % (cfg.train.device_batch_size * p.world_size * cfg.model.max_seq_len) == 0
        assert cfg.data.val_data == cfg.data.train_exclude_data == [str(tmp_path / VALIDATION_SHARD)]
        assert cfg.data.val_fraction == 0 and cfg.data.content_budget_segmented_packing
        assert cfg.optim.nanochat_checkout == "" and cfg.train.resume_from == ""
        assert cfg.data.bos_packing_mode == (
            "none" if p.group.startswith("full-") or p.group.endswith("adamw-grid") else "bestfit_segmented"
        )
        if p.parent:
            source = profiles[p.parent]
            assert source.settings["model"]["causal"]
            assert cfg.train.model_init_expected_sha256 == source.checkpoint_sha256
            assert cfg.train.model_init_expected_size_bytes == source.checkpoint_bytes
        with torch.device("meta"):
            m = DiffusionTransformer(
                _resolve_diffusion_config(cfg, PaperVocabulary()),
                rope_base=cfg.model.rope_base,
                compute_dtype=torch.bfloat16,
                pad_vocab_to=cfg.model.pad_vocab_to,
            )
        contract = resolve_parameterization_contract(m, cfg, batch_lr_scale=1.0)
        assert cfg.parameterization.contract_sha256 == contract["contract_sha256"], p.name
        bind_parameterization_contract(cfg, contract)


def test_historical_variants_and_rank_topology():
    profiles = recipes.read_profiles()
    for seed in (50, 51, 52):
        parent = f"block-d8-parent-s{seed}"
        a, b = (profiles[f"block-d8-{role}-s{seed}"] for role in ("convert", "convert-alt"))
        assert a.parent == b.parent == parent
        assert [p.settings["optim"]["lr"] for p in (a, b)] == [0.00175, 0.0035]
        assert a.settings["runtime"]["training_seed"] == seed
    assert len({profiles[f"block-d8-parent-s{s}"].checkpoint_sha256 for s in (50, 51, 52)}) == 3
    assert profiles["dev-37m-ar10-mdlm30-s48"].parent == "dev-37m-ar10-s48"
    assert profiles["dev-37m-ar10-s48"].settings["model"]["backbone_profile"] == "nanochat_block_v1"
    assert profiles["dev-ar10-s48"].settings["model"]["backbone_profile"] == "nanollada_v1"
    assert profiles["full-d8-convert"].parent == "full-d8-parent"
    assert (
        profiles["full-d8-parent"].checkpoint_sha256
        == "c402f24d115304d17866f3b805195ecdd83c7c7d3148705538a425f5d38330f9"
    )
    assert profiles["full-d10-parent"].world_size == 1
    assert profiles["full-d10-convert"].world_size == 8
    assert profiles["block-d12-parent-s50"].world_size == 4
    assert "--nproc-per-node=8" in recipes.launch_command(profiles["block-d20-scratch-s50"], Path("resolved.json"))
    assert "torch.distributed.run" not in recipes.launch_command(profiles["tiny-ar"], Path("resolved.json"))


def test_materialization_binds_hash_and_is_idempotent(tmp_path):
    ps = recipes.read_profiles()
    kw = dict(corpus=tmp_path / "corpus", tokenizer=tmp_path / "tokens", run_dir=tmp_path / "run")
    with pytest.raises(ValueError, match="--parent"):
        recipes.resolve(ps, "block-d8-convert-s52", **kw)
    with pytest.raises(ValueError, match="no parent"):
        recipes.resolve(ps, "block-d8-scratch-s52", **kw, parent=tmp_path / "weights")
    with pytest.raises(ValueError, match="portable weights require"):
        recipes.resolve(ps, "block-d8-convert-s52", **kw, parent=tmp_path / "parent.weights.pt")
    cfg = recipes.resolve(
        ps,
        "block-d8-convert-s52",
        **kw,
        parent=tmp_path / "parent.weights.pt",
        parent_sha256="d" * 64,
        parent_bytes=1234,
    )
    assert cfg.train.model_init_expected_sha256 == "d" * 64 and cfg.train.model_init_expected_size_bytes == 1234
    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs = list(pool.map(lambda _: recipes.write_resolved(cfg, tmp_path / "out"), range(3)))
    (path, digest), *_ = outputs
    assert len(set(outputs)) == 1
    assert path.stem == f"block-d8-convert-s52-{digest}"
    assert file_sha256(path) == digest
    assert load_config(path) == cfg


def test_corpus_inventory_rejects_subset(tmp_path):
    (tmp_path / VALIDATION_SHARD).touch()
    with pytest.raises(ValueError, match="shard names and byte sizes"):
        verify_corpus(tmp_path)


def test_editable_d4_stages_match_reference_settings():
    profiles = recipes.read_profiles()
    for name, stage in (("d4-ar10", "parent"), ("d4-scratch40", "scratch"), ("d4-block30", "convert")):
        editable = profiles[name]
        frozen = profiles[f"block-d4-{stage}-s48"]
        assert editable.world_size == frozen.world_size == 1
        assert editable.checkpoint_sha256 is None
        assert editable.validation_shard == frozen.validation_shard
        for section, fields in frozen.settings.items():
            if section == "recipe":
                continue
            expected = fields.copy()
            if section == "parameterization":
                expected.pop("contract_sha256")
            assert editable.settings[section] == expected
    parent, scratch, child = (profiles[name] for name in ("d4-ar10", "d4-scratch40", "d4-block30"))
    assert child.parent == parent.name
    assert scratch.parent is None and parent.parent is None
    assert scratch.settings["train"]["target_tokens"] == (
        parent.settings["train"]["target_tokens"] + child.settings["train"]["target_tokens"]
    )


def test_reusable_and_custom_cli_without_paper_profiles(tmp_path, monkeypatch, capsys):
    # A present but unreadable catalog also catches accidental eager loading.
    monkeypatch.setattr(recipes, "PAPER_PROFILES", tmp_path / "absent")
    recipes.main(["list"])
    output = capsys.readouterr().out
    assert "d4-block30" in output and "tiny-ar" in output and "full-d8-scratch" not in output
    recipes.PAPER_PROFILES.mkdir()
    (recipes.PAPER_PROFILES / "block-muon.json").write_text("invalid JSON")
    custom = json.loads((recipes.RECIPE_DIR / "profiles/tiny.json").read_text())
    custom["profiles"]["local-ar"] = custom["profiles"].pop("tiny-ar")
    custom["profiles"]["tiny-convert"]["parent"] = "local-ar"
    custom["profiles"]["local-ar"]["overrides"]["train"] = {"num_steps": 7}
    catalog = tmp_path / "custom.json"
    catalog.write_text(json.dumps(custom))
    args = [
        "--corpus",
        str(tmp_path / "local.parquet"),
        "--run-dir",
        str(tmp_path / "run"),
        "--output-dir",
        str(tmp_path),
    ]
    for name, extra, expected_steps in [
        ("tiny-ar", [], 2),
        ("d4-scratch40", ["--tokenizer", str(tmp_path / "tokenizer")], 7040),
        (
            "d4-block30",
            [
                "--tokenizer",
                str(tmp_path / "tokenizer"),
                "--parent",
                "ar.weights.pt",
                "--parent-sha256",
                "a" * 64,
                "--parent-bytes",
                "123",
            ],
            5280,
        ),
        ("local-ar", ["--catalog", str(catalog)], 7),
    ]:
        recipes.main(["materialize", name, *args, *extra])
        receipt = json.loads(capsys.readouterr().out)
        assert load_config(receipt["config"]).train.num_steps == expected_steps
