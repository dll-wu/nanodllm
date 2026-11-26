"""Boundaries that keep ordinary experiments runnable independently of the paper."""

import json
from pathlib import Path

from nanodllm import cli, training
from nanodllm.runtime import cli as runtime_cli
from nanodllm.runtime.training_loop import train
from recipes import launch


def test_public_training_routes_share_one_implementation():
    import pretrain

    assert pretrain.main is cli.main is runtime_cli.main
    assert training.train is runtime_cli.train is train


def test_default_recipe_list_only_reads_editable_catalogs(monkeypatch, tmp_path, capsys):
    # An incomplete frozen catalog exposes accidental reads from the default listing.
    (tmp_path / "block-muon.json").write_text("broken")
    monkeypatch.setattr(launch, "PAPER_PROFILES", tmp_path)
    launch.main(["list"])
    lines = capsys.readouterr().out.splitlines()
    assert {row.split("\t")[1] for row in lines[1:]} == {"tiny", "d4-matched"}


def test_original_config_paths_use_the_relocated_recipes():
    root = Path(__file__).resolve().parents[1]
    for config in (root / "recipes" / "preview").glob("*.json"):
        assert json.loads((root / "configs" / config.name).read_bytes()) == json.loads(config.read_bytes())
