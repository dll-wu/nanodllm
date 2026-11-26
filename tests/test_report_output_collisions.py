import json
from pathlib import Path

import pytest
import torch
from nanodllm import report_io
from nanodllm.evaluation import cli as corpus
from nanodllm.generation import cli as generation
from nanodllm.runtime.config import DataConfig, ExperimentConfig
from nanodllm.runtime.data_identity import build_data_identity
from nanodllm.runtime.data_identity_io import load_data_identity, write_data_identity
from nanodllm.tasks import cli as tasks
from safetensors.torch import save_file
from scripts.prepare_tiny_corpus import write_parquet
from scripts.tasks_demo import create

GENERATE = "--checkpoint model.pt --prompt hello"
SCORE_TASKS = "score --checkpoint model.pt --roster roster.json --bundle bundle"
SCORE_CORPUS = "score --checkpoint model.pt --panel panel.pt"


def rejected(main, args, output, capsys):
    with pytest.raises(SystemExit) as error:
        main([*args.split(), "--output", str(output)])
    assert error.value.code == 2
    assert "--output" in capsys.readouterr().err.split("error:")[-1]


@pytest.mark.parametrize(
    "main,args,source",
    [
        (generation.main, GENERATE, "model.pt"),
        (generation.main, "--checkpoint model.pt --prompts prompts.jsonl", "prompts.jsonl"),
        (generation.main, "--checkpoint model.pt --study-parquet shard", "shard"),
        (generation.main, GENERATE + " --data-identity identity.json", "identity.json"),
        (generation.main, GENERATE + " --reference-model ref", "ref/nested/shard.safetensors"),
        (generation.main, GENERATE + " --tokenizer tok", "tok/config.json"),
        (tasks.main, SCORE_TASKS, "model.pt"),
        (tasks.main, SCORE_TASKS, "roster.json"),
        (tasks.main, SCORE_TASKS, "bundle"),
        (tasks.main, SCORE_TASKS + " --tokenizer tok", "tok/tokenizer.json"),
        (tasks.main, "compare left right", "left"),
        (tasks.main, "compare left right", "right"),
        (corpus.main, "prepare packed --parquet shard", "shard"),
        (corpus.main, SCORE_CORPUS, "model.pt"),
        (corpus.main, SCORE_CORPUS, "panel.pt"),
        (corpus.main, SCORE_CORPUS + " --manifest manifest", "manifest"),
        (corpus.main, SCORE_CORPUS + " --tokenizer tok", "tok/token_bytes.json"),
        (corpus.main, "compare --reference left --candidate right", "left"),
        (corpus.main, "compare --reference left --candidate right", "right"),
    ],
)
@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink"])
def test_cli_refuses_input_before_loading(main, args, source, alias, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    source = Path(source)
    source.parent.mkdir(exist_ok=True, parents=True)
    if source.suffix == ".safetensors":
        save_file({"weight": torch.zeros(1)}, source)
        index = {"metadata": {"total_size": 4}, "weight_map": {"weight": "nested/shard.safetensors"}}
        Path("ref/model.safetensors.index.json").write_text(json.dumps(index))
    else:
        source.write_bytes(b"original input")
    original = source.read_bytes()
    output = Path("report")
    if alias == "direct":
        output = source
    elif alias == "symlink":
        output.symlink_to(source)
    else:
        output.hardlink_to(source)
    rejected(main, args, output, capsys)
    assert source.read_bytes() == original
    assert output.samefile(source)


@pytest.mark.parametrize(
    "main,args,name",
    [
        (generation.main, GENERATE + " --training-checkpoint --tokenizer tok", "data_identity.v1.json"),
        (tasks.main, SCORE_TASKS + " --training-checkpoint --tokenizer tok", "data_identity.v2.json"),
        (corpus.main, SCORE_CORPUS + " --trusted-training-checkpoint --tokenizer tok", "data_identity.v1.json"),
        (
            generation.main,
            GENERATE + " --training-checkpoint --tokenizer tok --data-identity run",
            "run/data_identity.v2.json",
        ),
    ],
)
def test_training_identity_inputs(main, args, name, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    source = Path(name)
    source.parent.mkdir(exist_ok=True)
    source.write_bytes(b"identity input")
    Path("out").hardlink_to(source)
    rejected(main, args, "out", capsys)
    assert source.read_bytes() == b"identity input"
    assert Path("out").samefile(source)


def test_compact_identity_inventory(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    write_parquet(Path("train.parquet"), ["one", "two"], 1)
    Path("tok").mkdir()
    Path("tok/config.json").write_text("{}")
    config = ExperimentConfig(data=DataConfig(train_data=["train.parquet"], tokenizer="hf", tokenizer_path="tok"))
    record = build_data_identity(config)
    sidecar = write_data_identity(tmp_path, record, compact=True, verification={})
    assert load_data_identity(sidecar) == record
    compact = json.loads(sidecar.read_bytes())
    inventory = tmp_path / compact["dataset_inventory"]["path"]
    output = Path("results.json")
    output.hardlink_to(inventory)
    rejected(corpus.main, SCORE_CORPUS + " --trusted-training-checkpoint --tokenizer tok", output, capsys)
    assert output.samefile(inventory)
    assert load_data_identity(sidecar) == record
    compact["dataset_inventory"]["path"] = "../outside.json"
    sidecar.write_text(json.dumps(compact))
    with pytest.raises(ValueError, match="unsafe inventory reference"):
        corpus.main(
            [*SCORE_CORPUS.split(), "--trusted-training-checkpoint", "--tokenizer", "tok", "--output", str(output)]
        )


@pytest.mark.parametrize(
    "cli,args,name",
    [
        (generation, "--checkpoint model.pt --study-parquet shard", "paper/tasks/generation32.json"),
        (generation, "--checkpoint model.pt --study-parquet shard", "paper/panels/corpus-v1.json"),
        (corpus, "prepare packed --parquet shard", "paper/panels/corpus-v1.json"),
        (corpus, SCORE_CORPUS, "paper/panels/corpus-v1.json"),
        (tasks, "score --bundle bundle --checkpoint model.pt", "paper/tasks/core21.json"),
    ],
)
def test_implicit_paper_inputs(cli, args, name, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    asset = Path(name)
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"paper fixture")
    Path("out").hardlink_to(asset)
    if cli is tasks:
        monkeypatch.setattr(tasks, "study_roster", lambda: asset)
    else:
        monkeypatch.setattr(cli, "release_asset", Path)
    if args == SCORE_CORPUS:
        monkeypatch.setattr(corpus, "load_panel", lambda *a, **kw: {"reference": True})
    rejected(cli.main, args, "out", capsys)
    assert asset.read_bytes() == b"paper fixture"
    assert Path("out").samefile(asset)


def test_directory_bundle_guards_payloads_and_allows_sibling_reports(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    args = SCORE_TASKS.replace("--bundle bundle", "--bundle .") + " --task selected"
    asset = Path("data/task.jsonl")
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"original task")
    output = Path("results.json")
    output.hardlink_to(asset)
    specs = [{"label": "selected", "data_path": "data/task.jsonl"}, {"label": "unused", "data_path": "results.json"}]
    monkeypatch.setattr(tasks, "read_roster", lambda p: specs)
    monkeypatch.setattr(tasks, "score", lambda *a: {"aggregates": {"n": 1}})
    rejected(tasks.main, args, output, capsys)
    assert output.samefile(asset)
    output.unlink()
    output.symlink_to("current.json")
    output.write_bytes(b"old report")
    previous = Path("previous")
    previous.hardlink_to(output.resolve())
    assert tasks.main([*args.split(), "--output", str(output)]) == 0
    assert output.is_symlink()
    assert json.loads(output.read_bytes()) == {"aggregates": {"n": 1}}
    assert previous.read_bytes() == b"old report"
    assert asset.read_bytes() == b"original task"


def test_generation_writes_through_unrelated_report_symlink(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    create(tmp_path)
    report = Path("current.json")
    Path("latest.json").symlink_to(report)
    generation.main(
        [
            "--checkpoint",
            "demo-11.weights.pt",
            "--prompt",
            "hi",
            "--capacity",
            "1",
            "--seed",
            "1",
            "--output",
            "latest.json",
        ]
    )
    assert Path("latest.json").is_symlink()
    assert json.loads(report.read_bytes())["summary"]["samples"] == 1


def test_linked_snapshot_asset_and_directory_alias(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    Path("snapshot").mkdir()
    Path("tokenizer").symlink_to("snapshot", target_is_directory=True)
    Path("blob").write_bytes(b"cached tokenizer")
    Path("snapshot/config.json").symlink_to("../blob")
    args = GENERATE + " --tokenizer tokenizer"
    rejected(generation.main, args, "blob", capsys)
    rejected(generation.main, args, "tokenizer/config.json", capsys)
    assert Path("blob").read_bytes() == b"cached tokenizer"
    assert Path("snapshot/config.json").is_symlink()


def test_new_reference_model_report_and_unrelated_existing_nested_report(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    create(tmp_path)
    monkeypatch.setattr(generation, "score_outputs", lambda *a: {})
    args = "--checkpoint demo-11.weights.pt --prompt hi --capacity 1 --seed 1 --reference-model ."
    assert generation.main([*args.split(), "--output", "results.json"]) == 0
    assert json.loads(Path("results.json").read_bytes())["summary"]["samples"] == 1
    # On the next run score_outputs hashes every top-level JSON file.
    rejected(generation.main, args, "results.json", capsys)
    Path("reports").mkdir()
    Path("reports/results.json").write_text("old report")
    assert generation.main([*args.split(), "--output", "reports/results.json"]) == 0
    assert json.loads(Path("reports/results.json").read_bytes())["summary"]["samples"] == 1


def test_corpus_report_inside_tokenizer_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(corpus, "load_tokenizer", lambda p: object())
    monkeypatch.setattr(corpus, "load_panel", lambda *a, **kw: {"reference": False})
    monkeypatch.setattr(corpus, "load_checkpoint", lambda *a, **kw: object())
    monkeypatch.setattr(corpus, "score_panel", lambda *a, **kw: {"summary": {}})
    args = [*SCORE_CORPUS.split(), "--tokenizer", ".", "--output", "results.json"]
    corpus.main(args)
    corpus.main(args)
    assert json.loads(Path("results.json").read_bytes()) == {"summary": {}}


def test_failed_serialization_preserves_report_and_cleans_staging_file(tmp_path):
    path = tmp_path / "report.json"
    path.write_bytes(b"previous report")
    with pytest.raises(ValueError, match="JSON"):
        report_io.save_result(path, {"partial": 1, "invalid": float("nan")})
    assert path.read_bytes() == b"previous report"
    assert list(tmp_path.iterdir()) == [path]
