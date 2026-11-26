import json
import math
from dataclasses import asdict

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from nanodllm.evaluation.block import common_prediction, conditional_native_nelbo
from nanodllm.evaluation.bos import corruption_plan
from nanodllm.evaluation.checkpoint import load_checkpoint, tokenizer_identity
from nanodllm.evaluation.cli import main
from nanodllm.evaluation.compare import paired_comparison
from nanodllm.evaluation.panels import (
    FULL_SEEDS,
    block_custom,
    bos_from_packed,
    load_panel,
    pack_documents,
    reference_manifest,
)
from nanodllm.runtime.config import ModelConfig
from nanodllm.runtime.data import ByteTokenizer
from nanodllm.runtime.model import DiffusionConfig, DiffusionTransformer
from nanodllm.weights import export_weights, file_sha256


def weights(tmp_path, family, *, block_size=32):
    tok = ByteTokenizer()
    ar = family == "ar"
    size = block_size if family == "block" else 0
    cfg = DiffusionConfig(
        sequence_len=320,
        vocab_size=tok.get_vocab_size(),
        n_layer=1,
        n_head=2,
        n_embd=16,
        mask_token_id=tok.get_mask_token_id(),
        causal=ar,
        bd3lm_block_size=size,
    )
    settings = ModelConfig(depth=1, n_head=2, n_embd=16, max_seq_len=320, causal=ar, bd3lm_block_size=size)
    model = DiffusionTransformer(cfg, compute_dtype=torch.float32, pad_vocab_to=settings.pad_vocab_to)
    # Uniform logits give an independent, closed-form CE for each scored token.
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    trainer = tmp_path / (family + ".pt")
    torch.save(
        dict(
            config=asdict(cfg),
            train_config=dict(
                model=asdict(settings),
                noise={"type": "bd3lm" if size else "mdlm"},
                data={"tokenizer": "byte"},
                train={},
            ),
            model_state_dict=model.state_dict(),
            step=1,
            train_tokens=128,
            train_valid_tokens=128,
        ),
        trainer,
    )
    path = tmp_path / (family + ".weights.pt")
    export_weights(trainer, path, expected_source_sha256=file_sha256(trainer))
    return path, trainer


def test_row_packing_bos_and_noise():
    tok = ByteTokenizer()
    packed = pack_documents([["ab", "cdefghij", "k"]], tok, 2, 8)
    b, e = tok.get_bos_token_id(), tok.get_eos_token_id()
    assert packed["x_0"].tolist() == [[b, 97, 98, e, b, 99, 100, 101], [102, 103, 104, 105, 106, e, b, 107]]
    assert packed["position_ids"].tolist() == [[0, 1, 2, 3, 0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 0, 1]]
    panel = bos_from_packed(packed, tok)
    assert [(r["source_row"], r["source_start"], r["source_end"]) for r in panel["examples"]] == [
        (0, 0, 4),
        (0, 4, 8),
        (1, 6, 8),
    ]
    assert panel["counts"]["content_bytes"] == 6
    t, u, receipt = corruption_plan(panel, FULL_SEEDS[0])
    assert sorted(t.tolist()) == pytest.approx([1 / 6, 0.5, 5 / 6])
    assert u.shape == (3, 8)
    assert receipt["mask_uniforms_sha256"] == "f2877f4446b2364968ab9efb9188d215555a1bc7c11c3a8b3cce3df977a06912"


@pytest.mark.parametrize(
    "kind,family", [("packed", "ar"), ("packed", "mdlm"), ("bos", "ar"), ("bos", "mdlm"), ("dev32", "block")]
)
def test_cli_portable_model_replay(tmp_path, kind, family, capsys):
    shard = tmp_path / "validation.parquet"
    docs = ["abc", "defghijk", "lmnop", "qrstuvw", "xyz"] if kind != "dev32" else ["a" * 222, "b" * 703]
    pq.write_table(pa.table({"text": docs}), shard, row_group_size=1)
    path, trainer = weights(tmp_path, family)
    panel = tmp_path / "panel.pt"
    opts = ["--count", "2", "--width", "8"] if kind != "dev32" else ["--per-stratum", "1"]
    main(["prepare", kind, "--custom", "--parquet", str(shard), "--output", str(panel), *opts])
    moved = tmp_path / "relocated.pt"
    panel.rename(moved)
    out = tmp_path / "report.json"
    main(
        [
            "score",
            "--panel",
            str(moved),
            "--checkpoint",
            str(path),
            "--batch-size",
            "4" if kind == "dev32" else "1",
            "--output",
            str(out),
        ]
    )
    report = json.loads(out.read_text())
    assert report["kind"] == kind and not report["reference"] and report["checkpoint_sha256"] == file_sha256(path)
    assert "prefix_ids" not in out.read_text() and str(tmp_path) not in out.read_text()
    compare = tmp_path / "paired.json"
    main(["compare", "--reference", str(out), "--candidate", str(out), "--output", str(compare)])
    assert all(m["ci95_bpb"] == [0.0, 0.0] for m in json.loads(compare.read_text())["metrics"].values())
    if family == "ar":
        assert report["seeds"] == []
        assert report["summary"]["ar/content"]["bpb"] == pytest.approx(math.log2(ByteTokenizer().get_vocab_size()))
    if family == "mdlm":
        scores = report["summary"]
        assert report["seeds"] == list(FULL_SEEDS)
        assert scores["nelbo/mean_plans"]["bpb"] == pytest.approx(
            sum(scores[f"nelbo/{s}"]["bpb"] for s in FULL_SEEDS) / 4
        )
    if family == "block":
        # Short row has 31 body targets and EOS. Long row has 32 body targets.
        assert report["summary"]["common/block_end"]["bytes"] == 63
        assert report["summary"]["native/mean_eval_seeds"]["bytes"] == 31 + 128
    with pytest.raises(ValueError, match="export"):
        load_checkpoint(trainer)
    trusted = load_checkpoint(trainer, trusted_training_checkpoint=True)
    assert trusted.tokenizer_sha256 == tokenizer_identity(ByteTokenizer())
    replay = torch.load(moved, weights_only=True)
    replay["panel_sha256"] = "0" * 64
    torch.save(replay, moved)
    with pytest.raises(ValueError, match="replay hash"):
        load_panel(moved, ByteTokenizer())
    capsys.readouterr()


def test_block_mask_canvas_and_eos_accounting(tmp_path):
    path, _ = weights(tmp_path, "block")
    model = load_checkpoint(path).model
    policy = dict(topology="causal_prefix", block_length=32, clean_stream_mode="causal")
    row = dict(id="fixture", prefix_ids=[256] + [65] * 191, gold_ids=[66] * 31 + [258], gold_token_bytes=[1] * 31 + [0])
    inputs = []

    def capture(module, args):
        inputs.append(args[0].detach().clone())

    with model.register_forward_pre_hook(capture):
        common_prediction(model, row, policy={**policy, "common_canvas": "legacy"})
        common_prediction(model, row, policy={**policy, "common_canvas": "block_end"})
    assert inputs[0].shape[-1] == 193 and inputs[32].shape[-1] == 224
    assert torch.equal(inputs[31], inputs[-1])
    assert inputs[32][0, 192:].eq(model.config.mask_token_id).all()
    estimate = conditional_native_nelbo(model, row, policy=policy, seed=2026110201, draws=16, batch_size=4)
    assert estimate["bytes"] == 31 and estimate["nats"] > estimate["content_nats"]
    assert estimate["nats"] == pytest.approx(205.5041389465332)


def test_custom_native_block_panel(tmp_path, monkeypatch):
    def unexpected_asset(name):
        pytest.fail(f"custom block scoring requested {name}")

    monkeypatch.setattr("nanodllm.evaluation.panels.release_asset", unexpected_asset)
    path, _ = weights(tmp_path, "block", block_size=8)
    shard = tmp_path / "local.parquet"
    pq.write_table(pa.table({"text": ["short", "abcdefgh", "abcdefgh", "0123456789abcdef"]}), shard)
    panel = tmp_path / "block.pt"
    main(["prepare", "block", "--parquet", str(shard), "--block-size", "8", "--count", "2", "--output", str(panel)])
    data = load_panel(panel, ByteTokenizer())["data"]
    assert data["block_size"] == 8
    assert [r["raw_document_index"] for r in data["rows"]] == [1, 3]
    assert data["rows"][0]["prefix_ids"] == [256, *b"abcdefg"]
    assert data["rows"][0]["gold_ids"] == [ord("h"), 258]
    out = tmp_path / "score.json"
    main(["score", "--checkpoint", str(path), "--panel", str(panel), "--output", str(out)])
    result = json.loads(out.read_bytes())
    assert result["reference"] is False
    for canvas in ("legacy", "block_end"):
        common = result["summary"][f"common/{canvas}"]
        assert common["bytes"] == 9
        assert common["bpb"] == pytest.approx(math.log2(ByteTokenizer().get_vocab_size()))
    assert result["summary"]["native/mean_eval_seeds"]["bytes"] == 9
    main(["prepare", "block", "--parquet", str(shard), "--block-size", "4", "--output", str(panel)])
    with pytest.raises(ValueError, match="native BL4"):
        main(["score", "--checkpoint", str(path), "--panel", str(panel), "--output", str(out)])


def test_custom_disjointness_and_reference_metadata():
    docs = [f"document {i:02d} " + ("x" * 220 if i % 2 else "z" * 705) for i in range(40)]
    tok = ByteTokenizer()
    dev = block_custom([docs], tok, per_stratum=16, skip_per_stratum=0)
    held = block_custom([docs], tok, per_stratum=2, skip_per_stratum=16)
    assert len(dev) == 32 and len(held) == 4
    assert {tuple(r["prefix_ids"][1:65]) for r in dev}.isdisjoint(tuple(r["prefix_ids"][1:65]) for r in held)
    m = reference_manifest()
    for name, count in (("dev32", 32), ("holdout256", 256)):
        rows = m["block_panels"][name]["rows"]
        assert len(rows) == len({r[0] for r in rows}) == count
        assert sum(r[2] == "short" for r in rows) == count // 2
    assert set(r[1] for r in m["block_panels"]["dev32"]["rows"]).isdisjoint(
        r[1] for r in m["block_panels"]["holdout256"]["rows"]
    )


def test_reject_unpaired_report():
    report = dict(
        schema="nanodllm.paper.corpus-scores.v1",
        kind="bos",
        reference=False,
        panel_sha256="a",
        tokenizer_sha256="b",
        device="cpu",
        device_batch=4,
        dtype="float32",
        depth=1,
        seeds=[],
        plans={},
        checkpoint_sha256="weights",
        rows=[dict(id=0, stratum="bos", metrics={"ar/content": [math.log(2), 1]})],
        summary={"ar/content": {"nats": math.log(2), "bytes": 1, "bpb": 1}},
    )
    with pytest.raises(ValueError, match="panel_sha256"):
        paired_comparison(report, {**report, "panel_sha256": "other"})


def test_reference_address_replay_in_original_order(tmp_path):
    from nanodllm.evaluation.panels import block_prompt, block_reference, block_root
    from nanodllm.runtime.data import build_token_bytes

    texts = ["synthetic " + str(i) + " " + "a" * (230 + i) for i in range(4)]
    path = tmp_path / "synthetic.parquet"
    pq.write_table(pa.table({"text": texts}), path, row_group_size=2)
    tok = ByteTokenizer()
    byte_table = build_token_bytes(tok).tolist()
    selected = [block_prompt(texts[i], i, tok, byte_table) for i in (3, 0, 2)]
    addresses = [
        [r["raw_document_index"], r["id"][6:], r["length_stratum"], r["source_remaining_tokens"]] for r in selected
    ]
    manifest = {"rows": addresses, "targets_sha256": block_root(selected)}
    with pq.ParquetFile(path) as source:
        assert block_reference(source, tok, manifest) == selected
        with pytest.raises(ValueError, match="targets differ"):
            block_reference(source, tok, {**manifest, "targets_sha256": "0" * 64})
        addresses[0][0] = 1
        with pytest.raises(ValueError, match="document or tokenization"):
            block_reference(source, tok, manifest)


@pytest.mark.parametrize(
    "clean_mode,expected",
    [
        ("causal", (178.53825092315674, 178.54800939559937, 206.14352416992188)),
        ("block_causal", (178.57272481918335, 178.58008527755737, 206.16643524169922)),
    ],
)
def test_nonuniform_native_checkpoint_parity(clean_mode, expected):
    # Pinned float32 CPU values for a two-layer model with nonuniform logits.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(91)
        cfg = DiffusionConfig(
            sequence_len=320, vocab_size=260, n_layer=2, n_head=2, n_embd=16, mask_token_id=257, bd3lm_block_size=32
        )
        model = DiffusionTransformer(cfg, compute_dtype=torch.float32).eval()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.normal_(std=0.12)
    policy = dict(topology="causal_prefix", block_length=32, clean_stream_mode=clean_mode)
    row = dict(
        id="fixture",
        prefix_ids=[256] + ([8, 23, 51, 97] * 48)[:191],
        gold_ids=[66, 5, 23, 81] * 8,
        gold_token_bytes=[1] * 32,
    )
    actual = [
        common_prediction(model, row, policy={**policy, "common_canvas": canvas})["aggregate"]["content_nll_nats"]
        for canvas in ("legacy", "block_end")
    ]
    actual.append(conditional_native_nelbo(model, row, policy=policy, seed=2026110201, draws=16, batch_size=4)["nats"])
    assert actual == pytest.approx(expected, abs=8e-5, rel=0)


def test_prefix_exclusions_guard_replay():
    from nanodllm.evaluation.panels import verify_reference_exclusions
    from nanodllm.runtime.replication_eval_support import digest

    tok = ByteTokenizer()
    packed = pack_documents([["a" * 80]], tok, 1, 70)
    first = digest([97] * 64)
    dev = {"prefix_ids": [256] + [98] * 191}
    held = {"prefix_ids": [256] + [99] * 191}
    excl = dict(
        body_tokens=64,
        bos_unique_prefixes=1,
        bos_prefix_set_sha256=digest([first]),
        development_predictive_and_generation_prefix_sha256=[digest([98] * 64)],
    )
    verify_reference_exclusions("dev32", [dev], packed, excl)
    verify_reference_exclusions("holdout256", [held], packed, excl)
    with pytest.raises(ValueError, match="earlier prompt"):
        verify_reference_exclusions("holdout256", [dev], packed, excl)
    with pytest.raises(ValueError, match="corpus prefix"):
        verify_reference_exclusions("holdout256", [{"prefix_ids": [256] + [97] * 191}], packed, excl)
