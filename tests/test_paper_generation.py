import json
import math

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from nanodllm.evaluation.block_checkpoint import load_block
from nanodllm.generation.cli import main
from nanodllm.generation.panel import reconstruct, study_panel
from nanodllm.generation.reference import encode_continuation, score_outputs
from nanodllm.generation.sampling import Decoder, generate_panel, read_prompts
from nanodllm.runtime.block_recipe_sampling import generate
from nanodllm.runtime.core_task_data import json_digest
from nanodllm.runtime.data import ByteTokenizer
from scripts.tasks_demo import create


@pytest.mark.parametrize(
    "policy,mode,k",
    [
        ("left-to-right", "left_to_right", 1),
        ("sample-keep", "confidence", 4),
        ("sample-redraw", "confidence_resample", 2),
    ],
)
def test_sampler_equivalence(tmp_path, policy, mode, k):
    create(tmp_path)
    model = load_block(tmp_path / "demo-11.weights.pt", None, "cpu", generation=True)
    prompts = read_prompts([{"id": "sample", "text": "Dial?"}], model.tokenizer)
    decoder = Decoder(k, policy, 1, 0.9, 12, 5)
    result = generate_panel(model, prompts, decoder, [71], "cpu")["samples"][0]
    native = generate(
        model.model,
        model.tokenizer,
        prompts[0],
        dict(k=k, policy=mode, temperature=1, top_p=0.9),
        71,
        capacity=12,
        block_length=5,
    )
    assert result["text"] == native["text"] and result["work"] == native["work"]
    assert result["repeated_4_fraction"] == native["repeated_4_fraction"]
    assert result["tokens_per_forward"] == native["content_tokens"] / native["work"]["model_forwards"]


def test_eos_accounting_and_reject_oversized_prompt(tmp_path):
    create(tmp_path)
    model = load_block(tmp_path / "demo-11.weights.pt", None, "cpu", generation=True)
    eos = model.tokenizer.get_eos_token_id()

    def force_eos(module, inputs, out):
        out.zero_()
        out[..., eos] = 100
        return out

    with model.model.lm_head.register_forward_hook(force_eos):
        p = read_prompts([dict(id="only", text="")], model.tokenizer)
        result = generate_panel(model, p, Decoder(1, "left-to-right", 0, 1, 8, 4), [2], "cpu")
    row = result["samples"][0]
    assert row["eos_present"] and not row["capacity_hit"] and row["content_tokens"] == 0
    assert row["work"]["clean_commit_forwards"] == 1 and row["work"]["active_forwards"] == 4
    assert result["summary"]["mean_repeat4"] is None
    with pytest.raises(ValueError, match="trained context"):
        generate_panel(model, p, Decoder(capacity=128), [2], "cpu")
    with pytest.raises(ValueError, match="BOS"):
        read_prompts([dict(id="bad", prefix_ids=[0, 1])], model.tokenizer)


def test_local_panel_reconstruction(tmp_path):
    path = tmp_path / "docs.parquet"
    pq.write_table(pa.table({"text": [" first", "second ", " third "]}), path, row_group_size=1)
    ref = [dict(id="third", prefix_ids=[256, 116, 104]), dict(id="first", prefix_ids=[256, 102, 105])]
    panel = dict(
        rows=[dict(row=2, id="third"), dict(row=0, id="first")], prefix_tokens=3, prompts_sha256=json_digest(ref)
    )
    assert reconstruct(pq.ParquetFile(path), ByteTokenizer(), panel) == ref
    published = study_panel()
    assert len(published["rows"]) == 32
    assert sum(r["length_stratum"] == "short" for r in published["rows"]) == 16


@pytest.mark.parametrize("family", ["gpt2", "gemma4"])
def test_offline_reference_scores_only_target_and_pools(tmp_path, family):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel(vocab={"[UNK]": 0, "[BOS]": 1, "[PAD]": 2, "lamp": 3, "ok": 4}, unk_token="[UNK]")
        ),
        bos_token="[BOS]",
        pad_token="[PAD]",
        unk_token="[UNK]",
    )
    tokenizer.save_pretrained(tmp_path)
    torch.manual_seed(5)
    if family == "gemma4":
        pytest.importorskip("transformers.models.gemma4")
        from transformers import Gemma4Config, Gemma4ForConditionalGeneration, Gemma4TextConfig

        config = Gemma4TextConfig(
            vocab_size=5,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            global_head_dim=8,
            hidden_size_per_layer_input=0,
            max_position_embeddings=32,
            layer_types=["sliding_attention", "full_attention"],
            final_logit_softcapping=30.0,
            use_bidirectional_attention="vision",
            pad_token_id=2,
            bos_token_id=1,
        )
        model = Gemma4ForConditionalGeneration(
            Gemma4Config(text_config=config, vision_config=None, audio_config=None)
        ).eval()
    else:
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=5,
                n_positions=16,
                n_embd=8,
                n_layer=1,
                n_head=1,
                bos_token_id=1,
                eos_token_id=None,
                pad_token_id=2,
            )
        ).eval()
    model.save_pretrained(tmp_path)
    rows = [dict(prompt_id="a", text="ok"), dict(prompt_id="a", text="")]
    encoded = encode_continuation(tokenizer, "lamp", "ok", max_length=4)
    assert encoded["input_ids"] == [1, 3, 4] and encoded["target_start"] == 2
    score = score_outputs(rows, {"a": "lamp"}, tmp_path, "cpu")
    with torch.no_grad():
        nll = -model(torch.tensor([[1, 3, 4]])).logits[0, 1].log_softmax(-1)[4].item()
    assert score["ppl"] == pytest.approx(math.exp(nll))
    assert score["bpb"] == pytest.approx(nll / (2 * math.log(2)))
    assert (score["tokens"], score["bytes"]) == (1, 2) and rows[1]["reference"]["perplexity"] is None


def test_generation_cli(tmp_path):
    create(tmp_path)
    path = tmp_path / "result.json"
    assert (
        main(
            [
                "--checkpoint",
                str(tmp_path / "demo-11.weights.pt"),
                "--prompts",
                str(tmp_path / "prompts.jsonl"),
                "--capacity",
                "12",
                "--output",
                str(path),
            ]
        )
        == 0
    )
    report = json.loads(path.read_bytes())
    assert report["summary"]["samples"] == 4 and report["decoder"]["block_length"] == 8
    assert "The lamp blinked twice." not in path.read_text()


def test_literal_prompt_stdout_uses_available_context(tmp_path, capsys, monkeypatch):
    def unexpected_asset(name):
        pytest.fail(f"custom generation requested {name}")

    monkeypatch.setattr("nanodllm.generation.panel.release_asset", unexpected_asset)
    create(tmp_path)
    args = ["--checkpoint", str(tmp_path / "demo-11.weights.pt"), "--seed", "7", "--prompt"]
    main([*args, "a" * 125])
    report = json.loads(capsys.readouterr().out)
    assert report["decoder"]["capacity"] == 2
    assert report["summary"]["samples"] == 1
    assert isinstance(report["samples"][0]["text"], str)
    with pytest.raises(ValueError, match="no room"):
        main([*args, "a" * 127])
