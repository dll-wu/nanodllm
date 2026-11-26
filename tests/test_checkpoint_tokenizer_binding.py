import json
from pathlib import Path

import pytest
from nanodllm.cli import main as pretrain
from nanodllm.evaluation.checkpoint import load_checkpoint, load_tokenizer
from nanodllm.runtime.data_identity_io import load_data_identity, write_data_identity
from nanodllm.weights import export_weights, file_sha256, load_weights
from scripts.prepare_tiny_corpus import write_parquet
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast


def _write_tokenizer(path, ordinary):
    ordinary = [*ordinary, *(f"word{i}" for i in range(24))]
    special = ["[UNK]", "[BOS]", "[EOS]", "[MASK]", "[PAD]"]
    vocab = {token: i for i, token in enumerate([*special, *ordinary])}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token=special[0],
        bos_token=special[1],
        eos_token=special[2],
        mask_token=special[3],
        pad_token=special[4],
    )
    tokenizer.save_pretrained(path)
    (path / "token_bytes.json").write_text(
        json.dumps(
            {
                "format": "nanodllm.token_bytes.v1",
                "dtype": "int32",
                "vocab_size": len(vocab),
                "values": [0] * len(special) + [len(s) for s in ordinary],
                "special_token_ids": tokenizer.all_special_ids,
            }
        )
    )


def test_checkpoint_authenticates_tokenizer_sidecar_for_export_and_scoring(tmp_path):
    shard = tmp_path / "corpus.parquet"
    write_parquet(shard, ["one two " * 64] * 32, 4)
    base = Path(__file__).resolve().parents[1] / "recipes/tiny-block.json"
    for name, ordinary in [("a", ["one", "two"]), ("b", ["two", "one"])]:
        tok, run = tmp_path / f"tok-{name}", tmp_path / f"run-{name}"
        _write_tokenizer(tok, ordinary)
        config = json.loads(base.read_text())
        config["train"].update(num_steps=1, save_every=1, run_dir=str(run))
        config["data"].update(tokenizer="hf", tokenizer_path=str(tok), train_data=[str(shard)])
        filename = tmp_path / f"{name}.json"
        filename.write_text(json.dumps(config))
        pretrain(["--config", str(filename)])
    tok_a, tok_b = tmp_path / "tok-a", tmp_path / "tok-b"
    assert load_tokenizer(tok_a).encode("one") == [5]
    assert load_tokenizer(tok_b).encode("one") == [6]
    checkpoint = tmp_path / "run-a/model.pt"
    sha = file_sha256(checkpoint)
    side_b = tmp_path / "run-b/data_identity.v1.json"
    out = tmp_path / "export.weights.pt"
    for sidecar in [None, side_b]:
        with pytest.raises(ValueError, match="differ"):
            export_weights(
                checkpoint, out, expected_source_sha256=sha, tokenizer_path=tok_b, data_identity_path=sidecar
            )
        with pytest.raises(ValueError, match="differ"):
            load_checkpoint(
                checkpoint, trusted_training_checkpoint=True, tokenizer_path=tok_b, data_identity_path=sidecar
            )
        assert not out.exists()

    # A relocated compact representation must authenticate the same logical record.
    relocated = tmp_path / "compact"
    relocated.mkdir()
    record = load_data_identity(checkpoint.parent)
    compact = write_data_identity(relocated, record, compact=True, verification={})
    exported = export_weights(
        checkpoint, out, expected_source_sha256=sha, tokenizer_path=tok_a, data_identity_path=compact
    )
    load_weights(out, tokenizer_path=tok_a, expected_checkpoint_sha256=exported)
    load_checkpoint(checkpoint, trusted_training_checkpoint=True, tokenizer_path=tok_a)
