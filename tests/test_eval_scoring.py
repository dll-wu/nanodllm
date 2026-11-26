import json
import math

import pytest
import torch
from nanodllm import eval as public_eval
from nanodllm.runtime import data, eval_rank_suffix
from nanodllm.runtime.data import ByteTokenizer, DLMDataBatch


def _use_validation_rows(monkeypatch, tokenizer, rows):
    ids = torch.tensor(rows)
    attention = ids != tokenizer.PAD_ID
    batch = DLMDataBatch(x_0=ids, loss_mask=attention & (ids != tokenizer.BOS_ID), attention_mask=attention)

    def loader(data_config, tokenizer, batch_size, seq_len, split, device):
        assert split == "val"
        assert batch.x_0.shape == (batch_size, seq_len)
        return iter([batch])

    monkeypatch.setattr(data, "make_dataloader", loader)


def test_conditional_generation_keeps_raw_rows_and_scores_eos_truncated_content(monkeypatch, tmp_path):
    tokenizer = ByteTokenizer()
    q, r = ord("q"), ord("r")
    rows = [[ord("p"), q, q, q, q], [q, r, r, r, r]]
    _use_validation_rows(monkeypatch, tokenizer, rows)
    byte_lengths = torch.ones(tokenizer.VOCAB_SIZE)
    byte_lengths[q] = 3
    monkeypatch.setattr(data, "build_token_bytes", lambda _tok: byte_lengths)
    suffixes = [[q, tokenizer.EOS_ID, tokenizer.MASK_ID, q], [q, r, r, r]]

    def sample(_model, prefixes, continuation_len):
        return torch.cat((prefixes, torch.tensor(suffixes)), dim=1)

    class MaskAll:
        def corrupt(self, tokens, t, maskable):
            return tokens.masked_fill(maskable, tokenizer.MASK_ID), maskable.float()

    def uniform_logits(tokens, attention_mask):
        return torch.zeros(*tokens.shape, tokenizer.VOCAB_SIZE)

    metrics = public_eval.evaluate_conditional_generation_metrics(
        uniform_logits,
        tokenizer,
        None,
        MaskAll(),
        sample,
        batch_size=2,
        seq_len=5,
        device="cpu",
        gen_samples=2,
        continuation_len=4,
        prefix_fracs=(0.25,),
        run_dir=str(tmp_path),
    )
    records = [
        json.loads(line) for line in (tmp_path / "conditional_generation_samples.jsonl").read_text().splitlines()
    ]
    assert [row["prefix_token_ids"] for row in records] == [[ord("p")], [q]]
    assert [row["continuation_token_ids"] for row in records] == suffixes
    assert [row["continuation_text"] for row in records] == ["q", "qrrr"]
    assert [row["has_eos"] for row in records] == [True, False]
    assert metrics["conditional_gen_p25_mean_continuation_len"] == 3
    assert metrics["conditional_gen_p25_blocked_token_rate"] == 1 / 8
    assert metrics["conditional_gen_p25_content_bytes"] == 9
    assert metrics["conditional_gen_p25_self_ce_tokens"] == 5
    assert metrics["conditional_gen_p25_self_ce"] == pytest.approx(math.log(tokenizer.VOCAB_SIZE))


@pytest.mark.parametrize(
    ("suffix", "ar_rank", "masked_rank", "content_count"),
    [([ord("a"), ord("b"), ByteTokenizer.EOS_ID], 1.5, 2.5, 2), (list(b"cbb"), 7 / 3, 7 / 3, 3)],
)
def test_conditional_rank_on_one_sequence_with_frozen_frequency_strata(
    monkeypatch, suffix, ar_rank, masked_rank, content_count
):
    tokenizer = ByteTokenizer()
    _use_validation_rows(monkeypatch, tokenizer, [[tokenizer.BOS_ID, ord("p"), ord("q"), *suffix]])
    strata = {ord("a"): 2, ord("b"): 3, ord("c"): 7}
    monkeypatch.setattr(eval_rank_suffix, "_conditional_rank_frequency_deciles", lambda *args: (strata, 17))

    def logits(tokens, attention_mask):
        scores = torch.zeros(*tokens.shape, tokenizer.VOCAB_SIZE)
        scores[..., ord("a")] = 3
        scores[..., ord("b")] = 2
        scores[..., ord("c")] = torch.where(tokens == tokenizer.MASK_ID, 4, 1)
        return scores

    args = dict(model=logits, tokenizer=tokenizer, data_config=None, device="cpu", batches=1, batch_size=1, seq_len=6)
    ar = public_eval.evaluate_ar_next_token_conditional_rank_suffix(**args)
    assert ar["ar_next_token_conditional_rank_suffix_gold_rank_mean"] == ar_rank
    masked = public_eval.evaluate_conditional_rank_suffix(
        **args,
        include_leave_one_out=True,
        include_stratified_matched=True,
        stratified_target_per_decile=3,
    )
    assert masked["conditional_rank_freq_batches"] == 17
    assert masked["conditional_rank_decile_3_content_token_count"] == suffix.count(ord("b"))
    for context in ("", "loo_", "stratified_prefix_", "stratified_partial25_", "stratified_loo_"):
        assert masked[f"conditional_rank_{context}suffix_content_token_count"] == content_count
        assert masked[f"conditional_rank_{context}suffix_gold_rank_mean"] == masked_rank
