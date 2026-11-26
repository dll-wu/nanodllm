import json
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from nanodllm.data import ByteTokenizer
from nanodllm.eval import evaluate_generation_metrics
from nanodllm.noise import MaskUniformNoise, MDLMNoise, UniformNoise
from nanodllm.runtime.eval import _self_reconstruction_ce
from nanodllm.sampling import (
    block_special_logits,
    sample_bd3lm,
    sample_block_causal_pilot,
    sample_fhs,
    sample_mask_uniform,
    sample_mdlm,
    sampling_logit_bias,
)


def test_generation_blocking_does_not_block_eos():
    tok = ByteTokenizer()
    logits = torch.zeros(1, 1, tok.get_vocab_size())
    logits[..., tok.get_eos_token_id()] = 10.0
    block_special_logits(logits, tok)
    assert logits[..., tok.get_eos_token_id()].item() > 0
    assert logits[..., tok.get_mask_token_id()].item() < -1e8
    assert logits[..., tok.get_bos_token_id()].item() < -1e8
    assert logits[..., tok.get_pad_token_id()].item() < -1e8


class _TinyGenerationModel(torch.nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.vocab_size = vocab_size

    def forward(self, z_t, attention_mask=None, position_ids=None):
        logits = torch.zeros(*z_t.shape, self.vocab_size, device=z_t.device)
        logits.scatter_(-1, z_t.clamp_max(self.vocab_size - 1).unsqueeze(-1), 4.0)
        return logits + self.anchor


class _NoiseLevelRecordingModel(_TinyGenerationModel):
    def __init__(self, vocab_size: int):
        super().__init__(vocab_size)
        self.config = SimpleNamespace(time_conditioning=True)
        self.noise_levels = []

    def forward(self, z_t, attention_mask=None, position_ids=None, noise_level=None):
        self.noise_levels.append(noise_level)
        return super().forward(z_t, attention_mask=attention_mask, position_ids=position_ids)


class _AttentionRecordingModel(_TinyGenerationModel):
    def __init__(self, vocab_size: int):
        super().__init__(vocab_size)
        self.attention_masks = []

    def forward(self, z_t, attention_mask=None, position_ids=None):
        if attention_mask is not None:
            self.attention_masks.append(attention_mask.detach().clone())
        return super().forward(z_t, attention_mask=attention_mask, position_ids=position_ids)


class _LengthPolicyModel(_TinyGenerationModel):
    def __init__(self, vocab_size: int, predicted_position: int):
        super().__init__(vocab_size)
        self.predicted_position = int(predicted_position)

    def forward(self, z_t, attention_mask=None, position_ids=None, return_length_logits=False):
        logits = super().forward(z_t, attention_mask=attention_mask, position_ids=position_ids)
        if not return_length_logits:
            return logits
        length_logits = torch.zeros(z_t.shape[0], z_t.shape[1], device=z_t.device)
        length_logits[:, self.predicted_position] = 5.0
        return logits, length_logits


class _TargetLengthEosModel(_TinyGenerationModel):
    def __init__(self, vocab_size: int, eos_id: int, fill_id: int):
        super().__init__(vocab_size)
        self.config = SimpleNamespace(target_length_conditioning=True)
        self.eos_id = int(eos_id)
        self.fill_id = int(fill_id)
        self.target_length_calls = []

    def forward(self, z_t, attention_mask=None, position_ids=None, target_lengths=None):
        del attention_mask, position_ids
        if target_lengths is None:
            raise AssertionError("target_lengths were not passed to the sampler model call")
        target_lengths = target_lengths.to(device=z_t.device, dtype=torch.long).reshape(-1)
        self.target_length_calls.append(target_lengths.detach().cpu().clone())
        logits = torch.full((*z_t.shape, self.vocab_size), -8.0, device=z_t.device)
        logits[..., self.fill_id] = 8.0
        target_positions = target_lengths[:, None] - 1
        positions = torch.arange(z_t.shape[1], device=z_t.device).view(1, -1)
        rows, cols = (positions == target_positions).nonzero(as_tuple=True)
        logits[rows, cols, self.eos_id] = 16.0
        return logits + self.anchor


def test_generation_metrics_do_not_require_judge_model(tmp_path):
    tok = ByteTokenizer()
    model = _TinyGenerationModel(tok.get_vocab_size())
    noise = MDLMNoise(tok.get_mask_token_id())

    def sample_fn(_model, n, _length):
        row = torch.tensor([ord("h"), ord("i"), tok.get_eos_token_id(), ord("x")], dtype=torch.long)
        return row.repeat(n, 1)

    metrics = evaluate_generation_metrics(
        model,
        tok,
        noise,
        sample_fn,
        batch_size=2,
        run_dir=str(tmp_path),
        device=torch.device("cpu"),
        gen_samples=4,
        gen_len=4,
        log_samples=2,
        score_mode="self",
        self_score_t=0.5,
        step=7,
        block_size=2,
    )

    assert metrics["gen_score_mode"] == "self"
    assert metrics["gen_samples"] == 4
    assert metrics["gen_eos_rate"] == 1.0
    assert metrics["gen_eos_position_0002_count"] == 4
    assert metrics["gen_eos_position_0002_rate"] == 1.0
    assert metrics["gen_eos_block_count"] == 2
    assert metrics["gen_eos_block_1_count"] == 4
    assert metrics["gen_eos_block_1_rate"] == 1.0
    assert metrics["gen_exact_duplicate_rate"] == 0.75
    assert metrics["gen_max_token_id"] == ord("h")
    assert metrics["gen_max_token_text"] == "h"
    assert metrics["gen_max_token_fraction"] == 0.5
    assert metrics["gen_self_ce_tokens"] > 0
    assert (tmp_path / "gen_samples.log").exists()


def test_self_reconstruction_ce_passes_time_conditioning_noise_level():
    tok = ByteTokenizer()
    model = _NoiseLevelRecordingModel(tok.get_vocab_size())
    noise = UniformNoise(tok.get_mask_token_id(), tok.get_vocab_size())
    tokens = torch.tensor([[ord("h"), ord("i"), tok.get_eos_token_id()]], dtype=torch.long)

    _ce, count = _self_reconstruction_ce(
        model,
        tok,
        noise,
        tokens,
        torch.device("cpu"),
        t_val=0.6,
    )

    assert count == 2
    assert model.noise_levels[-1] == 0.6


def test_sample_mdlm_can_return_trace():
    tok = ByteTokenizer()
    model = _TinyGenerationModel(tok.get_vocab_size())
    noise = MDLMNoise(tok.get_mask_token_id())

    tokens, trace = sample_mdlm(
        model,
        noise,
        tok,
        n=2,
        length=8,
        n_steps=4,
        scoring_mode="margin_jitter",
        jitter_scale=0.01,
        return_trace=True,
    )

    assert tokens.shape == (2, 8)
    assert trace["steps"] > 0
    assert trace["candidate_count"] > 0
    assert trace["commit_count"] > 0
    assert trace["final_mask_count"] == 0


def test_sample_mdlm_applies_temporary_logit_bias():
    tok = ByteTokenizer()
    model = _TinyGenerationModel(tok.get_vocab_size())
    noise = MDLMNoise(tok.get_mask_token_id())
    logit_bias = torch.zeros(tok.get_vocab_size())
    logit_bias[ord("z")] = 10.0

    with sampling_logit_bias(logit_bias):
        tokens = sample_mdlm(
            model,
            noise,
            tok,
            n=1,
            length=4,
            n_steps=1,
            scoring_mode="maxprob",
            return_trace=False,
        )

    assert tokens.tolist() == [[ord("z")] * 4]


def test_sample_fhs_argmax_length_policy_places_eos_and_traces_positions():
    tok = ByteTokenizer()
    model = _LengthPolicyModel(tok.get_vocab_size(), predicted_position=2)
    noise = MDLMNoise(tok.get_mask_token_id())

    tokens, trace = sample_fhs(
        model,
        noise,
        tok,
        n=2,
        length=6,
        n_steps=2,
        token_selection="argmax",
        length_policy="argmax_eos",
        return_trace=True,
    )

    assert torch.equal(tokens[:, 2], torch.full((2,), tok.get_eos_token_id()))
    assert trace["length_policy_applied"] == 1
    assert trace["length_policy_pre_eos_count"] == 0
    assert trace["length_policy_post_eos_count"] == 2
    assert trace["length_policy_predicted_position_0002_count"] == 2
    assert trace["length_policy_post_eos_position_0002_count"] == 2


def test_sample_fhs_records_target_length_conditioning_trace():
    tok = ByteTokenizer()
    model = _TargetLengthEosModel(tok.get_vocab_size(), tok.get_eos_token_id(), ord("a"))
    noise = MDLMNoise(tok.get_mask_token_id())
    target_lengths = torch.tensor([2, 5], dtype=torch.long)

    tokens, trace = sample_fhs(
        model,
        noise,
        tok,
        n=2,
        length=6,
        n_steps=2,
        token_selection="argmax",
        length_policy="none",
        return_trace=True,
        target_lengths=target_lengths,
    )

    assert torch.equal(tokens[0, 1], torch.tensor(tok.get_eos_token_id()))
    assert torch.equal(tokens[1, 4], torch.tensor(tok.get_eos_token_id()))
    assert model.target_length_calls
    assert torch.equal(model.target_length_calls[0], target_lengths)
    assert trace["target_length_conditioning_applied"] == 1
    assert trace["target_length_sample_count"] == 2
    assert trace["target_length_values"] == [2, 5]
    assert trace["target_length_pre_policy_eos_positions"] == [1, 4]
    assert trace["target_length_post_policy_eos_positions"] == [1, 4]
    assert trace["target_length_forced_eos_count"] == 0
    assert trace["target_length_no_forced_eos_success_count"] == 2
    assert trace["target_length_pre_policy_eos_abs_error_sum"] == 0


def test_sample_bd3lm_can_return_trace():
    tok = ByteTokenizer()
    model = _TinyGenerationModel(tok.get_vocab_size())
    noise = MDLMNoise(tok.get_mask_token_id())

    tokens, trace = sample_bd3lm(
        model,
        noise,
        tok,
        n=2,
        length=8,
        n_steps=2,
        block_size=4,
        scoring_mode="margin_jitter",
        jitter_scale=0.01,
        return_trace=True,
    )

    assert tokens.shape == (2, 8)
    assert trace["steps"] > 0
    assert trace["candidate_count"] > 0
    assert trace["commit_count"] > 0
    assert trace["final_mask_count"] == 0


def test_sample_block_causal_pilot_uses_block_causal_attention_and_trace():
    tok = ByteTokenizer()
    model = _AttentionRecordingModel(tok.get_vocab_size())
    noise = MDLMNoise(tok.get_mask_token_id())

    tokens, trace = sample_block_causal_pilot(
        model,
        noise,
        tok,
        n=2,
        length=8,
        n_steps=2,
        block_size=4,
        scoring_mode="margin_jitter",
        jitter_scale=0.01,
        return_trace=True,
    )

    assert tokens.shape == (2, 8)
    assert trace["steps"] > 0
    assert trace["candidate_count"] > 0
    assert trace["commit_count"] > 0
    assert trace["final_mask_count"] == 0
    assert trace["num_blocks"] == 2
    assert trace["nfe_per_block"] == 2
    assert model.attention_masks
    attn = model.attention_masks[0]
    assert attn.shape == (2, 8, 8)
    assert bool(attn[0, 0, 3])
    assert not bool(attn[0, 0, 4])
    assert bool(attn[0, 4, 0])


def test_sample_mask_uniform_can_return_trace():
    tok = ByteTokenizer()
    model = _TinyGenerationModel(tok.get_vocab_size())
    noise = MaskUniformNoise(tok.get_mask_token_id(), tok.get_vocab_size(), uniform_mix_prob=0.5)

    tokens, trace = sample_mask_uniform(
        model,
        noise,
        tok,
        n=2,
        length=8,
        n_steps=4,
        scoring_mode="margin_jitter",
        jitter_scale=0.01,
        return_trace=True,
    )

    assert tokens.shape == (2, 8)
    assert not (tokens == tok.get_mask_token_id()).any()
    assert trace["steps"] > 0
    assert trace["candidate_count"] > 0
    assert trace["commit_count"] > 0
    assert trace["final_mask_count"] == 0


def test_generation_metrics_aggregate_sampler_trace(tmp_path):
    tok = ByteTokenizer()
    model = _TinyGenerationModel(tok.get_vocab_size())
    noise = MDLMNoise(tok.get_mask_token_id())

    def sample_fn(_model, n, _length):
        row = torch.tensor([ord("h"), ord("i"), tok.get_eos_token_id(), ord("x")], dtype=torch.long)
        trace = {
            "steps": 2,
            "candidate_count": 8,
            "commit_count": 8,
            "commit_count_min": 2,
            "commit_count_max": 6,
            "candidate_entropy_sum": 4.0,
            "candidate_top1_prob_sum": 2.0,
            "pred_max_token_frac_sum": 1.0,
            "pred_max_token_frac_steps": 2,
            "final_mask_count": 0,
            "length_policy_applied": 1,
            "length_policy_pre_eos_count": 0,
            "length_policy_post_eos_count": 2,
            "length_policy_predicted_position_count": 2,
            "length_policy_predicted_position_sum": 4,
            "length_policy_predicted_length_sum": 6,
            "length_policy_post_eos_position_0002_count": 2,
            "length_policy_post_eos_position_max_len": 4,
        }
        return row.repeat(n, 1), trace

    metrics = evaluate_generation_metrics(
        model,
        tok,
        noise,
        sample_fn,
        batch_size=2,
        run_dir=str(tmp_path),
        device=torch.device("cpu"),
        gen_samples=4,
        gen_len=4,
        log_samples=0,
        score_mode="none",
        step=7,
    )

    assert metrics["gen_trace_batches"] == 2
    assert metrics["gen_trace_steps_mean"] == 2
    assert metrics["gen_trace_commit_tokens_min"] == 2
    assert metrics["gen_trace_commit_tokens_max"] == 6
    assert metrics["gen_trace_candidate_entropy_mean"] == 0.5
    assert metrics["gen_length_policy_applied"] == 2
    assert metrics["gen_length_policy_post_eos_count"] == 4
    assert metrics["gen_length_policy_predicted_position_mean"] == 2.0
    assert metrics["gen_length_policy_predicted_length_mean"] == 3.0
    assert metrics["gen_length_policy_post_eos_position_0002_count"] == 4


def test_generation_metrics_aggregate_target_length_trace(tmp_path):
    tok = ByteTokenizer()
    model = _TinyGenerationModel(tok.get_vocab_size())
    noise = MDLMNoise(tok.get_mask_token_id())

    def sample_fn(_model, n, _length):
        assert n == 2
        tokens = torch.tensor(
            [
                [ord("h"), tok.get_eos_token_id(), ord("x"), ord("x")],
                [ord("h"), ord("i"), ord("j"), tok.get_eos_token_id()],
            ],
            dtype=torch.long,
        )
        trace = {
            "steps": 1,
            "candidate_count": 8,
            "commit_count": 8,
            "commit_count_min": 8,
            "commit_count_max": 8,
            "candidate_entropy_sum": 1.0,
            "candidate_top1_prob_sum": 7.0,
            "pred_max_token_frac_sum": 0.5,
            "pred_max_token_frac_steps": 1,
            "final_mask_count": 0,
            "target_length_values": [2, 4],
            "target_length_pre_policy_eos_positions": [1, -1],
            "target_length_post_policy_eos_positions": [1, 3],
            "target_length_forced_eos_count": 1,
        }
        return tokens, trace

    metrics = evaluate_generation_metrics(
        model,
        tok,
        noise,
        sample_fn,
        batch_size=2,
        run_dir=str(tmp_path),
        device=torch.device("cpu"),
        gen_samples=2,
        gen_len=4,
        log_samples=0,
        score_mode="none",
        step=0,
    )

    assert metrics["gen_target_length_conditioning_applied"] == 1
    assert metrics["gen_target_length_sample_count"] == 2
    assert metrics["gen_target_length_pre_policy_eos_count"] == 1
    assert metrics["gen_target_length_post_policy_eos_count"] == 2
    assert metrics["gen_target_length_forced_eos_count"] == 1
    assert metrics["gen_target_length_no_forced_eos_success_count"] == 1
    assert metrics["gen_target_length_pre_policy_eos_abs_error_median"] == pytest.approx(0.0)
    assert metrics["gen_target_length_post_policy_eos_abs_error_median"] == pytest.approx(0.0)
    assert metrics["gen_target_length_length_0002_count"] == 1
    assert metrics["gen_target_length_length_0004_count"] == 1


def test_generation_metrics_estimates_prior_correction_from_resolved_config(tmp_path):
    data_path = tmp_path / "train.parquet"
    pq.write_table(pa.table({"text": ["aaaa bbbb", "aaaa cccc", "dddd eeee", "ffff gggg"]}), data_path)
    (tmp_path / "resolved_config.json").write_text(
        json.dumps(
            {
                "data": {
                    "tokenizer": "byte",
                    "train_data": [str(data_path)],
                    "val_data": [],
                    "val_fraction": 0.0,
                    "random_length_ratio": 0.0,
                    "continuation_ratio": 0.0,
                },
                "train": {"eval_only": True},
                "eval": {
                    "gen_prior_correction_strength": 0.25,
                    "gen_prior_correction_batches": 1,
                    "gen_prior_correction_smoothing": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )
    tok = ByteTokenizer()
    model = _TinyGenerationModel(tok.get_vocab_size())
    noise = MDLMNoise(tok.get_mask_token_id())

    def sample_fn(_model, n, _length):
        row = torch.tensor([ord("h"), ord("i"), tok.get_eos_token_id(), ord("x")], dtype=torch.long)
        return row.repeat(n, 1)

    metrics = evaluate_generation_metrics(
        model,
        tok,
        noise,
        sample_fn,
        batch_size=2,
        run_dir=str(tmp_path),
        device=torch.device("cpu"),
        gen_samples=2,
        gen_len=4,
        log_samples=0,
        score_mode="none",
        step=0,
    )

    assert metrics["gen_prior_correction_strength"] == 0.25
    assert metrics["gen_prior_correction_batches"] == 1
    assert metrics["gen_prior_correction_token_count"] > 0
