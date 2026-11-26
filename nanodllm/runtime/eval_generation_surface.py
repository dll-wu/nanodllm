"""Shared per-row and corpus metrics for generated tokens and continuations."""

from collections import Counter
from dataclasses import dataclass

import torch

from .eval_generation_metrics import (
    _content_ids,
    _distinct_ngram_ratio,
    _entropy_std,
    _eos_position_metrics,
    _max_token_fraction,
    _ngram_entropy,
    _repeat_ngram_rate,
    _token_byte_total,
    _token_entropy,
    _truncate_at_eos,
)
from .eval_rank_metrics import _single_token_text


@dataclass
class GenerationRow:
    text: str
    truncated: list[int]
    has_eos: bool
    content: list[int]
    content_bytes: int


class GenerationStats:
    """Accumulate raw-token leakage, EOS-truncated lengths and content-only diversity."""

    def __init__(self, tokenizer, token_bytes: torch.Tensor):
        self.tokenizer = tokenizer
        self.token_bytes = token_bytes
        self.eos_id = tokenizer.get_eos_token_id()
        self.special_ids = tokenizer.get_special_token_ids()
        self.blocked_ids = tokenizer.get_generation_blocked_token_ids()
        self.texts: list[str] = []
        self.content_sequences: list[list[int]] = []
        self.content_counts: Counter[int] = Counter()
        self.decoded_lengths: list[int] = []
        self.eos_positions: Counter[int] = Counter()
        self.blocked_count = 0
        self.token_count = 0
        self.content_bytes = 0

    def add(self, ids: list[int]) -> GenerationRow:
        truncated, has_eos = _truncate_at_eos(ids, self.eos_id)
        content = _content_ids(ids, self.special_ids, self.eos_id)
        text = self.tokenizer.decode(ids)
        content_bytes = _token_byte_total(content, self.token_bytes)
        self.texts.append(text)
        self.content_sequences.append(content)
        self.content_counts.update(content)
        self.decoded_lengths.append(len(truncated))
        if has_eos:
            self.eos_positions[len(truncated) - 1] += 1
        self.token_count += len(ids)
        self.blocked_count += sum(1 for token_id in ids if int(token_id) in self.blocked_ids)
        self.content_bytes += content_bytes
        return GenerationRow(text, truncated, has_eos, content, content_bytes)

    def metrics(self, max_len: int) -> dict[str, float | int | str]:
        samples = len(self.texts)
        eos_count = sum(self.eos_positions.values())
        entropies = [_token_entropy(ids) for ids in self.content_sequences]
        max_fractions = [_max_token_fraction(ids) for ids in self.content_sequences]
        content_tokens = sum(self.content_counts.values())
        max_id, max_count = self.content_counts.most_common(1)[0] if self.content_counts else (-1, 0)
        empty_count = sum(1 for text in self.texts if not text.strip())
        return {
            "samples": samples,
            "empty_rate": empty_count / samples if samples else 0.0,
            "eos_rate": eos_count / samples if samples else 0.0,
            "eos_count": eos_count,
            "eos_token_id": int(self.eos_id) if self.eos_id is not None else -1,
            "avg_decoded_tokens": sum(self.decoded_lengths) / samples if samples else 0.0,
            "avg_chars": sum(map(len, self.texts)) / samples if samples else 0.0,
            "avg_content_bytes": self.content_bytes / samples if samples else 0.0,
            "content_tokens": content_tokens,
            "content_bytes": self.content_bytes,
            "content_token_byte_fertility": content_tokens / self.content_bytes if self.content_bytes else 0.0,
            "exact_duplicate_rate": 1.0 - (len(set(self.texts)) / samples) if samples else 0.0,
            "blocked_token_rate": self.blocked_count / self.token_count if self.token_count else 0.0,
            "distinct_1": _distinct_ngram_ratio(self.content_sequences, 1),
            "distinct_2": _distinct_ngram_ratio(self.content_sequences, 2),
            "distinct_3": _distinct_ngram_ratio(self.content_sequences, 3),
            "repeat_4gram_rate": _repeat_ngram_rate(self.content_sequences, 4),
            "max_token_frac_mean": sum(max_fractions) / samples if samples else 0.0,
            "max_token_id": int(max_id),
            "max_token_text": _single_token_text(self.tokenizer, int(max_id)),
            "max_token_count": int(max_count),
            "max_token_fraction": max_count / content_tokens if content_tokens else 0.0,
            "token_entropy_mean": sum(entropies) / samples if samples else 0.0,
            "token_entropy_std": _entropy_std(entropies),
            "unigram_entropy": _ngram_entropy(self.content_sequences, 1),
            "bigram_entropy": _ngram_entropy(self.content_sequences, 2),
            **_eos_position_metrics("", self.eos_positions, samples, max_len),
        }
