"""Pure nanochat-compatible CORE prompt and target-span preparation.

No task download, checkpoint loading, or diffusion scoring is performed here.
The tokenizer must encode ordinary text without automatic BOS/EOS insertion.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from .core_protocol import FEWSHOT_SEED_BASE, TASK_SHUFFLE_SEED, TASK_TYPES


def _text(value: object, name: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be text")


def _choices(options: tuple[str, ...], gold: int) -> None:
    if not isinstance(options, tuple) or len(options) < 2 or any(not isinstance(value, str) for value in options):
        raise ValueError("choices/context_options must be a tuple of at least two strings")
    if type(gold) is not int or not 0 <= gold < len(options):
        raise ValueError("gold must index an option")


@dataclass(frozen=True)
class MultipleChoiceItem:
    query: str
    choices: tuple[str, ...]
    gold: int

    def __post_init__(self) -> None:
        _text(self.query, "query")
        _choices(self.choices, self.gold)


@dataclass(frozen=True)
class SchemaItem:
    context_options: tuple[str, ...]
    continuation: str
    gold: int

    def __post_init__(self) -> None:
        _text(self.continuation, "continuation")
        _choices(self.context_options, self.gold)


@dataclass(frozen=True)
class LanguageModelingItem:
    context: str
    continuation: str

    def __post_init__(self) -> None:
        _text(self.context, "context")
        _text(self.continuation, "continuation")


CoreItem = MultipleChoiceItem | SchemaItem | LanguageModelingItem
MC_SPAN_POLICIES = frozenset({"common_prefix", "context_boundary"})
LM_SPAN_POLICIES = frozenset({"strict_prefix", "answer_boundary"})


class PromptSpanError(ValueError):
    """An expected unscorable span, with a content-free reason for CPU audits."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


class CoreTokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def get_bos_token_id(self) -> int | None: ...


def task_indices(size: int, max_per_task: int | None = None) -> tuple[int, ...]:
    """Shuffle original row indices, then cap, before choosing demonstration rows."""
    if type(size) is not int or size < 1:
        raise ValueError("task size must be a positive integer")
    if max_per_task is not None and (type(max_per_task) is not int or max_per_task < 1):
        raise ValueError("max_per_task must be None or a positive integer")
    indices = list(range(size))
    random.Random(TASK_SHUFFLE_SEED).shuffle(indices)
    return tuple(indices[:max_per_task])


def fewshot_indices(size: int, index: int, num_fewshot: int) -> tuple[int, ...]:
    """Indices are relative to the already shuffled and capped task list."""
    if type(size) is not int or size < 1 or type(index) is not int or not 0 <= index < size:
        raise ValueError("index must address a nonempty task")
    if type(num_fewshot) is not int or not 0 <= num_fewshot < size:
        raise ValueError("num_fewshot must fit the pool excluding the current item")
    pool = [candidate for candidate in range(size) if candidate != index]
    return tuple(random.Random(FEWSHOT_SEED_BASE + index).sample(pool, num_fewshot))


def _mc_context(item: MultipleChoiceItem, delimiter: str, examples: Sequence[MultipleChoiceItem]) -> str:
    prefix = "".join(f"{example.query}{delimiter}{example.choices[example.gold]}\n\n" for example in examples)
    return f"{prefix}{item.query}{delimiter}"


def render_prompts(item: CoreItem, delimiter: str, examples: Sequence[CoreItem] = ()) -> tuple[str, ...]:
    """Render full strings before tokenization; preserve MC/schema whitespace."""
    _text(delimiter, "delimiter")
    if type(item) not in (MultipleChoiceItem, SchemaItem, LanguageModelingItem):
        raise ValueError("unsupported CORE item")
    if any(type(example) is not type(item) for example in examples):
        raise ValueError("few-shot examples must have the same task type")
    if isinstance(item, MultipleChoiceItem):
        context = _mc_context(item, delimiter, examples)
        return tuple(f"{context}{choice}" for choice in item.choices)
    if isinstance(item, SchemaItem):
        prefix = "".join(
            f"{example.context_options[example.gold]}{delimiter}{example.continuation}\n\n" for example in examples
        )
        return tuple(f"{prefix}{context}{delimiter}{item.continuation}" for context in item.context_options)
    prefix = "".join(f"{example.context.strip()}{delimiter}{example.continuation}\n\n" for example in examples)
    context = f"{prefix}{item.context.strip()}{delimiter}"
    return context.strip(), f"{context}{item.continuation}"


def _common_length(rows: tuple[tuple[int, ...], ...], *, suffix: bool = False) -> int:
    for offset in range(min(map(len, rows))):
        index = -offset - 1 if suffix else offset
        if any(row[index] != rows[0][index] for row in rows[1:]):
            return offset
    return min(map(len, rows))


def _literal_lm_prompts(item: LanguageModelingItem, delimiter: str, examples: Sequence[CoreItem]) -> tuple[str, str]:
    """Retain task whitespace so the answer's character boundary is explicit."""
    _text(delimiter, "delimiter")
    if any(not isinstance(example, LanguageModelingItem) for example in examples):
        raise ValueError("few-shot examples must have the same task type")
    prefix = "".join(f"{example.context}{delimiter}{example.continuation}\n\n" for example in examples)
    prompt = f"{prefix}{item.context}{delimiter}"
    return prompt, prompt + item.continuation


def _answer_boundary(tokenizer: CoreTokenizer, prompt: str, full: str, tokens: tuple[int, ...]) -> int:
    """Mask every full-text token overlapping answer characters, including merges."""
    encode_offsets = getattr(tokenizer, "encode_with_offsets", None)
    if not callable(encode_offsets):
        raise ValueError("answer_boundary requires a tokenizer with character offsets")
    encoded, offsets = encode_offsets(full)
    if tuple(encoded) != tokens or len(offsets) != len(tokens):
        raise ValueError("offset tokenization must match the ordinary full-text encoding")
    previous = (0, 0)
    start = None
    for index, offset in enumerate(offsets):
        if (
            not isinstance(offset, (tuple, list))
            or len(offset) != 2
            or any(type(value) is not int for value in offset)
            or not 0 <= offset[0] < offset[1] <= len(full)
            or offset[0] < previous[0]
            or offset[1] < previous[1]
        ):
            raise ValueError("token offsets must be nonempty monotone character intervals")
        previous = offset
        if start is None and offset[1] > len(prompt):
            start = index + 1  # The explicit BOS has no text offset.
    if start is None:
        raise PromptSpanError("empty_target", "answer has no token overlapping its characters")
    return start


@dataclass(frozen=True)
class TargetSequence:
    tokens: tuple[int, ...]
    start: int
    end: int
    cropped_tokens: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.tokens, tuple) or any(type(token) is not int or token < 0 for token in self.tokens):
            raise ValueError("tokens must be a tuple of nonnegative integer IDs")
        if (
            type(self.start) is not int
            or type(self.end) is not int
            or not 1 <= self.start < self.end <= len(self.tokens)
        ):
            raise ValueError("target span must be nonempty and retain a causal predecessor")
        if type(self.cropped_tokens) is not int or self.cropped_tokens < 0:
            raise ValueError("cropped_tokens must be a nonnegative integer")


@dataclass(frozen=True)
class PreparedExample:
    task_type: str
    sequences: tuple[TargetSequence, ...]
    bos_token_id: int
    gold: int | None

    def __post_init__(self) -> None:
        if self.task_type not in TASK_TYPES or not isinstance(self.sequences, tuple) or not self.sequences:
            raise ValueError("invalid prepared example")
        if any(not isinstance(sequence, TargetSequence) for sequence in self.sequences):
            raise ValueError("sequences must contain validated TargetSequence values")
        if type(self.bos_token_id) is not int or self.bos_token_id < 0:
            raise ValueError("BOS must be a nonnegative integer")
        if self.task_type == "language_modeling":
            if len(self.sequences) != 1 or self.gold is not None:
                raise ValueError("LM examples have one continuation and no choice index")
        elif len(self.sequences) < 2 or type(self.gold) is not int or not 0 <= self.gold < len(self.sequences):
            raise ValueError("MC/schema examples require options and a valid gold index")


def prepare_example(
    item: CoreItem,
    tokenizer: CoreTokenizer,
    *,
    delimiter: str,
    examples: Sequence[CoreItem] = (),
    max_tokens: int,
    crop_multiple: int = 1,
    mc_span_policy: str = "common_prefix",
    lm_span_policy: str = "strict_prefix",
) -> PreparedExample:
    """Locate answer tokens, then crop context without dropping scored targets.

    Unlike the upstream start>=0 assertion, we require start>=1: target token
    zero has no causal predecessor and must never reach a negative loss slice.
    """
    if type(max_tokens) is not int or max_tokens < 2:
        raise ValueError("max_tokens must be an integer >= 2")
    if type(crop_multiple) is not int or crop_multiple < 1:
        raise ValueError("crop_multiple must be a positive integer")
    if not isinstance(mc_span_policy, str) or mc_span_policy not in MC_SPAN_POLICIES:
        raise ValueError("unsupported MC span policy")
    if not isinstance(lm_span_policy, str) or lm_span_policy not in LM_SPAN_POLICIES:
        raise ValueError("unsupported LM span policy")
    bos = tokenizer.get_bos_token_id()
    if type(bos) is not int or bos < 0:
        raise ValueError("tokenizer must supply an explicit BOS token ID")
    prompts = (
        _literal_lm_prompts(item, delimiter, examples)
        if isinstance(item, LanguageModelingItem) and lm_span_policy == "answer_boundary"
        else render_prompts(item, delimiter, examples)
    )
    rows = tuple((bos, *tokenizer.encode(prompt)) for prompt in prompts)
    if any(type(token) is not int or token < 0 for row in rows for token in row):
        raise ValueError("tokenizer must return nonnegative integer token IDs")
    if isinstance(item, MultipleChoiceItem):
        task_type = "multiple_choice"
        boundary_rows = rows
        if mc_span_policy == "context_boundary":
            # Including the context-only encoding stops shared answer text
            # from entering conditioning, and retreats across boundary merges.
            context_tokens = (bos, *tokenizer.encode(_mc_context(item, delimiter, examples)))
            if any(type(token) is not int or token < 0 for token in context_tokens):
                raise ValueError("tokenizer must return nonnegative integer token IDs")
            boundary_rows = (context_tokens, *rows)
        starts = (_common_length(boundary_rows),) * len(rows)
        gold = item.gold
    elif isinstance(item, SchemaItem):
        task_type = "schema"
        suffix = _common_length(rows, suffix=True)
        starts = tuple(len(row) - suffix for row in rows)
        gold = item.gold
    else:
        task_type = "language_modeling"
        without, complete = rows
        if lm_span_policy == "answer_boundary":
            start = _answer_boundary(tokenizer, prompts[0], prompts[1], complete[1:])
        elif complete[: len(without)] != without:
            raise PromptSpanError(
                "lm_prefix_mismatch", "LM context must be an exact token prefix of the completed prompt"
            )
        else:
            start = len(without)
        starts = (start,)
        rows = (complete,)
        gold = None
    sequences = []
    for row, start in zip(rows, starts, strict=True):
        if start >= len(row):
            raise PromptSpanError("empty_target", "target span must be nonempty and retain a causal predecessor")
        if start < 1:
            raise PromptSpanError("missing_predecessor", "target span must be nonempty and retain a causal predecessor")
        crop = max(0, len(row) - max_tokens)
        crop = (crop + crop_multiple - 1) // crop_multiple * crop_multiple
        if start - crop < 1:
            raise PromptSpanError("context_capacity", "target span must be nonempty and retain a causal predecessor")
        sequences.append(TargetSequence(row[crop:], start - crop, len(row) - crop, crop))
    return PreparedExample(task_type, tuple(sequences), bos, gold)
