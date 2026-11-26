"""Versioned CORE reference metadata and count-based aggregation, without data I/O.

This is not an admitted benchmark runner. In particular, a complete reference
roster does not imply permission to use its datasets or official CORE parity.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

SCHEMA = "nanodllm.core_reference_protocol.v1"
PROTOCOL_ID = "dclm_core_v2_nanochat_ar_reference.v1"
TASK_TYPES = ("multiple_choice", "schema", "language_modeling")
TASK_SHUFFLE_SEED = 1337
FEWSHOT_SEED_BASE = 1234
SCORING_RULES = {
    "bos": "prepend_once_before_truncation;right_pad_with_bos",
    "tokenization": "whole_rendered_prompt;no_automatic_special_tokens;no_eos",
    "mc_span": "common_token_prefix_across_choices",
    "schema_span": "common_token_suffix_across_contexts",
    "lm_span": "strict_token_prefix_without_continuation",
    "candidate_score": "mean_causal_token_nll;minimum;first_exact_tie",
    "lm_score": "all_teacher_forced_next_token_argmax_match;not_generation",
    "context_limit": "left_crop;preserve_all_targets_and_one_predecessor",
    "task_shuffle_seed": TASK_SHUFFLE_SEED,
    "fewshot_seed_base": FEWSHOT_SEED_BASE,
    "fewshot_pool": "shuffled_then_capped_task;exclude_current_index",
    "aggregation": "unweighted_mean_of_(accuracy-baseline)/(1-baseline);no_clipping",
}


def _fields(value: object, expected: set[str], name: str) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name} must have exactly these fields: {sorted(expected)}")
    return value


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class SourcePin:
    url: str
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.url, str) or not re.fullmatch(
            r"https://raw\.githubusercontent\.com/[\w-]+/[\w-]+/[0-9a-f]{40}/[\w./-]+", self.url
        ):
            raise ValueError("source URL must pin a full GitHub commit")
        if not isinstance(self.sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("source sha256 must be a lowercase SHA-256")


@dataclass(frozen=True)
class ReferenceTask:
    label: str
    task_type: str
    reference_num_fewshot: int
    baseline_percent: float
    data_admission: str

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", self.label):
            raise ValueError("invalid task label")
        if self.task_type not in TASK_TYPES:
            raise ValueError("unsupported task_type")
        _integer(self.reference_num_fewshot, "reference_num_fewshot")
        if type(self.baseline_percent) not in (int, float) or not 0 <= self.baseline_percent < 100:
            raise ValueError("baseline_percent must be finite and in [0, 100)")
        if self.data_admission not in ("pending_review", "excluded"):
            raise ValueError("reference metadata cannot admit task data")
        if self.label == "jeopardy" and self.data_admission != "excluded":
            raise ValueError("Jeopardy remains excluded under the existing data policy")


@dataclass(frozen=True)
class CoreReferenceProtocol:
    sources: tuple[tuple[str, SourcePin], ...]
    tasks: tuple[ReferenceTask, ...]
    protocol_id: str = PROTOCOL_ID
    status: str = "reference_only"


def load_core_reference_protocol(path: str | Path) -> CoreReferenceProtocol:
    """Read metadata only; never resolve URLs, task paths, or a dataset bundle."""
    raw = _fields(
        json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object),
        {"schema", "protocol_id", "status", "sources", "scoring", "data_bundle", "tasks"},
        "CORE reference protocol",
    )
    if (raw["schema"], raw["protocol_id"], raw["status"]) != (SCHEMA, PROTOCOL_ID, "reference_only"):
        raise ValueError("unsupported CORE reference protocol version/status")
    if raw["scoring"] != SCORING_RULES:
        raise ValueError("scoring rules differ from the implemented contract")
    if raw["data_bundle"] is not None:
        raise ValueError("this reference-only protocol does not admit real task bundles")
    sources = _fields(raw["sources"], {"roster", "baselines", "ar_scorer", "orchestration"}, "sources")
    pins = tuple(
        (role, SourcePin(**_fields(value, {"url", "sha256"}, f"source {role}"))) for role, value in sources.items()
    )
    if not isinstance(raw["tasks"], list) or len(raw["tasks"]) != 22:
        raise ValueError("the reference roster must contain 22 task entries")
    tasks: list[ReferenceTask] = []
    for entry in raw["tasks"]:
        fields = _fields(
            entry,
            {
                "label",
                "task_type",
                "reference_num_fewshot",
                "baseline_percent",
                "data_admission",
                "continuation_delimiter",
                "data_sha256",
            },
            "task",
        )
        if fields["continuation_delimiter"] is not None or fields["data_sha256"] is not None:
            raise ValueError("bundle-derived delimiters and data identities are not yet bound")
        tasks.append(
            ReferenceTask(
                **{key: value for key, value in fields.items() if key not in ("continuation_delimiter", "data_sha256")}
            )
        )
    if len({task.label for task in tasks}) != len(tasks):
        raise ValueError("duplicate task label")
    return CoreReferenceProtocol(pins, tuple(tasks))


@dataclass(frozen=True)
class TaskCounts:
    correct: int
    total: int

    def __post_init__(self) -> None:
        _integer(self.correct, "correct")
        _integer(self.total, "total", 1)
        if self.correct > self.total:
            raise ValueError("correct cannot exceed total")


@dataclass(frozen=True)
class TaskSummary:
    label: str
    correct: int
    total: int
    accuracy: float
    baseline: float
    centered_accuracy: float


@dataclass(frozen=True)
class ReferenceSummary:
    protocol_id: str
    metric_name: str
    mean_centered_accuracy: float
    tasks: tuple[TaskSummary, ...]
    missing_tasks: tuple[str, ...]
    official_core_available: bool = False


def summarize_reference_counts(
    protocol: CoreReferenceProtocol,
    counts: Mapping[str, TaskCounts],
    *,
    require_full_roster: bool = True,
) -> ReferenceSummary:
    """Aggregate raw counts without treating metadata coverage as benchmark parity.

    A subset requires explicit opt-in and is named as a subset. Even a full
    synthetic/reference calculation cannot emit an official CORE result.
    """
    known = {task.label for task in protocol.tasks}
    if not counts or set(counts) - known:
        raise ValueError("counts must be nonempty and contain only reference tasks")
    if any(not isinstance(value, TaskCounts) for value in counts.values()):
        raise ValueError("counts must contain validated TaskCounts")
    missing = tuple(task.label for task in protocol.tasks if task.label not in counts)
    if missing and require_full_roster:
        raise ValueError(f"full CORE reference roster is incomplete: {', '.join(missing)}")
    rows: list[TaskSummary] = []
    for task in protocol.tasks:
        if task.label not in counts:
            continue
        count = counts[task.label]
        accuracy = count.correct / count.total
        baseline = task.baseline_percent / 100
        rows.append(
            TaskSummary(
                task.label, count.correct, count.total, accuracy, baseline, (accuracy - baseline) / (1 - baseline)
            )
        )
    coverage = "subset" if missing else "full_roster"
    return ReferenceSummary(
        protocol.protocol_id,
        f"core_v2_reference_{coverage}_centered_mean",
        math.fsum(row.centered_accuracy for row in rows) / len(rows),
        tuple(rows),
        missing,
    )
