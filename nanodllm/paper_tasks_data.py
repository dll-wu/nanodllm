"""Local task rosters and corrected block-aligned example preparation."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .paper_runtime.core_prompting import (
    LanguageModelingItem,
    MultipleChoiceItem,
    PreparedExample,
    SchemaItem,
    fewshot_indices,
    prepare_example,
    task_indices,
)
from .paper_runtime.core_task_data import json_digest
from .release_assets import release_asset


@dataclass(frozen=True)
class LocalTask:
    label: str
    task_type: str
    delimiter: str
    num_fewshot: int
    data_sha256: str


@dataclass(frozen=True)
class Example:
    original_index: int
    demonstration_indices: tuple[int, ...]
    sample_id: str
    example: PreparedExample


def parse_item(row, kind):
    if kind == "multiple_choice":
        if not isinstance(row["choices"], list):
            raise ValueError("expected choices array")
        return MultipleChoiceItem(row["query"], tuple(row["choices"]), row["gold"])
    if kind == "schema":
        if not isinstance(row["context_options"], list):
            raise ValueError("expected context_options array")
        return SchemaItem(tuple(row["context_options"]), row["continuation"], row["gold"])
    return LanguageModelingItem(row["context"], row["continuation"])


PAPER_ROSTER = release_asset("evaluation/tasks/core21.json")


@dataclass(frozen=True)
class Panel:
    task: LocalTask
    baseline_v1: float
    baseline_v2: float
    examples: tuple
    selection_sha256: str
    prepared_inputs_sha256: str


def retain_bos(example, block_size):
    sequences = []
    for seq in example.sequences:
        if seq.cropped_tokens % block_size:
            raise ValueError("unaligned crop")
        if seq.tokens[0] != example.bos_token_id:
            if seq.cropped_tokens <= 0 or seq.start < 1:
                raise ValueError("cannot restore BOS without dropping an answer")
            seq = replace(seq, tokens=(example.bos_token_id, *seq.tokens[1:]))
        sequences.append(seq)
    return replace(example, sequences=tuple(sequences))


def is_study_roster(path):
    return (
        hashlib.sha256(Path(path).read_bytes()).hexdigest()
        == "cd0bc00ecfa019d5153ac537a80d5f9ae5dc9db30d2d0cdc5432f8b7d18e0f25"
    )


def check_study_tokenizer(path, content_sha256):
    if is_study_roster(path) and json.loads(Path(path).read_bytes())["tokenizer_sha256"] != content_sha256:
        raise ValueError("paper roster requires the pinned tokenizer contents")


def read_payload(root, spec):
    root = Path(root).resolve(strict=True)
    if root.is_dir():
        path = (root / spec["data_path"]).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("task file must be inside the supplied bundle")
        return path.read_bytes()
    with zipfile.ZipFile(root) as archive:
        suffix = spec["data_path"]
        matches = [i for i in archive.infolist() if i.filename == suffix or i.filename.endswith("/" + suffix)]
        if len(matches) != 1 or matches[0].file_size != spec["data_bytes"]:
            raise ValueError(f"missing, ambiguous or wrong-sized archive member for {spec['label']}")
        return archive.read(matches[0])


def read_roster(path):
    roster = json.loads(Path(path).read_text(encoding="utf-8"))
    if roster.get("schema") != "nanodllm.paper_tasks.roster.v1":
        raise ValueError("unsupported task roster")
    labels = set()
    for row in roster["tasks"]:
        label = row["label"]
        if not re.fullmatch("[a-z][a-z0-9_]*", label) or label in labels:
            raise ValueError("expected unique task identifiers")
        labels.add(label)
        if row["task_type"] not in {"multiple_choice", "schema", "language_modeling"}:
            raise ValueError("unsupported task type")
        if not isinstance(row["delimiter"], str) or not 0 <= row["num_fewshot"] < row["selected_count"]:
            raise ValueError("invalid template settings")
        if row["selected_count"] != min(row["raw_count"], 500):
            raise ValueError("invalid capped denominator")
        if not all(0 <= row[k] < 1 for k in ("baseline_v1", "baseline_v2")):
            raise ValueError("baselines must be fractions in [0, 1)")
    if not labels:
        raise ValueError("empty task roster")
    return sorted(roster["tasks"], key=lambda s: s["task_type"] != "language_modeling")


def prepare_panel(spec, root, tokenizer, *, context=1024, block_size=32, limit=None):
    data = read_payload(root, spec)
    digest = hashlib.sha256(data).hexdigest()
    if digest != spec["data_sha256"]:
        raise ValueError(f"task data SHA-256 mismatch for {spec['label']}")
    try:
        items = tuple(parse_item(json.loads(line), spec["task_type"]) for line in data.splitlines() if line.strip())
    except (TypeError, ValueError, KeyError):
        raise ValueError(f"invalid local item in {spec['label']}") from None
    if len(items) != spec["raw_count"]:
        raise ValueError("task denominator changed")
    task = LocalTask(spec["label"], spec["task_type"], spec["delimiter"], spec["num_fewshot"], digest)
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer")
    pool = task_indices(len(items), 500)
    examples = []
    for i, original in enumerate(pool[:limit]):
        demos = tuple(pool[j] for j in fewshot_indices(len(pool), i, task.num_fewshot))
        example = prepare_example(
            items[original],
            tokenizer,
            delimiter=task.delimiter,
            examples=[items[j] for j in demos],
            max_tokens=context,
            crop_multiple=block_size,
            mc_span_policy="context_boundary",
            lm_span_policy="answer_boundary",
        )
        examples.append(Example(original, demos, f"{task.label}:{original}", retain_bos(example, block_size)))
    selected = json_digest([(r.original_index, r.demonstration_indices) for r in examples])
    inputs = json_digest([(r.original_index, json_digest(asdict(r.example))) for r in examples])
    if len(examples) == spec["selected_count"] and (
        spec.get("selection_sha256", selected) != selected or spec.get("prepared_inputs_sha256", inputs) != inputs
    ):
        raise ValueError(f"prepared panel differs from the pinned rows/tokenizer/crops for {task.label}")
    return Panel(task, spec["baseline_v1"], spec["baseline_v2"], tuple(examples), selected, inputs)
