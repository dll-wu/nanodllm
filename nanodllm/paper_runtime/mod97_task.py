"""Deterministic modulo-97 computation graphs for the D4 task-floor screen."""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass

MODULUS = 97
TRAIN_DEPTHS = (1, 2, 3, 4)
EVAL_DEPTHS = (0, 1, 4, 5, 6)
STATEMENT_RANGE = (8, 16)
ANSWER_DATASET = "mod97_task_a_answer_v1"
TRACE_DATASET = "mod97_task_a_trace_v1"
DATASETS = frozenset({ANSWER_DATASET, TRACE_DATASET})
SUPERVISION_BY_DATASET = {ANSWER_DATASET: "answer", TRACE_DATASET: "trace"}
TRAIN_SEED_BASE = 2_026_091_700_000
EVAL_SEED_BASE = 8_026_091_700_000

_NAME_ROOTS = (
    "amber",
    "cobalt",
    "saffron",
    "jade",
    "indigo",
    "marigold",
    "silver",
    "willow",
    "cedar",
    "raven",
    "orchid",
    "citrine",
    "violet",
    "maple",
    "coral",
    "juniper",
    "topaz",
    "lilac",
    "falcon",
    "birch",
)
_CONST = re.compile(r"^([a-z][a-z0-9_]*) = ([0-9]+)$")
_OP = re.compile(r"^([a-z][a-z0-9_]*) = (add|sub|mul)\(([a-z][a-z0-9_]*), ([0-9]+)\)$")
_TRACE_PAIR = re.compile(r"([a-z][a-z0-9_]*)\s*=\s*(-?[0-9]+)")
_ANSWER = re.compile(r"-?[0-9]+")
_TRACE_ANSWER = re.compile(r"Answer\s*:\s*(-?[0-9]+)", re.IGNORECASE)


@dataclass(frozen=True)
class Statement:
    target: str
    operation: str
    left: str | None
    right: int


@dataclass(frozen=True)
class Mod97Problem:
    problem_id: str
    seed: int
    depth: int
    shuffled: bool
    statements: tuple[Statement, ...]
    displayed_statements: tuple[Statement, ...]
    values: tuple[tuple[str, int], ...]
    target: str
    answer: int


def is_mod97_dataset(name: str) -> bool:
    return name in DATASETS


def _base36(value: int) -> str:
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    number = int(value)
    if number == 0:
        return "0"
    out = []
    while number:
        number, remainder = divmod(number, 36)
        out.append(alphabet[remainder])
    return "".join(reversed(out))


def _names(seed: int, count: int, rng: random.Random) -> list[str]:
    roots = list(_NAME_ROOTS)
    rng.shuffle(roots)
    seed_tag = _base36(seed)[-5:]
    names = []
    for index in range(count):
        root = roots[index % len(roots)]
        cycle = index // len(roots)
        if index % 3 == 0:
            name = root
        elif index % 3 == 1:
            name = f"{root}_{seed_tag}{index}"
        else:
            name = f"{root}{seed_tag}{cycle}"
        if name in names:
            name = f"{name}_{index}"
        names.append(name)
    return names


def _apply(operation: str, left: int, right: int) -> int:
    if operation == "add":
        return (left + right) % MODULUS
    if operation == "sub":
        return (left - right) % MODULUS
    if operation == "mul":
        return (left * right) % MODULUS
    raise ValueError(f"unsupported operation: {operation}")


def render_statement(statement: Statement) -> str:
    if statement.operation == "const":
        return f"{statement.target} = {statement.right}"
    if statement.left is None:
        raise ValueError("nonconstant statement has no left operand")
    return f"{statement.target} = {statement.operation}({statement.left}, {statement.right})"


def interpret_declarations(lines: tuple[str, ...] | list[str]) -> dict[str, int]:
    pending: list[Statement] = []
    for line in lines:
        constant = _CONST.fullmatch(line)
        if constant:
            pending.append(Statement(constant.group(1), "const", None, int(constant.group(2))))
            continue
        operation = _OP.fullmatch(line)
        if operation:
            pending.append(
                Statement(operation.group(1), operation.group(2), operation.group(3), int(operation.group(4)))
            )
            continue
        raise ValueError(f"invalid declaration: {line!r}")
    if len({statement.target for statement in pending}) != len(pending):
        raise ValueError("duplicate declaration target")
    values: dict[str, int] = {}
    while pending:
        next_pending = []
        progressed = False
        for statement in pending:
            if statement.operation == "const":
                values[statement.target] = statement.right % MODULUS
                progressed = True
            elif statement.left in values:
                values[statement.target] = _apply(statement.operation, values[statement.left], statement.right)
                progressed = True
            else:
                next_pending.append(statement)
        if not progressed:
            raise ValueError("declarations contain an unresolved dependency or cycle")
        pending = next_pending
    return values


def generate_problem(
    seed: int, depth: int, *, statement_count: int | None = None, shuffled: bool | None = None
) -> Mod97Problem:
    if type(seed) is not int or type(depth) is not int or depth < 0:
        raise ValueError("seed and nonnegative integer depth are required")
    rng = random.Random(seed)
    minimum = max(STATEMENT_RANGE[0], depth + 1)
    maximum = max(minimum, STATEMENT_RANGE[1])
    count = rng.randint(minimum, maximum) if statement_count is None else int(statement_count)
    if count < depth + 1 or not STATEMENT_RANGE[0] <= count <= STATEMENT_RANGE[1]:
        raise ValueError("statement count cannot realize the requested depth")
    names = _names(seed, count, rng)
    operations = ("add", "sub", "mul")
    statements: list[Statement] = [Statement(names[0], "const", None, rng.randrange(MODULUS))]
    values: dict[str, int] = {names[0]: statements[0].right}
    previous = names[0]
    for index in range(1, depth + 1):
        operation = rng.choice(operations)
        right = rng.randrange(1, MODULUS)
        statement = Statement(names[index], operation, previous, right)
        statements.append(statement)
        values[statement.target] = _apply(operation, values[previous], right)
        previous = statement.target
    target = previous
    distractor_names: list[str] = []
    for index in range(depth + 1, count):
        if not distractor_names or rng.random() < 0.4:
            statement = Statement(names[index], "const", None, rng.randrange(MODULUS))
            value = statement.right
        else:
            left = rng.choice(distractor_names)
            operation = rng.choice(operations)
            right = rng.randrange(1, MODULUS)
            statement = Statement(names[index], operation, left, right)
            value = _apply(operation, values[left], right)
        statements.append(statement)
        distractor_names.append(statement.target)
        values[statement.target] = value
    use_shuffle = bool(seed % 2) if shuffled is None else bool(shuffled)
    displayed = list(statements)
    if use_shuffle:
        rng.shuffle(displayed)
    interpreted = interpret_declarations([render_statement(statement) for statement in displayed])
    if interpreted != values or interpreted[target] != values[target]:
        raise AssertionError("generated problem failed interpreter verification")
    identity = {
        "seed": seed,
        "depth": depth,
        "shuffled": use_shuffle,
        "statements": [statement.__dict__ for statement in statements],
        "target": target,
    }
    problem_id = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return Mod97Problem(
        problem_id=problem_id,
        seed=seed,
        depth=depth,
        shuffled=use_shuffle,
        statements=tuple(statements),
        displayed_statements=tuple(displayed),
        values=tuple(values.items()),
        target=target,
        answer=values[target],
    )


def problem_text(problem: Mod97Problem) -> str:
    declarations = "\n".join(render_statement(statement) for statement in problem.displayed_statements)
    return f"Arithmetic is modulo 97. Declarations may appear in any order.\n{declarations}\nReturn {problem.target}."


def training_parts(problem: Mod97Problem, supervision: str) -> tuple[str, str]:
    if supervision == "answer":
        return problem_text(problem) + "\nAnswer:", f" {problem.answer}"
    if supervision == "trace":
        values = dict(problem.values)
        trace = "; ".join(f"{statement.target}={values[statement.target]}" for statement in problem.statements)
        return problem_text(problem) + "\nTrace:", f" {trace}.\nAnswer: {problem.answer}"
    raise ValueError(f"unsupported supervision: {supervision}")


def train_problem(index: int, data_seed: int) -> Mod97Problem:
    if type(index) is not int or index < 0:
        raise ValueError("training problem index must be nonnegative")
    depth = TRAIN_DEPTHS[index % len(TRAIN_DEPTHS)]
    seed = TRAIN_SEED_BASE + int(data_seed) * 1_000_003 + index
    return generate_problem(seed, depth, shuffled=bool(index % 2))


def evaluation_problem(depth: int, index: int, evaluation_seed: int) -> Mod97Problem:
    if depth not in EVAL_DEPTHS or type(index) is not int or index < 0:
        raise ValueError("unsupported evaluation stratum")
    seed = EVAL_SEED_BASE + int(evaluation_seed) * 1_000_003 + depth * 100_003 + index
    return generate_problem(seed, depth, shuffled=bool(index % 2))


def parse_generated(text: str, supervision: str) -> tuple[int | None, dict[str, int]]:
    if supervision == "answer":
        match = _ANSWER.search(text)
        return (int(match.group(0)) if match else None), {}
    if supervision == "trace":
        answer = _TRACE_ANSWER.search(text)
        prefix = text[: answer.start()] if answer else text
        pairs = {name: int(value) for name, value in _TRACE_PAIR.findall(prefix)}
        return (int(answer.group(1)) if answer else None), pairs
    raise ValueError(f"unsupported supervision: {supervision}")
