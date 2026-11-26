"""Materialize named optimizer groups with exact, disjoint parameter coverage."""

from collections.abc import Mapping, Sequence

import torch.nn as nn


def optimizer_group_parameters(
    model: nn.Module,
    contract: Mapping[str, object],
) -> dict[str, list[nn.Parameter]]:
    """Materialize contract optimizer groups while rechecking exact coverage."""

    named = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    raw_groups = contract.get("optimizer_groups")
    if not isinstance(raw_groups, list):
        raise TypeError("parameterization contract optimizer_groups must be a list")
    groups: dict[str, list[nn.Parameter]] = {}
    observed: list[str] = []
    for group in raw_groups:
        if not isinstance(group, dict) or not isinstance(group.get("group_id"), str):
            raise TypeError("parameterization optimizer group is malformed")
        names = group.get("parameter_names")
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            raise TypeError("parameterization optimizer group parameter_names must be strings")
        group_id = str(group["group_id"])
        if group_id in groups:
            raise ValueError(f"duplicate parameterization optimizer group {group_id!r}")
        unknown = sorted(set(names) - set(named))
        if unknown:
            raise ValueError(f"parameterization optimizer group {group_id!r} contains unknown names: {unknown}")
        groups[group_id] = [named[name] for name in names]
        observed.extend(names)
    missing = sorted(set(named) - set(observed))
    duplicates = sorted({name for name in observed if observed.count(name) > 1})
    extra = sorted(set(observed) - set(named))
    if missing or duplicates or extra or len(observed) != len(named):
        raise RuntimeError(
            "optimizer groups must cover every trainable parameter exactly once: "
            f"missing={missing}, duplicates={duplicates}, extra={extra}"
        )
    return groups


def validate_trainable_inventory(
    model: nn.Module,
    records: Sequence[Mapping[str, object]],
    *,
    excluded_parameter_prefixes: tuple[str, ...] = (),
) -> None:
    expected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith(excluded_parameter_prefixes)
    ]
    observed = [str(record["name"]) for record in records]
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        duplicates = sorted({name for name in observed if observed.count(name) > 1})
        extra = sorted(set(observed) - set(expected))
        raise RuntimeError(
            "parameterization taxonomy must cover every trainable named parameter exactly once: "
            f"missing={missing}, duplicates={duplicates}, extra={extra}"
        )
