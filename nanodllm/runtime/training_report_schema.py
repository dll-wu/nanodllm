from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FieldGroup:
    """One named group of related fields within a report fragment."""

    name: str
    fields: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ReportFragment:
    """One named, independently owned contribution to a flat report."""

    name: str
    fields: Mapping[str, object]


def _merge_named_fields(
    sources: Iterable[tuple[str, Mapping[str, object]]],
    *,
    owner_kind: str,
) -> dict[str, object]:
    merged: dict[str, object] = {}
    owners: dict[str, str] = {}
    for owner, fields in sources:
        for key, value in fields.items():
            if key in merged:
                raise ValueError(
                    f"report field {key!r} from {owner_kind} {owner!r} collides with {owner_kind} {owners[key]!r}"
                )
            merged[key] = deepcopy(value)
            owners[key] = owner
    return merged


def merge_field_groups(*groups: FieldGroup) -> dict[str, object]:
    """Flatten field groups in order while rejecting every duplicate field."""

    return _merge_named_fields(
        ((group.name, group.fields) for group in groups),
        owner_kind="field group",
    )


def merge_report_fragments(*fragments: ReportFragment) -> dict[str, object]:
    """Flatten report fragments in order while rejecting every duplicate field."""

    return _merge_named_fields(
        ((fragment.name, fragment.fields) for fragment in fragments),
        owner_kind="report fragment",
    )
