"""Display conventions shared by the paper exporters."""

import math


def one(rows: list[dict], **match: object) -> dict:
    found = [row for row in rows if all(row.get(k) == v for k, v in match.items())]
    if len(found) != 1:
        raise ValueError(f"Expected one row for {match}, found {len(found)}")
    return found[0]


def num(value: float | None, places: int = 4) -> str:
    if value is None:
        return "---"
    if not math.isfinite(value):
        raise ValueError("Non-finite table value")
    return f"{value:.{places}f}"


def bpb(value: float | None) -> str:
    return num(value, 3)


def bpb_delta(value: float) -> str:
    places = 4 if 0 < abs(value) < 0.001 else 3
    return f"{value:+.{places}f}"


def bpb_interval(low: float, high: float) -> str:
    places = 4 if any(0 < abs(v) < 0.001 for v in (low, high)) else 3
    return f"[{low:+.{places}f}, {high:+.{places}f}]"
