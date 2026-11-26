"""Locate release assets shipped at the same relative paths in source and wheels."""

from pathlib import Path


def release_asset(relative_name: str) -> Path:
    return Path(__file__).resolve().parents[1] / relative_name
