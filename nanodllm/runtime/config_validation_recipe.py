"""Recipe lifecycle and declared configuration-field validation."""

from dataclasses import asdict
from typing import Any


def validate_recipe(config) -> None:
    if config.recipe.status not in ("unspecified", "historical", "provisional", "locked"):
        raise ValueError("recipe.status must be one of 'unspecified', 'historical', 'provisional', or 'locked'")
    if config.recipe.status != "unspecified" and not config.recipe.name:
        raise ValueError("recipe.name is required when recipe.status is not 'unspecified'")
    for field_name, paths in (
        ("locked_fields", config.recipe.locked_fields),
        ("provisional_fields", config.recipe.provisional_fields),
    ):
        if not isinstance(paths, list) or any(not isinstance(path, str) or not path for path in paths):
            raise ValueError(f"recipe.{field_name} must be a list of non-empty dotted field paths")
        if len(set(paths)) != len(paths):
            raise ValueError(f"recipe.{field_name} must not contain duplicates")
    overlap = set(config.recipe.locked_fields) & set(config.recipe.provisional_fields)
    if overlap:
        raise ValueError(f"recipe locked_fields and provisional_fields overlap: {sorted(overlap)}")
    if config.recipe.status == "locked" and config.recipe.provisional_fields:
        raise ValueError("recipe.status='locked' cannot declare provisional_fields")
    if config.recipe.status == "provisional" and not config.recipe.provisional_fields:
        raise ValueError("recipe.status='provisional' requires at least one provisional field")
    config_tree = asdict(config)
    for dotted_path in config.recipe.locked_fields + config.recipe.provisional_fields:
        node: Any = config_tree
        for part in dotted_path.split("."):
            if not isinstance(node, dict) or part not in node or part == "recipe":
                raise ValueError(f"recipe.provisional_fields contains unknown config path: {dotted_path}")
            node = node[part]
