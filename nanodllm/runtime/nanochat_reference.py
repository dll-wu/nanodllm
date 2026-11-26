"""Load the pinned nanochat optimizer with an optional eager CPU execution path."""

from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path
from types import ModuleType

UPSTREAM_COMMIT = "92d63d4e8bb4df75c3b71618f31ddde2378b2bcd"
SOURCE_HASHES = {
    "nanochat/optim.py": "7c9b714c6f76e3a9d50fc530044b2767d0cd89391237e609548f1cd730afed8f",
    "LICENSE": "42bb09f339a8af8b90d3b4969b6f6e4ba4aff3b5c614103b031fccdf71c8f772",
}


def verified_source(checkout: Path, relative: str) -> bytes:
    if relative not in SOURCE_HASHES:
        raise ValueError(f"unapproved nanochat source: {relative}")
    candidate = checkout / relative
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"not a regular upstream source: {candidate}")
    source = candidate.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_HASHES[relative]:
        raise ValueError(f"nanochat source differs from {UPSTREAM_COMMIT}: {relative}")
    return source


def _module(
    checkout: Path,
    relative: str,
    bindings: dict,
    *,
    eager_optimizer=False,
    persistent=False,
):
    tree = ast.parse(verified_source(checkout, relative), filename=str(checkout / relative))
    body = []
    compile_count = 0
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("nanochat."):
            continue
        if eager_optimizer and isinstance(node, ast.FunctionDef):
            decorators = []
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call) and ast.unparse(decorator.func) == "torch.compile":
                    compile_count += 1
                else:
                    decorators.append(decorator)
            node.decorator_list = decorators
        body.append(node)
    if eager_optimizer and compile_count != 2:
        raise ValueError("upstream optimizer compile boundary changed")
    tree.body = body
    name = "nanodllm._nanochat_reference_" + Path(relative).stem
    module = ModuleType(name)
    module.__file__ = str(checkout / relative)
    module.__dict__.update(bindings)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        exec(compile(ast.fix_missing_locations(tree), module.__file__, "exec", dont_inherit=True), module.__dict__)
    finally:
        if persistent:
            pass  # Dynamo resolves functions by module name during lazy compilation.
        elif previous is None:
            del sys.modules[name]
        else:
            sys.modules[name] = previous
    return module
