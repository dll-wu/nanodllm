"""Check both distribution formats against the public source tree."""

import argparse
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
SOURCE_GROUPS = {
    "configs": ("*.json",),
    "recipes": ("*.py", "*.json", "*.md"),
    "docs": ("*.md",),
    "examples": ("*.txt",),
    "nanodllm": ("*.py",),
    "scripts": ("*.py",),
    "tests": ("*.py",),
    "paper": ("*.json", "*.py", "*.tex", "*.md"),
}
TOP = {
    "AGENTS.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "DESIGN_DECISIONS.md",
    "LICENSE",
    "MANIFEST.in",
    "README.md",
    "SECURITY.md",
    "THIRD_PARTY_NOTICES.md",
    "pyproject.toml",
    "raften.toml",
    "pretrain.py",
    ".github/workflows/quality.yml",
}
VENDOR_LICENSE = "nanodllm/runtime/vendor/nanochat/LICENSE"
FORBIDDEN = {
    ".git",
    ".venv",
    "__pycache__",
    "runs",
    "results",
    "reports",
    "dist",
    "build",
    "checkpoints",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "venv",
}


def source_files(group, patterns):
    return {p.relative_to(ROOT).as_posix() for pat in patterns for p in (ROOT / group).rglob(pat) if p.is_file()}


def check(directory):
    (sdist,) = directory.glob("*.tar.gz")
    (wheel,) = directory.glob("*.whl")
    with tarfile.open(sdist, "r:gz") as archive:
        members = [PurePosixPath(f.name) for f in archive.getmembers() if f.isfile()]
    if len({p.parts[0] for p in members}) != 1:
        raise ValueError("expected one source distribution root")
    source = {PurePosixPath(*p.parts[1:]).as_posix() for p in members}
    with zipfile.ZipFile(wheel) as archive:
        binaries = {name for name in archive.namelist() if not name.endswith("/")}
    expected = TOP | {VENDOR_LICENSE}
    for group, patterns in SOURCE_GROUPS.items():
        expected.update(source_files(group, patterns))
    missing = expected - source
    required_wheel = (
        {"pretrain.py", VENDOR_LICENSE}
        | source_files("nanodllm", ("*.py",))
        | source_files("paper", ("*.json", "*.py", "*.tex"))
        | source_files("recipes", ("*.json", "*.py"))
    )
    missing |= required_wheel - binaries
    if missing:
        raise ValueError(f"distribution files missing: {sorted(missing)}")
    for name in source | binaries:
        p = PurePosixPath(name)
        if FORBIDDEN.intersection(p.parts) or p.suffix.lower() in {".pt", ".pth", ".parquet", ".ckpt", ".pyc"}:
            raise ValueError(f"unexpected local artifact: {name}")
        if p.parts[0] == "data":
            raise ValueError(f"unexpected raw corpus: {name}")
    if any(PurePosixPath(name).parts[0] in {"configs", "tests", "docs", "scripts", "examples"} for name in binaries):
        raise ValueError("source-only directories at wheel root")
    print(f"Verified {len(source)} source files and {len(binaries)} wheel entries.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    check(parser.parse_args().directory)
