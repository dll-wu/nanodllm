"""Verify the sdist and wheel against the public files in a Git commit."""

import argparse
import hashlib
import stat
import subprocess
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

import tomllib

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SUFFIXES = {
    "configs": {".json"},
    "recipes": {".py", ".json", ".md"},
    "docs": {".md"},
    "examples": {".txt"},
    "nanodllm": {".py"},
    "scripts": {".py"},
    "tests": {".py"},
    "paper": {".json", ".py", ".tex", ".md"},
}
WHEEL_SUFFIXES = {"nanodllm": {".py"}, "recipes": {".json", ".py"}, "paper": {".json", ".py", ".tex"}}
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
    "resolved-configs",
    "resolved-paper-configs",
    "dist",
    "build",
    "checkpoints",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "venv",
}


def resolve_commit(root, revision):
    return subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}"], cwd=root, text=True
    ).strip()


def inventories(root, commit):
    """Map archive paths to Git blob IDs. Only named backend outputs have no blob."""
    project = tomllib.loads(subprocess.check_output(["git", "show", f"{commit}:pyproject.toml"], cwd=root).decode())[
        "project"
    ]
    stem = f"nano_dllm-{project['version']}"
    source = {}
    wheel = {}
    tree = subprocess.check_output(["git", "ls-tree", "-rz", "--full-tree", commit], cwd=root)
    for entry in tree.rstrip(b"\0").split(b"\0"):
        header, raw_name = entry.split(b"\t", 1)
        mode, _, blob = header.decode().split()
        name = raw_name.decode()
        path = PurePosixPath(name)
        group = path.parts[0]
        if name not in TOP and name != VENDOR_LICENSE and path.suffix not in SOURCE_SUFFIXES.get(group, ()):
            continue
        if FORBIDDEN.intersection(path.parts):
            raise ValueError(f"unexpected local artifact: {name}")
        if mode == "120000":
            # Older commits have in-tree config aliases that setuptools copies as regular files.
            target = subprocess.check_output(
                ["git", "cat-file", "--batch-check", "--follow-symlinks"],
                input=f"{commit}:{name}\n",
                cwd=root,
                text=True,
            ).split()
            if len(target) != 3 or target[1] != "blob":
                raise ValueError(f"expected a committed file behind {name}")
            blob = target[0]
        elif mode not in {"100644", "100755"}:
            raise ValueError(f"unsupported Git entry: {name}")
        source[name] = blob
        if name in {"pretrain.py", VENDOR_LICENSE} or path.suffix in WHEEL_SUFFIXES.get(group, ()):
            wheel[name] = blob
    missing = (TOP | {VENDOR_LICENSE}) - source.keys()
    if missing:
        raise ValueError(f"required public files missing from Git: {sorted(missing)}")

    metadata = f"{stem}.dist-info"
    for license_name in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
        wheel[f"{metadata}/licenses/{license_name}"] = source[license_name]
    for name in ("METADATA", "WHEEL", "RECORD", "entry_points.txt", "top_level.txt"):
        wheel[f"{metadata}/{name}"] = None
    for name in ("PKG-INFO", "setup.cfg", "nano_dllm.egg-info/SOURCES.txt"):
        source[name] = None
    return {f"{stem}/{name}": blob for name, blob in source.items()}, wheel


def check_archive_owners(archive):
    owner_fields = {"uid", "gid", "uname", "gname"}
    if owner_fields.intersection(archive.pax_headers):
        raise ValueError("source archive contains global ownership metadata")
    for member in archive.getmembers():
        if member.uid or member.gid or member.uname or member.gname or owner_fields.intersection(member.pax_headers):
            raise ValueError(f"source archive contains ownership metadata on {member.name}")


def check_inventory(label, entries, expected):
    directories = {str(p) for name in expected for p in PurePosixPath(name).parents if p != PurePosixPath(".")}
    seen = set()
    for name, is_directory in entries:
        if name in seen:
            raise ValueError(f"{label} duplicate entry: {name}")
        seen.add(name)
        if name not in (directories if is_directory else expected):
            raise ValueError(f"{label} unexpected entry: {name}")
    missing = expected.keys() - seen
    if missing:
        raise ValueError(f"{label} missing entries: {sorted(missing)}")


def check_blob(label, name, stream, size, expected):
    digest = hashlib.new("sha1" if len(expected) == 40 else "sha256")
    digest.update(f"blob {size}\0".encode())
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"{label} content differs from Git: {name}")


def check(directory, revision="HEAD", *, root=ROOT):
    commit = resolve_commit(root, revision)
    source, binaries = inventories(root, commit)
    (sdist,) = directory.glob("*.tar.gz")
    (wheel,) = directory.glob("*.whl")
    with tarfile.open(sdist, "r:gz") as archive:
        check_archive_owners(archive)
        members = archive.getmembers()
        for member in members:
            if member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE):
                raise ValueError(f"sdist unsupported entry type: {member.name}")
        check_inventory("sdist", ((m.name, m.isdir()) for m in members), source)
        for member in members:
            if blob := source.get(member.name):
                with archive.extractfile(member) as stream:
                    check_blob("sdist", member.name, stream, member.size, blob)

    with zipfile.ZipFile(wheel) as archive:
        entries = archive.infolist()
        for entry in entries:
            kind = stat.S_IFMT(entry.external_attr >> 16)
            if kind not in (0, stat.S_IFDIR if entry.is_dir() else stat.S_IFREG):
                raise ValueError(f"wheel unsupported entry type: {entry.filename}")
        check_inventory("wheel", ((e.filename.removesuffix("/"), e.is_dir()) for e in entries), binaries)
        for entry in entries:
            if blob := binaries.get(entry.filename):
                with archive.open(entry) as stream:
                    check_blob("wheel", entry.filename, stream, entry.file_size, blob)
    print(f"Verified {len(source)} sdist files and {len(binaries)} wheel files against {commit}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--revision", default="HEAD", help="Git commit to check against (default: HEAD)")
    args = parser.parse_args()
    check(args.directory, args.revision)
