"""Exercise the release checker on real setuptools builds from a dirty checkout."""

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from scripts.build_release_dist import publish
from scripts.check_release_dist import ROOT, check, resolve_commit


def build_checkout(checkout, outdir):
    # setuptools >=77 can build wheels itself. Exercise the backend offline even
    # when the development environment lacks the optional standalone wheel tool.
    subprocess.run(
        [sys.executable, "-m", "build", "--no-isolation", "--skip-dependency-check", "--outdir", outdir],
        cwd=checkout,
        check=True,
        stdout=subprocess.DEVNULL,
    )


@pytest.fixture(scope="module")
def builds(tmp_path_factory):
    if not (ROOT / ".git").exists():
        pytest.skip("release inventory comparisons require the committed source checkout")
    work = tmp_path_factory.mktemp("inventory")
    checkout = work / "checkout"
    revision = resolve_commit(ROOT, "HEAD")
    subprocess.run(["git", "clone", "--local", "--shared", "--no-checkout", "--quiet", ROOT, checkout], check=True)
    subprocess.run(["git", "checkout", "--detach", "--quiet", revision], cwd=checkout, check=True)

    clean = work / "clean"
    build_checkout(checkout, clean)
    profile = checkout / "recipes/profiles/tiny.json"
    profile.write_bytes(profile.read_bytes() + b"\n")
    subprocess.run(["git", "add", "--", "recipes/profiles/tiny.json"], cwd=checkout, check=True)
    changed = work / "changed"
    build_checkout(checkout, changed)
    (profile.parent / "untracked.json").write_text("{}\n")
    extra = work / "extra"
    build_checkout(checkout, extra)
    return checkout, clean, changed, extra


def test_clean_build_checks_against_commit_from_dirty_checkout(builds):
    checkout, clean, _, _ = builds
    check(clean, root=checkout)


@pytest.mark.parametrize("format_name", ["sdist", "wheel"])
@pytest.mark.parametrize("case", ["extra", "changed"])
def test_rejects_dirty_checkout_payloads(builds, tmp_path, format_name, case):
    checkout, clean, changed, extra = builds
    dirty = extra if case == "extra" else changed
    for kind, pattern in [("sdist", "*.tar.gz"), ("wheel", "*.whl")]:
        (artifact,) = (dirty if kind == format_name else clean).glob(pattern)
        shutil.copyfile(artifact, tmp_path / artifact.name)

    problem = (
        "unexpected entry: .*recipes/profiles/untracked.json"
        if case == "extra"
        else "content differs from Git: .*recipes/profiles/tiny.json"
    )
    with pytest.raises(ValueError, match=f"{format_name} {problem}"):
        check(tmp_path, root=checkout)


def test_rejects_extra_wheel_metadata(builds, tmp_path):
    checkout, clean, _, _ = builds
    shutil.copytree(clean, tmp_path, dirs_exist_ok=True)
    (wheel,) = tmp_path.glob("*.whl")
    with zipfile.ZipFile(wheel, "a") as archive:
        metadata = next(Path(name).parent for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        archive.writestr(f"{metadata.as_posix()}/local.json", "{}")
    with pytest.raises(ValueError, match=r"wheel unexpected entry: .*\.dist-info/local.json"):
        check(tmp_path, root=checkout)


def test_failed_copy_preserves_published_pair(tmp_path, monkeypatch):
    artifacts = tmp_path / "verified"
    outdir = tmp_path / "dist"
    artifacts.mkdir()
    outdir.mkdir()
    for name in ("package.tar.gz", "package.whl"):
        (artifacts / name).write_bytes(b"new release")
        (outdir / name).write_bytes(b"previous release")
    copy = shutil.copy2

    def fail_second_copy(source, target):
        if source.suffix == ".whl":
            target.write_bytes(b"partial")
            raise OSError("copy interrupted")
        return copy(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(shutil, "copy2", fail_second_copy)
        with pytest.raises(OSError, match="copy interrupted"):
            publish(artifacts, outdir)
    assert {p.name: p.read_bytes() for p in outdir.iterdir()} == {
        "package.tar.gz": b"previous release",
        "package.whl": b"previous release",
    }
    # Readers that already opened the old release keep it through replacement.
    with (outdir / "package.whl").open("rb") as previous:
        publish(artifacts, outdir)
        assert previous.read() == b"previous release"
    assert all(p.read_bytes() == b"new release" for p in outdir.iterdir())
