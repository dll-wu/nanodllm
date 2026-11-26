"""Build and verify a release sdist and wheel from one local Git commit."""

import argparse
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory

if __package__:
    from .check_release_dist import ROOT, check, resolve_commit
else:  # Direct invocation from scripts/.
    from check_release_dist import ROOT, check, resolve_commit


def publish(artifacts, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    files = [*artifacts.glob("*.tar.gz"), *artifacts.glob("*.whl")]
    distributions = [*outdir.glob("*.tar.gz"), *outdir.glob("*.whl")]
    if {path.name for path in distributions} - {path.name for path in files}:
        raise ValueError(f"output directory contains other distributions: {outdir}")
    # Complete both copies on the destination filesystem before replacing either release file.
    with TemporaryDirectory(dir=outdir) as staging:
        for artifact in files:
            shutil.copy2(artifact, Path(staging) / artifact.name)
        for artifact in files:
            (Path(staging) / artifact.name).replace(outdir / artifact.name)


def build(outdir, revision, no_isolation):
    commit = resolve_commit(ROOT, revision)
    print(f"Building release from Git commit {commit}", flush=True)
    with TemporaryDirectory() as staging:
        staging = Path(staging)
        source = staging / "source"
        artifacts = staging / "dist"
        snapshot = staging / "source.tar"
        subprocess.run(["git", "archive", "--format=tar", "-o", snapshot, commit], cwd=ROOT, check=True)
        with tarfile.open(snapshot) as archive:
            archive.extractall(source, filter="data")
        command = [sys.executable, "-m", "build", "--outdir", str(artifacts)]
        if no_isolation:
            command.append("--no-isolation")
        # The build frontend defaults to building the wheel from the sdist.
        subprocess.run(command, cwd=source, check=True)
        check(artifacts, commit)

        publish(artifacts, outdir)
    print(f"Release distributions written to {outdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="HEAD", help="Git commit to build (default: HEAD)")
    parser.add_argument("--outdir", default=Path("dist"), type=Path)
    parser.add_argument(
        "--no-isolation", action="store_true", help="Use build dependencies installed in this environment"
    )
    args = parser.parse_args()
    build(args.outdir, args.revision, args.no_isolation)
