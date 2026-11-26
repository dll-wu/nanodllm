"""Informational repository identity without exposing untracked file names."""

from pathlib import Path


def _live_source_provenance() -> dict[str, bool | int | None | str]:
    """Inspect Git metadata when this module comes directly from its checkout."""

    unavailable: dict[str, bool | int | None | str] = {
        "source_repository_detected": False,
        "source_git_status_available": False,
        "source_git_commit": "",
        "source_tracked_worktree_clean": None,
        "source_untracked_path_count": 0,
    }
    try:
        import subprocess

        source_path = Path(__file__).resolve()
        repo_hint = source_path.parents[1]
        repo_root = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={repo_hint}",
                "-C",
                str(repo_hint),
                "rev-parse",
                "--show-toplevel",
            ],
            capture_output=True,
            text=True,
        )
        if repo_root.returncode != 0:
            return unavailable

        repo_root_path = Path(repo_root.stdout.strip()).resolve()
        if (repo_root_path / "nanodllm" / "source_provenance.py").resolve() != source_path:
            return unavailable

        commit = subprocess.run(
            ["git", "-c", f"safe.directory={repo_root_path}", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=repo_root_path,
        )
        status = subprocess.run(
            ["git", "-c", f"safe.directory={repo_root_path}", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=repo_root_path,
        )
        if status.returncode != 0:
            return {
                **unavailable,
                "source_repository_detected": True,
                "source_git_commit": commit.stdout.strip() if commit.returncode == 0 else "",
            }

        lines = [line for line in status.stdout.splitlines() if line.strip()]
        tracked_dirty = any(not line.startswith("?? ") for line in lines)
        untracked_path_count = sum(1 for line in lines if line.startswith("?? "))
        return {
            "source_repository_detected": True,
            "source_git_status_available": True,
            "source_git_commit": commit.stdout.strip() if commit.returncode == 0 else "",
            "source_tracked_worktree_clean": not tracked_dirty,
            "source_untracked_path_count": untracked_path_count,
        }
    except Exception:
        return unavailable


def source_provenance_metrics() -> dict[str, bool | int | None | list[str] | str]:
    """Return informational source facts that never gate an otherwise successful run."""

    provenance = _live_source_provenance()
    repository_detected = bool(provenance["source_repository_detected"])
    status_available = bool(provenance["source_git_status_available"])
    tracked_clean = provenance["source_tracked_worktree_clean"]
    untracked_path_count = int(provenance["source_untracked_path_count"])
    warnings: list[str] = []
    if not repository_detected:
        warnings.append("Git checkout metadata is unavailable; this may be an installed distribution")
    elif not status_available:
        warnings.append("Git checkout detected, but worktree status could not be inspected")
    else:
        if tracked_clean is False:
            warnings.append("Git checkout contains tracked modifications")
        if untracked_path_count:
            warnings.append("Git checkout contains untracked paths")
    worktree_dirty = None if not status_available else tracked_clean is False or untracked_path_count > 0
    return {
        "status": "ok",
        **provenance,
        "source_worktree_dirty": worktree_dirty,
        "source_provenance_is_informational": True,
        "source_provenance_warnings": warnings,
    }
