"""Step-aware discovery of resumable checkpoint files."""

from pathlib import Path


def _checkpoint_step(path: Path) -> int | None:
    prefix = "model_step"
    suffix = ".pt"
    name = path.name
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    step_text = name[len(prefix) : -len(suffix)]
    return int(step_text) if step_text.isdigit() else None


def _step_checkpoint_candidates(run_dir: Path) -> list[tuple[int, int, Path]]:
    candidates: list[tuple[int, int, Path]] = []
    for location_priority, directory in enumerate((run_dir / "checkpoints", run_dir)):
        if not directory.is_dir():
            continue
        for path in directory.glob("model_step*.pt"):
            step = _checkpoint_step(path)
            if step is not None and path.is_file():
                candidates.append((step, location_priority, path))
    return candidates


def _latest_checkpoint_in_run_dir(run_dir: str | Path) -> Path | None:
    run_path = Path(run_dir)
    canonical_alias = next(
        (path for path in (run_path / "model_latest.pt", run_path / "model.pt") if path.is_file()),
        None,
    )
    step_candidates = _step_checkpoint_candidates(run_path)
    if not step_candidates:
        return canonical_alias
    latest_step_number, _, latest_step = max(step_candidates, key=lambda candidate: (candidate[0], candidate[1]))
    if canonical_alias is None:
        return latest_step
    for step, _, step_path in step_candidates:
        if canonical_alias.samefile(step_path):
            return canonical_alias if step == latest_step_number else latest_step
    if latest_step.stat().st_mtime_ns > canonical_alias.stat().st_mtime_ns:
        return latest_step
    return canonical_alias


def _resolve_resume_path(run_dir: str, resume_from: str) -> str | None:
    if not resume_from:
        return None
    if resume_from == "latest":
        candidate = _latest_checkpoint_in_run_dir(run_dir)
        if candidate is not None:
            return str(candidate)
        raise FileNotFoundError(f"no checkpoint found for train.resume_from='latest' in {run_dir}")
    return resume_from
