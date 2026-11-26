"""Source-checkout entry point for corrected task scoring."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nanodllm.tasks.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
