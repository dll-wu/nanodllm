"""Find versioned manifests in a checkout or the installed distribution."""

import sysconfig
from pathlib import Path


def release_asset(relative_name: str) -> Path:
    source = Path(__file__).resolve().parents[1] / relative_name
    if source.exists():
        return source
    return Path(sysconfig.get_path("data")) / "share" / "nano-dllm" / relative_name
