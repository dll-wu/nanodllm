"""Optional, bounded nvidia-smi sampling; never owns CUDA or training state."""

import csv
import math
import subprocess
import threading
import time
from pathlib import Path

from .telemetry import append_jsonl

FIELDS = ("utilization.gpu", "memory.used", "power.draw", "temperature.gpu", "clocks.sm", "clocks.mem")
NAMES = ("gpu_busy_percent", "device_memory_used_mib", "power_w", "temperature_c", "sm_clock_mhz", "memory_clock_mhz")


def parse_sample(output: str) -> dict:
    rows = list(csv.reader(output.strip().splitlines()))
    if len(rows) != 1 or len(rows[0]) != len(FIELDS):
        raise ValueError("expected one GPU sample with six fields")
    values = {}
    for name, raw in zip(NAMES, rows[0], strict=True):
        try:
            value = float(raw.strip())
        except ValueError:
            value = None
        values[name] = value if value is not None and math.isfinite(value) else None
    return values


class GpuSampler:
    """A missing metric is null; failed device queries stop sampling, not training."""

    def __init__(self, path: Path, device_uuid: str, interval: float, context):
        self.path, self.interval, self.context = path, interval, context
        # PyTorch exposes a bare _CUuuid on some supported versions; nvidia-smi
        # accepts the physical-GPU spelling. Preserve already-qualified UUIDs.
        self.device_uuid = device_uuid if device_uuid.startswith(("GPU-", "MIG-")) else "GPU-" + device_uuid
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="gpu-telemetry", daemon=True)

    def start(self):
        self.thread.start()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=4)

    def _run(self):
        while not self.stop.is_set():
            record = {"schema": "nanodllm.gpu_sample.v1", "unix_seconds": time.time(), **self.context()}
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "-i",
                        self.device_uuid,
                        "--query-gpu=" + ",".join(FIELDS),
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=3,
                    check=True,
                )
                record.update(status="ok", **parse_sample(result.stdout))
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                record.update(status="unavailable", reason=type(exc).__name__, **dict.fromkeys(NAMES))
            try:
                append_jsonl(str(self.path), record)
            except OSError:
                return
            if record["status"] != "ok":
                return
            self.stop.wait(self.interval)
