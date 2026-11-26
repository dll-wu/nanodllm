"""Opt-in phase timing and CUDA peaks, independent of optimizer/RNG state."""

import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import torch

from .gpu_monitor import GpuSampler
from .telemetry import append_jsonl, write_json_atomic


class TrainingMonitor:
    def __init__(self, directory, *, device, rank, interval, step, raw_tokens, content_tokens):
        self.directory = Path(directory) / "telemetry"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.device, self.rank = device, rank
        self.cuda = device.type == "cuda"
        self.path = str(self.directory / f"phases-rank{rank}.jsonl")
        self.phase_name = "train"
        self.step, self.raw_tokens, self.content_tokens = step, raw_tokens, content_tokens
        self.peak_allocated_mib = self.peak_reserved_mib = 0.0
        self.phase_peaks = {}
        self.train_seconds = 0.0
        self.closed = False
        self.sampler = None
        self._sync()
        self._memory()  # preserve peaks from model construction before resetting
        self._reset()
        self.phase_start = self.started = self.last_log = time.perf_counter()
        self.last_train_seconds = 0.0
        self.last_raw_tokens, self.last_content_tokens = raw_tokens, content_tokens
        if self.cuda:
            uuid = getattr(torch.cuda.get_device_properties(device), "uuid", None)
            if uuid is not None:
                self.sampler = GpuSampler(self.directory / f"gpu-rank{rank}.jsonl", str(uuid), interval, self.context)
                self.sampler.start()
            else:
                append_jsonl(
                    str(self.directory / f"gpu-rank{rank}.jsonl"),
                    {
                        "schema": "nanodllm.gpu_sample.v1",
                        "status": "unavailable",
                        "reason": "device_uuid_unavailable",
                        "unix_seconds": time.time(),
                        **self.context(),
                    },
                )

    def context(self):
        return {"rank": self.rank, "phase": self.phase_name, "step": self.step, "train_tokens": self.raw_tokens}

    def progress(self, step, raw_tokens, content_tokens):
        self.step, self.raw_tokens, self.content_tokens = step, raw_tokens, content_tokens

    def _sync(self):
        if self.cuda:
            torch.cuda.synchronize(self.device)

    def _reset(self):
        if self.cuda:
            torch.cuda.reset_peak_memory_stats(self.device)

    def _memory(self):
        if not self.cuda:
            return {"peak_allocated_mib": 0.0, "peak_reserved_mib": 0.0}
        allocated = torch.cuda.max_memory_allocated(self.device) / 1024**2
        reserved = torch.cuda.max_memory_reserved(self.device) / 1024**2
        self.peak_allocated_mib = max(self.peak_allocated_mib, allocated)
        self.peak_reserved_mib = max(self.peak_reserved_mib, reserved)
        return {"peak_allocated_mib": allocated, "peak_reserved_mib": reserved}

    def lifetime_peaks(self):
        self._memory()
        return self.peak_allocated_mib, self.peak_reserved_mib

    def switch(self, name):
        self._sync()
        now = time.perf_counter()
        elapsed = now - self.phase_start
        memory = self._memory()
        if self.phase_name == "train":
            self.train_seconds += elapsed
        previous = self.phase_peaks.setdefault(self.phase_name, dict.fromkeys(memory, 0.0))
        for key, value in memory.items():
            previous[key] = max(previous[key], value)
        append_jsonl(
            self.path,
            {
                "schema": "nanodllm.training_phase.v1",
                "unix_seconds": time.time(),
                **self.context(),
                "seconds": elapsed,
                "content_tokens": self.content_tokens,
                **memory,
            },
        )
        self.phase_name = name
        self._reset()
        self.phase_start = time.perf_counter()

    @contextmanager
    def phase(self, name):
        previous = self.phase_name
        self.switch(name)
        try:
            yield
        finally:
            self.switch(previous)

    def log_fields(self):
        # Called at the existing synchronized log boundary, not on each microbatch.
        now = time.perf_counter()
        train = self.train_seconds + (now - self.phase_start if self.phase_name == "train" else 0.0)
        active, wall = train - self.last_train_seconds, now - self.last_log
        raw, content = self.raw_tokens - self.last_raw_tokens, self.content_tokens - self.last_content_tokens
        allocated, reserved = self.lifetime_peaks()
        result = {
            "unix_seconds": time.time(),
            "train_interval_seconds": active,
            "inclusive_interval_seconds": wall,
            "train_only_raw_tok_sec": raw / active if active > 0 else None,
            "train_only_content_tok_sec": content / active if active > 0 else None,
            "inclusive_raw_tok_sec": raw / wall if wall > 0 else None,
            "inclusive_content_tok_sec": content / wall if wall > 0 else None,
            "peak_vram_mb": allocated,
            "peak_reserved_mib": reserved,
        }
        self.last_log, self.last_train_seconds = now, train
        self.last_raw_tokens, self.last_content_tokens = self.raw_tokens, self.content_tokens
        return result

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.sampler is not None:
            self.sampler.close()
        self.switch("closed")
        allocated, reserved = self.lifetime_peaks()
        write_json_atomic(
            str(self.directory / f"summary-rank{self.rank}.json"),
            {
                "schema": "nanodllm.training_monitor.v1",
                "rank": self.rank,
                "step": self.step,
                "train_tokens": self.raw_tokens,
                "train_seconds": self.train_seconds,
                "wall_seconds": time.perf_counter() - self.started,
                "phase_peaks": self.phase_peaks,
                "peak_allocated_mib": allocated,
                "peak_reserved_mib": reserved,
                "scope": "This process attempt only. Phase peaks include live allocations at phase entry. GPU samples are device-wide, not process-exclusive or FLOP utilization.",
            },
        )


def initialize_monitor(session):
    config, resources = session.config, session.resources
    if config.train.telemetry_interval_seconds > 0:
        session.monitor = TrainingMonitor(
            config.train.run_dir,
            device=session.device,
            rank=session.rank,
            interval=config.train.telemetry_interval_seconds,
            step=session.start_step,
            raw_tokens=session.train_tokens,
            content_tokens=session.train_content_tokens,
        )
        resources.callback(session.monitor.close)
        session.evaluation_runtime = replace(session.evaluation_runtime, monitor=session.monitor)


def memory_snapshot(session, collect):
    memory = collect(
        device_type=session.config.runtime.device_type,
        rank=session.rank,
        local_rank=session.local_rank,
        world_size=session.world_size,
        is_ddp=session.is_ddp,
        **({"lifetime_peaks": session.monitor.lifetime_peaks()} if session.monitor is not None else {}),
    )
    peak = torch.cuda.max_memory_allocated() / 1024**2 if session.config.runtime.device_type == "cuda" else 0.0
    if session.monitor is not None:
        peak = session.monitor.lifetime_peaks()[0]
    return memory, peak
