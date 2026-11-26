"""Append-only artifacts produced by generation evaluation."""

from __future__ import annotations

import json
import os
from contextlib import ExitStack
from types import TracebackType
from typing import Any, TextIO

GENERATION_SAMPLE_SCHEMA = "nanodllm.generation_sample.v1"


class GenerationArtifacts:
    """Own preview logging and complete structured generation records."""

    def __init__(
        self,
        run_dir: str | None,
        tokenizer: Any,
        requested_length: int,
        step: int | None,
        sample_label: str | None,
    ) -> None:
        self.run_dir = run_dir
        self.tokenizer = tokenizer
        self.requested_length = int(requested_length)
        self.step = step
        self.sample_label = sample_label or ""
        self.samples_path = ""
        self._stack: ExitStack | None = None
        self._log_file: TextIO | None = None
        self._samples_file: TextIO | None = None
        self._records_written = 0
        self._entered = False

    def __enter__(self) -> GenerationArtifacts:
        if self._entered:
            raise RuntimeError("generation artifacts context cannot be entered twice")
        stack = ExitStack()
        try:
            if self.run_dir is not None:
                os.makedirs(self.run_dir, exist_ok=True)
                self._log_file = stack.enter_context(open(os.path.join(self.run_dir, "gen_samples.log"), "a"))
                self.samples_path = os.path.join(self.run_dir, "generation_samples.jsonl")
                self._samples_file = stack.enter_context(open(self.samples_path, "a", encoding="utf-8"))
        except Exception:
            stack.close()
            raise
        self._entered = True
        self._stack = stack
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        if self._stack is None:
            raise RuntimeError("generation artifacts context was not entered")
        stack = self._stack
        self._stack = None
        return stack.__exit__(exc_type, exc_value, traceback)

    def log(self, message: str) -> None:
        if self._log_file is None:
            print(message)
            return
        self._log_file.write(message + "\n")
        self._log_file.flush()

    def record(
        self,
        batch_index: int,
        batch_item_index: int,
        raw_token_ids: list[int],
        truncated_token_ids: list[int],
        has_eos: bool,
        content_token_count: int,
        content_byte_count: int,
    ) -> None:
        if self._samples_file is None:
            return
        text = self.tokenizer.decode(truncated_token_ids)
        record = {
            "schema": GENERATION_SAMPLE_SCHEMA,
            "step": self.step,
            "sample_label": self.sample_label,
            "sample_index": self._records_written,
            "batch_index": int(batch_index),
            "batch_item_index": int(batch_item_index),
            "requested_length": self.requested_length,
            "raw_token_ids": raw_token_ids,
            "eos_truncated_token_ids": truncated_token_ids,
            "text": text,
            "has_eos": bool(has_eos),
            "eos_position": int(len(truncated_token_ids) - 1) if has_eos else None,
            "empty": not text.strip(),
            "content_token_count": int(content_token_count),
            "content_byte_count": int(content_byte_count),
        }
        self._samples_file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self._records_written += 1

    def flush_samples(self) -> None:
        if self._samples_file is not None:
            self._samples_file.flush()

    def sample_metrics(self, sample_count: int) -> dict[str, int | str]:
        if not self.samples_path:
            return {}
        if self._records_written != int(sample_count):
            raise RuntimeError(
                f"generation sample record count {self._records_written} differs from evaluated count {sample_count}"
            )
        return {
            "gen_sample_records_schema": GENERATION_SAMPLE_SCHEMA,
            "gen_sample_records_written": int(sample_count),
            "gen_samples_path": self.samples_path,
        }
