from __future__ import annotations

from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from .config import ExperimentConfig
    from .data import ByteTokenizer, PretrainedHFTokenizer
    from .model import DiffusionConfig, DiffusionTransformer
    from .training_progress import TrainerProgress
    from .training_state import TrainingSession


@dataclass(frozen=True, slots=True)
class FinalReportInputs:
    """Explicit read-only view of the state required to build the final report."""

    architecture_fingerprint: Mapping[str, object]
    clean_label_smoothing_metrics: Mapping[str, object]
    config: ExperimentConfig
    counts: Mapping[str, int]
    cpu_threading: Mapping[str, object]
    data_identity_summary: Mapping[str, object]
    distributed_cuda_memory: Mapping[str, object]
    embedding_head_init_diagnostics_path: str
    entrypoint_mode: str
    eos_supervision_startup_metrics: Mapping[str, object]
    eos_supervision_train_metrics: Mapping[str, object]
    fixed_ce_trace_path: str
    fixed_ce_trace_steps: Sequence[int]
    flops_per_token: int
    grad_accum_steps: int
    grad_clip_enabled: bool
    grad_clip_threshold: float
    initial_state_artifact: Mapping[str, object]
    lr_schedule: Mapping[str, object]
    model_config: DiffusionConfig
    parameter_accounting: Mapping[str, object]
    progress: TrainerProgress
    raw_model: DiffusionTransformer
    record_embedding_head_init_diagnostics: bool
    resume_next_batch_digest_verified: bool
    resume_state: Mapping[str, object] | None
    source_checkpoint_load_receipt: Mapping[str, object]
    session_train_content_bytes: int
    session_train_content_tokens: int
    session_train_tokens: int
    session_train_valid_tokens: int
    special_token_ids: Set[int]
    start_step: int
    token_class_metadata: Mapping[str, object] | None
    token_edit_class_metadata: Mapping[str, object] | None
    tokenizer: ByteTokenizer | PretrainedHFTokenizer
    total_batch_size: int
    total_budget_tokens: int
    total_steps: int
    train_content_bytes: int
    train_content_tokens: int
    train_tokens: int
    train_valid_tokens: int
    training_seconds: float
    world_size: int
    source_metrics: Mapping[str, object]
    peak_vram_mb: float
    final_eval_metrics: Mapping[str, object]
    row_norm_metrics: Mapping[str, object]

    @classmethod
    def from_session(
        cls,
        session: TrainingSession,
        *,
        source_metrics: Mapping[str, object],
        peak_vram_mb: float,
        final_eval_metrics: Mapping[str, object],
        row_norm_metrics: Mapping[str, object],
        entrypoint_mode: str,
    ) -> Self:
        """Capture the report dependencies without modifying the training session."""

        return cls(
            architecture_fingerprint=session.architecture_fingerprint,
            clean_label_smoothing_metrics=session.clean_label_smoothing_metrics,
            config=session.config,
            counts=session.counts,
            cpu_threading=session.cpu_threading,
            data_identity_summary=session.data_identity_summary,
            distributed_cuda_memory=session.distributed_cuda_memory,
            embedding_head_init_diagnostics_path=session.embedding_head_init_diagnostics_path,
            entrypoint_mode=entrypoint_mode,
            eos_supervision_startup_metrics=session.eos_supervision_startup_metrics,
            eos_supervision_train_metrics=session.eos_supervision_train_metrics,
            fixed_ce_trace_path=session.fixed_ce_trace_path,
            fixed_ce_trace_steps=session.fixed_ce_trace_steps,
            flops_per_token=session.flops_per_token,
            grad_accum_steps=session.grad_accum_steps,
            grad_clip_enabled=session.grad_clip_enabled,
            grad_clip_threshold=session.grad_clip_threshold,
            initial_state_artifact=session.initial_state_artifact,
            lr_schedule=session.lr_schedule,
            model_config=session.model_config,
            parameter_accounting=session.parameter_accounting,
            progress=session.progress,
            raw_model=session.raw_model,
            record_embedding_head_init_diagnostics=session.record_embedding_head_init_diagnostics,
            resume_next_batch_digest_verified=session.resume_next_batch_digest_verified,
            resume_state=session.resume_state,
            source_checkpoint_load_receipt=session.source_checkpoint_load_receipt,
            session_train_content_bytes=session.session_train_content_bytes,
            session_train_content_tokens=session.session_train_content_tokens,
            session_train_tokens=session.session_train_tokens,
            session_train_valid_tokens=session.session_train_valid_tokens,
            special_token_ids=session.special_token_ids,
            start_step=session.start_step,
            token_class_metadata=session.token_class_metadata,
            token_edit_class_metadata=session.token_edit_class_metadata,
            tokenizer=session.tokenizer,
            total_batch_size=session.total_batch_size,
            total_budget_tokens=session.total_budget_tokens,
            total_steps=session.total_steps,
            train_content_bytes=session.train_content_bytes,
            train_content_tokens=session.train_content_tokens,
            train_tokens=session.train_tokens,
            train_valid_tokens=session.train_valid_tokens,
            training_seconds=session.training_seconds,
            world_size=session.world_size,
            source_metrics=source_metrics,
            peak_vram_mb=peak_vram_mb,
            final_eval_metrics=final_eval_metrics,
            row_norm_metrics=row_norm_metrics,
        )
