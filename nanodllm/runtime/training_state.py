from contextlib import ExitStack

import torch

from .config import ExperimentConfig
from .training_features import _is_causal, _uses_content_budget_segmented_packing
from .training_progress import TrainerProgress


class TrainingSession:
    """Mutable state shared by training lifecycle phases."""

    def __init__(self, config: ExperimentConfig, resources: ExitStack) -> None:
        self.config = config
        self.resources = resources
        self.monitor = None
        self.progress = TrainerProgress.fresh(int(config.model.max_seq_len))


def _content_byte_counts(session, x_batch: torch.Tensor, attention_mask_batch: torch.Tensor) -> tuple[int, int]:
    byte_values = session.training_token_bytes[x_batch]
    content_mask = attention_mask_batch & (byte_values > 0)
    content_tokens = int(content_mask.sum().item())
    content_bytes = int(byte_values[attention_mask_batch].sum().item())
    return (content_tokens, content_bytes)


def _record_ar_boundary_batch(
    session,
    x_batch: torch.Tensor,
    loss_mask_batch: torch.Tensor,
    attention_mask_batch: torch.Tensor,
    position_ids_batch: torch.Tensor | None = None,
    document_boundary_mask_batch: torch.Tensor | None = None,
    document_start_mask_batch: torch.Tensor | None = None,
) -> None:
    if not (
        _is_causal(session.config)
        or _uses_content_budget_segmented_packing(session.config)
        or session.config.data.prepend_bos
        or session.eos_supervision_audit_enabled
    ):
        return
    session.progress.lineage.ar_boundary_counts["microbatches"] += 1
    session.progress.lineage.ar_boundary_counts["valid_tokens"] += int(attention_mask_batch.sum().item())
    b, t = x_batch.shape
    row_valid = attention_mask_batch.sum(dim=1)
    row_lossable = (loss_mask_batch & attention_mask_batch).sum(dim=1)
    row_count = int(row_valid.numel())
    session.progress.lineage.row_telemetry["rows"] += row_count
    session.progress.lineage.row_telemetry["valid_tokens_sum"] += int(row_valid.sum().item())
    session.progress.lineage.row_telemetry["lossable_tokens_sum"] += int(row_lossable.sum().item())
    session.progress.lineage.row_telemetry["pad_tokens"] += int((~attention_mask_batch).sum().item())
    session.progress.lineage.row_telemetry["total_row_tokens"] += b * t
    if row_valid.numel():
        row_valid_min = int(row_valid.min().item())
        row_valid_max = int(row_valid.max().item())
        row_lossable_min = int(row_lossable.min().item())
        row_lossable_max = int(row_lossable.max().item())
        if session.progress.lineage.row_telemetry["rows"] == row_count:
            session.progress.lineage.row_telemetry["valid_tokens_min"] = row_valid_min
            session.progress.lineage.row_telemetry["lossable_tokens_min"] = row_lossable_min
        else:
            session.progress.lineage.row_telemetry["valid_tokens_min"] = min(
                int(session.progress.lineage.row_telemetry["valid_tokens_min"]), row_valid_min
            )
            session.progress.lineage.row_telemetry["lossable_tokens_min"] = min(
                int(session.progress.lineage.row_telemetry["lossable_tokens_min"]), row_lossable_min
            )
        session.progress.lineage.row_telemetry["valid_tokens_max"] = max(
            int(session.progress.lineage.row_telemetry["valid_tokens_max"]), row_valid_max
        )
        session.progress.lineage.row_telemetry["lossable_tokens_max"] = max(
            int(session.progress.lineage.row_telemetry["lossable_tokens_max"]), row_lossable_max
        )
    if session.special_token_ids:
        special_mask = torch.zeros_like(attention_mask_batch, dtype=torch.bool)
        for token_id in session.special_token_ids:
            special_mask |= x_batch == token_id
        special_mask &= attention_mask_batch
        session.progress.lineage.ar_boundary_counts["special_tokens"] += int(special_mask.sum().item())
        session.progress.lineage.ar_boundary_counts["special_loss_tokens"] += int(
            (special_mask & loss_mask_batch).sum().item()
        )
    if session.eos_token_id is not None:
        eos_mask = (x_batch == int(session.eos_token_id)) & attention_mask_batch
        session.progress.lineage.ar_boundary_counts["eos_tokens"] += int(eos_mask.sum().item())
        session.progress.lineage.ar_boundary_counts["eos_loss_tokens"] += int((eos_mask & loss_mask_batch).sum().item())
        session.progress.lineage.row_telemetry["eos_rows"] += int(eos_mask.any(dim=1).sum().item())
        session.progress.lineage.row_telemetry["eos_tokens"] += int(eos_mask.sum().item())
        session.progress.lineage.row_telemetry["eos_loss_tokens"] += int((eos_mask & loss_mask_batch).sum().item())
    if session.bos_token_id is not None:
        bos_mask = (x_batch == int(session.bos_token_id)) & attention_mask_batch
        bos_count = int(bos_mask.sum().item())
        session.progress.lineage.ar_boundary_counts["bos_tokens"] += bos_count
        session.progress.lineage.ar_boundary_counts["bos_loss_tokens"] += int((bos_mask & loss_mask_batch).sum().item())
        if bos_count:
            session.progress.lineage.ar_boundary_counts["microbatches_with_bos"] += 1
    if position_ids_batch is not None and document_boundary_mask_batch is not None:
        if document_start_mask_batch is not None:
            docs_per_row = (document_start_mask_batch & attention_mask_batch).sum(dim=1).float()
        else:
            docs_per_row = ((position_ids_batch == 0) & attention_mask_batch).sum(dim=1).float()
        session.progress.lineage.boundary_telemetry["documents_per_row_sum"] += int(docs_per_row.sum().item())
        session.progress.lineage.boundary_telemetry["documents_per_row_count"] += b
        causal_mask = torch.tril(torch.ones(t, t, dtype=torch.bool, device=x_batch.device))
        causal_mask = causal_mask.unsqueeze(0).expand(b, -1, -1)
        valid_q = attention_mask_batch[:, :, None]
        valid_k = attention_mask_batch[:, None, :]
        valid_pairs = valid_q & valid_k
        allowed_pairs = valid_pairs & document_boundary_mask_batch
        total_pairs = int(valid_pairs.sum().item())
        allowed_pair_count = int(allowed_pairs.sum().item())
        session.progress.lineage.boundary_telemetry["valid_pairs_sum"] += total_pairs
        session.progress.lineage.boundary_telemetry["allowed_pairs_sum"] += allowed_pair_count
        session.progress.lineage.boundary_telemetry["segment_blocked_pairs_sum"] += total_pairs - allowed_pair_count
        valid_causal = causal_mask & valid_q & valid_k
        total_causal = int(valid_causal.sum().item())
        allowed_causal = int((valid_causal & document_boundary_mask_batch).sum().item())
        session.progress.lineage.boundary_telemetry["causal_pairs_sum"] += total_causal
        session.progress.lineage.boundary_telemetry["blocked_pairs_sum"] += total_causal - allowed_causal


def _current_trainer_state(session: TrainingSession) -> dict[str, object]:
    return session.progress.to_checkpoint()
