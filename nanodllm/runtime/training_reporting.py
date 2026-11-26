from __future__ import annotations

from .training_report_context import build_final_context_fragments_v1
from .training_report_inputs import FinalReportInputs
from .training_report_measurements import build_final_measurement_fragments_v1
from .training_report_schema import ReportFragment, merge_report_fragments


def build_final_metrics_v1(inputs: FinalReportInputs) -> dict[str, object]:
    """Build the flat version-1 final report from explicitly owned fragments."""

    return merge_report_fragments(
        ReportFragment("source_provenance", inputs.source_metrics),
        *build_final_context_fragments_v1(inputs),
        *build_final_measurement_fragments_v1(inputs),
        ReportFragment("final_evaluation", inputs.final_eval_metrics),
        ReportFragment("final_lm_head_row_norms", inputs.row_norm_metrics),
    )
