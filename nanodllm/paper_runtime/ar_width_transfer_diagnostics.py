"""Sparse, observational AdamW summaries at frozen width-transfer updates."""

from contextlib import contextmanager
from pathlib import Path

import torch

from .ar_width_mup import active
from .telemetry import append_jsonl

MARKERS = (0, 1, 8, 176, 1760, 3520)


def _summary(tensor):
    if tensor is None:
        return None
    value = tensor.detach().double()
    return {
        "rms": float(value.square().mean().sqrt()),
        "max_abs": float(value.abs().max()),
        "finite": bool(torch.isfinite(value).all()),
        "dtype": str(tensor.dtype),
    }


class AdamDiagnostics:
    """Observe the actual optimizer step, preserving tensors, RNG and counters."""

    def __init__(self, model, optimizer, path, start_update=0):
        self.model, self.optimizer, self.path = model, optimizer, path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.update = int(start_update)
        self.before = {}

    def before_step(self, _optimizer, _args, _kwargs):
        if self.update + 1 in MARKERS:
            self.before = {name: p.detach().clone() for name, p in self.model.named_parameters()}

    def after_step(self, _optimizer, _args, _kwargs):
        self.update += 1
        if self.update in MARKERS:
            self.record()
        self.before = {}

    @torch.no_grad()
    def record(self):
        groups = {id(p): group for group in self.optimizer.param_groups for p in group["params"]}
        parameters = []
        for name, parameter in self.model.named_parameters():
            state = self.optimizer.state.get(parameter, {})
            group = groups[id(parameter)]
            step = int(state.get("step", 0))
            second = state.get("exp_avg_sq")
            denominator, fraction = None, None
            if second is not None and step > 0:
                # Diagnostic algebra in FP64; does not replace optimizer arithmetic.
                root = (second.double() / (1 - group["betas"][1] ** step)).sqrt()
                denominator = _summary(root + group["eps"])
                fraction = float((root <= group["eps"]).double().mean())
            parameters.append(
                {
                    "name": name,
                    "role": group.get("parameterization_role"),
                    "adam_step": step,
                    "lr": group["lr"],
                    "epsilon": group["eps"],
                    "weight_decay": group["weight_decay"],
                    "parameter": _summary(parameter),
                    "gradient": _summary(parameter.grad),
                    "update": _summary(parameter - self.before[name]) if name in self.before else None,
                    "first_moment": _summary(state.get("exp_avg")),
                    "second_moment": _summary(second),
                    "bias_corrected_denominator": denominator,
                    "epsilon_dominated_fraction": fraction,
                }
            )
        append_jsonl(
            str(self.path),
            {
                "schema": "nanodllm.ar_width_transfer_optimizer_diagnostics.v1",
                "completed_updates": self.update,
                "position": "post_optimizer_step" if self.update else "pre_first_update",
                "parameters": parameters,
            },
        )


@contextmanager
def diagnostic_scope(session):
    if not active(session.config):
        yield
        return
    if len(session.optimizers) != 1 or session.rank != 0 or session.world_size != 1:
        raise ValueError("width-transfer diagnostics require one optimizer and one process")
    optimizer = session.optimizers[0]
    observer = AdamDiagnostics(
        session.raw_model,
        optimizer,
        Path(session.config.train.run_dir) / "telemetry" / "ar-width-transfer.jsonl",
        session.start_step,
    )
    if session.start_step == 0:
        observer.record()
    pre = optimizer.register_step_pre_hook(observer.before_step)
    post = optimizer.register_step_post_hook(observer.after_step)
    try:
        yield
    finally:
        pre.remove()
        post.remove()
