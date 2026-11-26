"""Isolate adapter updates while preserving accumulated backbone gradients."""

from contextlib import contextmanager

import torch


@contextmanager
def _isolate_token_edit_adapter_gradients(raw_model):
    """Keep only adapter gradients from one T2T backward pass."""

    preserved_gradient_names = {
        name
        for name, parameter in raw_model.named_parameters()
        if not name.startswith("token_edit_adapters.") and parameter.grad is not None
    }
    handles = [
        parameter.register_hook(lambda gradient: torch.zeros_like(gradient))
        for name, parameter in raw_model.named_parameters()
        if not name.startswith("token_edit_adapters.") and parameter.requires_grad
    ]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        for name, parameter in raw_model.named_parameters():
            if not name.startswith("token_edit_adapters.") and name not in preserved_gradient_names:
                parameter.grad = None
