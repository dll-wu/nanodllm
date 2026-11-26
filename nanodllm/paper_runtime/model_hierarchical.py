"""Hierarchical vocabulary likelihoods over a model-owned class partition."""

import torch
from torch.nn import functional as F


class HierarchicalOutput:
    """Behavior only: parameter and buffer registration remain in the model."""

    def uses_hierarchical_output(self) -> bool:
        return self.output_factorization == "hierarchical_frequency_decile"

    def _class_token_ids(self, class_idx: int) -> torch.Tensor:
        return (self.token_class_ids == int(class_idx)).nonzero(as_tuple=True)[0]

    def hierarchical_token_log_probs(
        self,
        class_logits: torch.Tensor,
        token_logits: torch.Tensor,
    ) -> torch.Tensor:
        if not self.uses_hierarchical_output():
            return token_logits
        class_log_probs = F.log_softmax(class_logits.float(), dim=-1)
        token_logits = token_logits.float()
        log_probs = torch.empty_like(token_logits)
        copy_dim = log_probs.dim() - 1
        for class_idx in range(self.output_class_count):
            class_token_ids = self._class_token_ids(class_idx)
            selected = token_logits.index_select(-1, class_token_ids)
            selected = selected - torch.logsumexp(selected, dim=-1, keepdim=True)
            selected = selected + class_log_probs[..., class_idx].unsqueeze(-1)
            log_probs.index_copy_(copy_dim, class_token_ids, selected)
        return log_probs

    def hierarchical_ce(
        self,
        class_logits: torch.Tensor,
        token_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.uses_hierarchical_output():
            raise ValueError("hierarchical_ce requires hierarchical output factorization")
        targets = targets.to(dtype=torch.long)
        original_shape = targets.shape
        flat_targets = targets.reshape(-1)
        class_targets = self.token_class_ids.index_select(0, flat_targets)
        flat_class_logits = class_logits.reshape(-1, self.output_class_count).float()
        flat_token_logits = token_logits.reshape(-1, token_logits.shape[-1]).float()
        class_ce = F.cross_entropy(flat_class_logits, class_targets, reduction="none")
        within_ce = torch.empty_like(class_ce)
        for class_idx in range(self.output_class_count):
            row_ids = (class_targets == class_idx).nonzero(as_tuple=True)[0]
            if row_ids.numel() == 0:
                continue
            class_token_ids = self._class_token_ids(class_idx)
            class_logits_slice = flat_token_logits.index_select(0, row_ids).index_select(1, class_token_ids)
            class_targets_slice = self.token_local_ids.index_select(0, flat_targets.index_select(0, row_ids))
            within_values = F.cross_entropy(class_logits_slice, class_targets_slice, reduction="none")
            within_ce.index_copy_(0, row_ids, within_values)
        class_ce = class_ce.view(original_shape)
        within_ce = within_ce.view(original_shape)
        return class_ce + within_ce, class_ce, within_ce
