"""Reuse pinned single-rank kernels after DDP has averaged gradients exactly once.

All ranks own complete optimizer state. Rank-zero checkpoints consequently retain
the normal trainer restore contract; no native sharded state or collectives enter
this adapter. Only orchestration differs from upstream, not update equations.
"""

import torch


def replicated_optimizer(native_class, groups):
    class ReplicatedMuonAdamW(native_class):
        @torch.no_grad()
        def step(self):
            gathers = []
            for group in self.param_groups:
                if group["kind"] == "adamw":
                    info = self._reduce_adamw(group, 1)
                    self._compute_adamw(group, info, gathers, 0, 1)
                elif group["kind"] == "muon":
                    info = self._reduce_muon(group, 1)
                    self._compute_muon(group, info, gathers, 0)
                else:
                    raise ValueError(f"Unknown optimizer kind: {group['kind']}")
            self._finish_gathers(gathers)

    return ReplicatedMuonAdamW(groups)
