"""Exact and randomized cardinality quadrature for conditional masked NELBO.

The time-independent predictor permits analytic integration over masking time.
One subset of size k contributes mean(masked CE) / choose(n, k). See the
conditional-continuation-scoring document for the derivation and assumptions.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
from dataclasses import dataclass

PLAN_VERSION = "conditional_mask_cardinality.v1"


@dataclass(frozen=True)
class NelboOptions:
    exact_max_tokens: int = 7
    mc_draws: int = 32
    mc_replicates: int = 4
    seed: int = 1337

    def __post_init__(self) -> None:
        if type(self.exact_max_tokens) is not int or not 1 <= self.exact_max_tokens <= 12:
            raise ValueError("exact_max_tokens must be an integer in [1, 12]")
        if type(self.mc_draws) is not int or self.mc_draws < 1:
            raise ValueError("mc_draws must be a positive integer")
        if type(self.mc_replicates) is not int or self.mc_replicates < 2:
            raise ValueError("mc_replicates must be >= 2 to estimate Monte Carlo uncertainty")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")


@dataclass(frozen=True)
class MaskDraw:
    masked: tuple[int, ...]
    weight: float  # Coefficient on the MEAN masked-token CE, not the sum.
    replicate: int


@dataclass(frozen=True)
class MaskPlan:
    method: str
    target_tokens: int
    replicates: int
    draws: tuple[MaskDraw, ...]
    sha256: str


def _seed(seed: int, sample_id: str, group: int, n: int, replicate: int) -> int:
    # Exclude candidate index, gold label, checkpoint, device and DB: equal-length
    # alternatives and checkpoints share masks, regardless of execution batches.
    payload = json.dumps([PLAN_VERSION, seed, sample_id, group, n, replicate], ensure_ascii=True)
    return int.from_bytes(hashlib.sha256(payload.encode()).digest(), "big")


def make_mask_plan(n: int, options: NelboOptions, *, sample_id: str, group: int = 0) -> MaskPlan:
    if type(n) is not int or n < 1:
        raise ValueError("target length must be a positive integer")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample_id must be a nonempty stable string")
    if type(group) is not int or group < 0:
        raise ValueError("group must be a nonnegative integer")
    draws = []
    if n <= options.exact_max_tokens:
        method = "exact_full_mask" if n == 1 else "exact_subsets"
        replicates = 1
        for k in range(1, n + 1):
            weight = 1.0 / math.comb(n, k)
            draws.extend(MaskDraw(subset, weight, 0) for subset in itertools.combinations(range(n), k))
    else:
        method = "stratified_cardinality_mc"
        replicates = options.mc_replicates
        for replicate in range(replicates):
            rng = random.Random(_seed(options.seed, sample_id, group, n, replicate))
            for i in range(options.mc_draws):
                # Jitter one point inside each uniform stratum. The pooled
                # cardinality distribution is exactly uniform on 1..n, even
                # when n and the number of draws are not divisible.
                k = 1 + min(n - 1, int(n * (i + rng.random()) / options.mc_draws))
                subset = tuple(sorted(rng.sample(range(n), k)))
                draws.append(MaskDraw(subset, n / options.mc_draws, replicate))
    payload = [PLAN_VERSION, method, n, replicates, [(d.masked, d.weight, d.replicate) for d in draws]]
    digest = hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()
    return MaskPlan(method, n, replicates, tuple(draws), digest)


def reduce_mask_plan(plan: MaskPlan, masked_means: tuple[float, ...]) -> tuple[float, ...]:
    """Return independent replicate estimates of SUMMED conditional NELBO."""
    if len(masked_means) != len(plan.draws) or any(not math.isfinite(x) or x < 0 for x in masked_means):
        raise ValueError("one finite nonnegative mean CE is required per planned draw")
    weighted: list[list[float]] = [[] for _ in range(plan.replicates)]
    for draw, loss in zip(plan.draws, masked_means, strict=True):
        weighted[draw.replicate].append(draw.weight * loss)
    return tuple(math.fsum(values) for values in weighted)
