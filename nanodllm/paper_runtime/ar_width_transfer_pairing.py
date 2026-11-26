"""Complete consumed-batch identities for the scoped width-transfer experiment."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

from .data import DLMDataBatch
from .telemetry import write_json_atomic
from .training_checkpoint import _batch_six_tensor_digests

SCHEMA = "nanodllm.ar_width_transfer_batch_pairing.v1"
METHOD = "sha256_chain_of_ordered_native_six_tensor_digests_and_epoch_v1"
INITIAL_CHAIN = hashlib.sha256(METHOD.encode()).hexdigest()


class PairedBatchLoader:
    """Recompute replayed prefixes; exclude native one-batch lookahead from the endpoint."""

    def __init__(
        self,
        loader,
        path: Path,
        *,
        training_seed: int,
        source_shuffle_seed: int,
        expected_microbatches: int,
        grad_accum_steps: int,
        resume: bool,
    ):
        if expected_microbatches <= 0 or grad_accum_steps <= 0:
            raise ValueError("batch pairing requires positive horizon and accumulation")
        self.loader = iter(loader)
        self.path = path
        self.expected = expected_microbatches
        self.accumulation = grad_accum_steps
        self.count = 0
        self.chain = INITIAL_CHAIN
        self.identity = {
            "schema": SCHEMA,
            "digest_method": METHOD,
            "training_seed": training_seed,
            "source_shuffle_seed": source_shuffle_seed,
            "expected_microbatches": expected_microbatches,
        }
        self.previous = None
        if path.exists():
            if not resume:
                raise ValueError("existing batch pairing receipt requires explicit resume")
            self.previous = json.loads(path.read_text())
            if {k: self.previous.get(k) for k in self.identity} != self.identity:
                raise ValueError("batch pairing resume identity changed")
            count = self.previous.get("consumed_microbatches")
            chain = self.previous.get("chain_sha256")
            if (
                type(count) is not int
                or not 0 < count <= self.expected
                or not isinstance(chain, str)
                or len(chain) != 64
                or any(c not in "0123456789abcdef" for c in chain)
                or type(self.previous.get("complete")) is not bool
            ):
                raise ValueError("invalid previous batch pairing receipt")
        self.previous_verified = self.previous is None

    def __iter__(self) -> Iterator[DLMDataBatch]:
        return self

    def _receipt(self, *, complete: bool) -> dict:
        return {
            **self.identity,
            "consumed_microbatches": min(self.count, self.expected),
            "chain_sha256": self.chain,
            "complete": complete,
        }

    def __next__(self) -> DLMDataBatch:
        batch = next(self.loader)
        self.count += 1
        if self.count > self.expected:
            return batch  # Native checkpoint replay separately binds the prefetched next batch.
        entry = _batch_six_tensor_digests(batch)
        payload = json.dumps(
            {"index": self.count, "epoch": entry["epoch"], "batch_sha256": entry["combined_sha256"]},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        self.chain = hashlib.sha256(bytes.fromhex(self.chain) + payload).hexdigest()
        if self.previous is not None and self.count == self.previous["consumed_microbatches"]:
            if self.chain != self.previous["chain_sha256"]:
                raise ValueError("batch pairing replay prefix changed")
            self.previous_verified = True
        if self.previous_verified and self.count % self.accumulation == 0:
            # Prefix observation alone does not certify successful optimizer consumption.
            write_json_atomic(str(self.path), self._receipt(complete=False))
        return batch

    def finish(self) -> dict:
        if self.count < self.expected or not self.previous_verified:
            raise ValueError("batch pairing endpoint is incomplete or replay is unverified")
        receipt = self._receipt(complete=True)
        write_json_atomic(str(self.path), receipt)
        return receipt


def wrap_training_loader(config, loader, *, grad_accum_steps: int, total_steps: int):
    from .ar_width_mup import active

    if not active(config):
        return loader
    seed = config.runtime.training_seed if config.runtime.training_seed >= 0 else config.runtime.seed
    return PairedBatchLoader(
        loader,
        Path(config.train.run_dir) / "batch-pairing.json",
        training_seed=int(seed),
        source_shuffle_seed=int(config.data.shuffle_seed),
        expected_microbatches=total_steps * grad_accum_steps,
        grad_accum_steps=grad_accum_steps,
        resume=bool(config.train.resume_from),
    )


def finish_training_loader(loader) -> None:
    if isinstance(loader, PairedBatchLoader):
        loader.finish()
