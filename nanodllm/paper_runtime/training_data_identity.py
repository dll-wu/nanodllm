"""Freeze and compare data identities before constructing training loaders."""

from pathlib import Path

import torch.distributed as dist

from .config import ExperimentConfig
from .data_identity import build_data_identity
from .data_identity_io import write_data_identity
from .data_snapshot import VERIFICATION_POLICY


def _prepare_data_identity(
    config: ExperimentConfig,
    *,
    rank: int,
    world_size: int,
    is_ddp: bool,
) -> dict[str, bool | str]:
    """Freeze and cross-rank-check the data contract before constructing a loader."""

    mode = str(config.data.identity_mode)
    if mode == "off":
        return {
            "enabled": False,
            "mode": mode,
            "artifact": "",
            "identity_sha256": "",
            "identity_sha256_strength": "",
        }

    record: dict | None = None
    local_error = ""
    try:
        record = build_data_identity(
            config,
            full_sha256=mode == "full_sha256",
            world_size=int(world_size),
            assignment_epoch=1,
        )
    except Exception as exc:  # Gather failures so one rank cannot silently disagree.
        local_error = f"{type(exc).__name__}: {exc}"

    local_result = {
        "rank": int(rank),
        "error": local_error,
        "identity_sha256": str(record["hashes"]["identity_sha256"]) if record is not None else "",
    }
    gathered = [local_result]
    if is_ddp:
        gathered = [None for _ in range(int(world_size))]
        dist.all_gather_object(gathered, local_result)
    errors = [f"rank {item['rank']}: {item['error']}" for item in gathered if item and item["error"]]
    if errors:
        raise RuntimeError("data identity failed: " + "; ".join(errors))
    digests = {str(item["identity_sha256"]) for item in gathered if item}
    if len(digests) != 1:
        raise RuntimeError(f"data identity differs across ranks: {sorted(digests)}")
    assert record is not None

    digest = str(record["hashes"]["identity_sha256"])
    publication = ["", ""]
    if rank == 0:
        try:
            snapshot = bool(config.data.dataset_snapshot)
            artifact = write_data_identity(
                Path(config.train.run_dir),
                record,
                compact=snapshot,
                verification={
                    "policy": VERIFICATION_POLICY if snapshot else "full_inventory_at_startup",
                    "dataset_snapshot_sha256": config.data.dataset_snapshot_sha256,
                    "full_corpus_revalidated_at_startup": not snapshot,
                    "opened_shard_content_hashing": mode == "full_sha256" and snapshot,
                },
            )
            publication[0] = artifact.name
        except Exception as exc:
            publication[1] = f"{type(exc).__name__}: {exc}"
    if is_ddp:
        dist.broadcast_object_list(publication, src=0)
    if publication[1]:
        raise RuntimeError("cannot publish data identity: " + publication[1])
    artifact_name = publication[0]
    return {
        "enabled": True,
        "mode": mode,
        "artifact": artifact_name,
        "identity_sha256": digest,
        "identity_sha256_strength": str(record["hashes"]["identity_sha256_strength"]),
    }
