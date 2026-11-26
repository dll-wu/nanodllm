"""Stable NELBO row identities and sequence-level uncertainty reduction."""

import hashlib
import math
import struct

import numpy as np
import torch


def _update_identity_digest(digest, label: bytes, payload: bytes | None) -> None:
    """Add one unambiguous, length-prefixed field to a row identity digest."""

    digest.update(struct.pack("<I", len(label)))
    digest.update(label)
    if payload is None:
        digest.update(struct.pack("<Q", 2**64 - 1))
        return
    digest.update(struct.pack("<Q", len(payload)))
    digest.update(payload)


def _little_endian_row_bytes(row: torch.Tensor, dtype: str) -> bytes:
    array = row.detach().contiguous().numpy()
    return array.astype(dtype, copy=False).tobytes(order="C")


def _packed_bool_row_bytes(row: torch.Tensor) -> bytes:
    array = row.detach().to(dtype=torch.uint8).contiguous().numpy()
    return np.packbits(array, bitorder="little").tobytes(order="C")


def _nelbo_input_identity_sha256(
    x_0_row: torch.Tensor,
    attention_mask_row: torch.Tensor,
    loss_mask_row: torch.Tensor,
    position_ids_row: torch.Tensor | None,
    document_start_mask_row: torch.Tensor | None,
) -> str:
    digest = hashlib.sha256(b"nanodllm-nelbo-input-identity-v2\0")
    _update_identity_digest(digest, b"x_0_i64_le", _little_endian_row_bytes(x_0_row, "<i8"))
    _update_identity_digest(digest, b"attention_mask_packbits_le", _packed_bool_row_bytes(attention_mask_row))
    _update_identity_digest(digest, b"loss_mask_packbits_le", _packed_bool_row_bytes(loss_mask_row))
    _update_identity_digest(
        digest,
        b"position_ids_i64_le",
        None if position_ids_row is None else _little_endian_row_bytes(position_ids_row, "<i8"),
    )
    _update_identity_digest(
        digest,
        b"document_start_mask_packbits_le",
        None if document_start_mask_row is None else _packed_bool_row_bytes(document_start_mask_row),
    )
    return digest.hexdigest()


def _nelbo_uniform_sha256(uniform_row: torch.Tensor) -> str:
    digest = hashlib.sha256(b"nanodllm-nelbo-corruption-uniform-v1\0")
    _update_identity_digest(digest, b"uniform_f32_le", _little_endian_row_bytes(uniform_row, "<f4"))
    return digest.hexdigest()


def _nelbo_corruption_mask_sha256(active_mask_row: torch.Tensor) -> str:
    digest = hashlib.sha256(b"nanodllm-nelbo-corruption-mask-v2\0")
    _update_identity_digest(digest, b"active_mask_packbits_le", _packed_bool_row_bytes(active_mask_row))
    return digest.hexdigest()


def _nelbo_evaluation_sample_sha256(
    input_identity_sha256: str,
    sampled_t: float,
    corruption_uniform_sha256: str,
    corruption_mask_sha256: str,
) -> str:
    digest = hashlib.sha256(b"nanodllm-nelbo-evaluation-sample-v2\0")
    digest.update(bytes.fromhex(input_identity_sha256))
    digest.update(struct.pack("<f", float(sampled_t)))
    digest.update(bytes.fromhex(corruption_uniform_sha256))
    digest.update(bytes.fromhex(corruption_mask_sha256))
    return digest.hexdigest()


def _ordered_sha256_root(domain: bytes, row_digests: list[str]) -> str:
    digest = hashlib.sha256(domain + b"\0")
    digest.update(struct.pack("<Q", len(row_digests)))
    for row_digest in row_digests:
        digest.update(bytes.fromhex(row_digest))
    return digest.hexdigest()


def _delete_one_sequence_ratio_se(
    numerators: list[float],
    denominators: list[float],
    *,
    scale: float = 1.0,
) -> float | None:
    """Return a delete-one-sequence jackknife diagnostic for a ratio of totals."""

    if len(numerators) != len(denominators):
        raise ValueError("numerator and denominator sequence counts must match")
    if len(numerators) < 2:
        return None
    total_numerator = math.fsum(numerators)
    total_denominator = math.fsum(denominators)
    leave_one_out = []
    for numerator, denominator in zip(numerators, denominators, strict=True):
        remaining_denominator = total_denominator - denominator
        if remaining_denominator <= 0:
            return None
        leave_one_out.append((total_numerator - numerator) / (scale * remaining_denominator))
    mean = math.fsum(leave_one_out) / len(leave_one_out)
    return math.sqrt(
        (len(leave_one_out) - 1) / len(leave_one_out) * sum((estimate - mean) ** 2 for estimate in leave_one_out)
    )
