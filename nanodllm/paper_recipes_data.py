"""Portable training-input identities for the paper recipes."""

import hashlib
from pathlib import Path

VALIDATION_SHARD = "shard_06542.parquet"
SHARD_LIST_SHA256 = "e4c794c66628e2518d91992bc8960981cd187d6fb4ee4724ae2b2465d5350145"
TOKENIZER_FILES = {
    "tokenizer.json": "9d9c605ecd29573a761c0085b4eb8bf65f6647edb915c50ac7d540fa067de01e",
    "tokenizer_config.json": "2d2a3e592b34fffc229ba0c4356b84d8db4971c342737213ef823b9285328735",
    "tokenizer_ordinary.json": "5be8f649ad2fa26cae084a6ecc968502adb12454bca6c66ba2e9ca9dfecb6ac2",
    "token_bytes.json": "3c83f5c1250af0bd96bcb23121c7ca866aa0c94c7706f7053795a80a17f22a55",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_tokenizer(root: Path) -> None:
    for name in ("added_tokens.json", "special_tokens_map.json", "config.json"):
        if (root / name).exists():
            raise ValueError(f"unexpected tokenizer input {name} in paper tokenizer directory")
    for name, expected in TOKENIZER_FILES.items():
        if file_sha256(root / name) != expected:
            raise ValueError(f"paper tokenizer SHA-256 mismatch for {name}")


def verify_corpus(root: Path) -> None:
    """Match the trainer's recursive discovery and the portable name/size inventory."""
    listing = hashlib.sha256()
    files = sorted(root.rglob("*.parquet"))
    for path in files:
        listing.update(f"{path.relative_to(root).as_posix()} {path.stat().st_size}\n".encode())
    if len(files) != 6543 or listing.hexdigest() != SHARD_LIST_SHA256:
        raise ValueError("expected the paper corpus with its 6,543 recorded shard names and byte sizes")
    # The native loader sorts canonical paths before it shuffles row groups.
    # A cache with symlinks named by shard but resolved by blob ID can change that permutation.
    canonical = [str(path.resolve(strict=True)) for path in files]
    if canonical != sorted(set(canonical)):
        raise ValueError("resolved shard paths change the training order; use a local copy in numeric shard order")
