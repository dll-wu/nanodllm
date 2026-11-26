"""Input checks for the released sampling paths."""


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)
