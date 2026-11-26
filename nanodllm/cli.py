"""Train from a JSON config. All launchers share this entry point."""

from .runtime.cli import build_parser, main, resolve_config

__all__ = ["build_parser", "main", "resolve_config"]

if __name__ == "__main__":
    main()
