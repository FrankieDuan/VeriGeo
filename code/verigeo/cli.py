"""Command-line entry point: ``python -m verigeo``.

Runs the default batch generation pipeline defined by :data:`CONFIG`.
"""
from __future__ import annotations

from verigeo import run_generation_from_config


def main() -> int:
    run_generation_from_config()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
