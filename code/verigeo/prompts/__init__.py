"""Prompt asset loader.

Prompt bodies live in this directory as ``<name>.md`` files.
Use :func:`load_prompt` to read them at runtime; results are cached.
"""
from __future__ import annotations

import os
from functools import lru_cache

_PROMPT_DIR = os.path.dirname(os.path.abspath(__file__))


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """Read prompt ``name`` (without extension) from the ``prompts/`` directory.

    Preserves exact bytes from the source ``.md`` — including leading/trailing
    whitespace — because the engine concatenates these strings verbatim into
    LLM request bodies.
    """
    path = os.path.join(_PROMPT_DIR, f"{name}.md")
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


__all__ = ["load_prompt"]
