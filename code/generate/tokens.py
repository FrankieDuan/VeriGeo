"""Thread-local token accumulator.

Registered as an engine response observer via :mod:`verigeo.llm.hooks`.
The generation runner calls :meth:`start` at the beginning of each attempt,
then :meth:`get_totals` / :meth:`get_cache_totals` at the end.
"""
from __future__ import annotations

import threading
from typing import Tuple


class TokenAccumulator:
    _tls = threading.local()

    @classmethod
    def start(cls) -> None:
        cls._tls.prompt_tokens = 0
        cls._tls.completion_tokens = 0
        cls._tls.call_count = 0
        cls._tls.total_duration = 0.0
        cls._tls.cache_read_tokens = 0
        cls._tls.cache_create_tokens = 0

    @classmethod
    def record(cls, resp_data: dict) -> None:
        usage = resp_data.get("usage") or {}
        cls._tls.prompt_tokens = getattr(cls._tls, "prompt_tokens", 0) + int(usage.get("prompt_tokens", 0))
        cls._tls.completion_tokens = getattr(cls._tls, "completion_tokens", 0) + int(usage.get("completion_tokens", 0))
        cls._tls.call_count = getattr(cls._tls, "call_count", 0) + 1
        cache_read = int(usage.get("cache_read_input_tokens", 0))
        cache_create = int(usage.get("cache_creation_input_tokens", 0))
        if not cache_read:
            cache_read = int(usage.get("prompt_tokens_details", {}).get("cached_tokens", 0))
        cls._tls.cache_read_tokens = getattr(cls._tls, "cache_read_tokens", 0) + cache_read
        cls._tls.cache_create_tokens = getattr(cls._tls, "cache_create_tokens", 0) + cache_create

    @classmethod
    def get_totals(cls) -> Tuple[int, int, float, int]:
        return (
            getattr(cls._tls, "prompt_tokens", 0),
            getattr(cls._tls, "completion_tokens", 0),
            getattr(cls._tls, "total_duration", 0.0),
            getattr(cls._tls, "call_count", 0),
        )

    @classmethod
    def get_cache_totals(cls) -> Tuple[int, int]:
        return (
            getattr(cls._tls, "cache_read_tokens", 0),
            getattr(cls._tls, "cache_create_tokens", 0),
        )


__all__ = ["TokenAccumulator"]
