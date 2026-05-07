"""Hook wiring that replaces the legacy monkey-patches in pipeline.py.

Registers three things with :mod:`verigeo.llm.hooks`:

1. A request preprocessor that sanitizes request bodies for providers with
   quirks (GPT: strip ``strict`` / ``cache_control``; DeepSeek: drop
   ``response_format`` / image blocks).
2. A response observer that feeds token usage into :class:`TokenAccumulator`.
3. A cache policy that only enables Anthropic prompt-caching for Claude
   models (keeps OpenAI-style providers on ``/chat/completions``).
"""
from __future__ import annotations

from verigeo.llm import hooks

from generate.tokens import TokenAccumulator

_INSTALLED = False


def _sanitize_request(body: dict, model: str) -> dict:
    """Strip provider-incompatible fields from the outgoing request body."""
    lower = (model or "").lower()
    is_gpt = "gpt" in lower or "o1" in lower or "o3" in lower
    is_deepseek = "deepseek" in lower

    if is_gpt:
        rf = body.get("response_format")
        if isinstance(rf, dict) and rf.get("type") == "json_schema":
            js = rf.get("json_schema", {})
            if isinstance(js, dict) and js.get("strict"):
                js["strict"] = False
        for msg in body.get("messages", []):
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        block.pop("cache_control", None)

    if is_deepseek:
        body.pop("response_format", None)
        for msg in body.get("messages", []):
            content = msg.get("content")
            if isinstance(content, list):
                kept = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "image_url":
                            continue
                        block.pop("cache_control", None)
                        kept.append(block)
                    else:
                        kept.append(block)
                msg["content"] = kept

    return body


def _record_tokens(_body: dict, response: dict, _model: str) -> None:
    TokenAccumulator.record(response)


def _claude_only_cache(model: str) -> bool:
    return "claude" in (model or "").lower()


def install() -> None:
    """Register hooks with the engine. Idempotent."""
    global _INSTALLED
    if _INSTALLED:
        return
    hooks.register_request_preprocessor(_sanitize_request)
    hooks.register_response_observer(_record_tokens)
    hooks.set_cache_policy(_claude_only_cache)
    _INSTALLED = True


__all__ = ["install"]
