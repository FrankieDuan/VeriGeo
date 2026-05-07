"""Per-model configuration for the generation harness.

Reads ``LLM_<NAME>_*`` environment variables (populated via ``.env``).
All endpoints and API keys must come from environment variables — no
credentials or hostnames are embedded in this file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class ModelConfig:
    name: str
    endpoint: str
    model: str
    api_key: str
    max_tokens: int = 60000
    temperature: float = 1.0
    timeout_secs: int = 1000
    cost_per_1m_input: float = 0.0
    cost_per_1m_output: float = 0.0
    cost_per_1m_cache_read: float = 0.0


# (name, default_model, cost_in, cost_out, cost_cache)
# Endpoints MUST come from env (LLM_<NAME>_ENDPOINT). API keys also
# from env (LLM_<NAME>_API_KEY or VERIGEO_API_KEY). No hardcoded hosts.
_MODEL_DEFAULTS: List[Tuple[str, str, float, float, float]] = [
    ("gemini",           "gemini-2.5-pro",            1.80, 10.80, 0.180),
    ("claude-opus",      "claude-opus-4",             4.50, 22.50, 0.450),
    ("gpt-mini",         "gpt-4o-mini",               0.60,  3.60, 0.150),
    ("gpt",              "gpt-4o",                    2.50, 15.00, 0.250),
    ("deepseek",         "deepseek-chat",             0.435, 0.869, 0.004),
    ("qwen",             "qwen-plus",                 0.104, 0.624, 0.026),
]

_AVAILABLE_NAMES = tuple(d[0] for d in _MODEL_DEFAULTS)


def _resolve_api_key(name_upper: str) -> str:
    """Resolve a per-model API key, falling back to the shared VERIGEO_API_KEY."""
    for key in (f"LLM_{name_upper}_API_KEY", "VERIGEO_API_KEY"):
        val = os.environ.get(key, "").strip()
        if val:
            return val
    return ""


def load_model_configs(selected: Optional[List[str]] = None) -> List[ModelConfig]:
    """Load all available ModelConfig objects, optionally filtered by name."""
    configs: List[ModelConfig] = []
    for name, default_model, cost_in, cost_out, cost_cache in _MODEL_DEFAULTS:
        upper = name.upper().replace("-", "_")
        endpoint = os.environ.get(f"LLM_{upper}_ENDPOINT", "").strip()
        if endpoint.endswith("/v1"):
            endpoint = endpoint[:-3]
        endpoint = endpoint.rstrip("/")
        model = os.environ.get(f"LLM_{upper}_MODEL", default_model)
        api_key = _resolve_api_key(upper)
        configs.append(
            ModelConfig(
                name=name,
                endpoint=endpoint,
                model=model,
                api_key=api_key,
                max_tokens=int(os.environ.get(f"LLM_{upper}_MAX_TOKENS", "60000")),
                temperature=float(os.environ.get(f"LLM_{upper}_TEMPERATURE", "1.0")),
                timeout_secs=int(os.environ.get(f"LLM_{upper}_TIMEOUT", "1000")),
                cost_per_1m_input=float(os.environ.get(f"LLM_{upper}_COST_IN", str(cost_in))),
                cost_per_1m_output=float(os.environ.get(f"LLM_{upper}_COST_OUT", str(cost_out))),
                cost_per_1m_cache_read=float(os.environ.get(f"LLM_{upper}_COST_CACHE_READ", str(cost_cache))),
            )
        )
    if selected:
        wanted = set(selected)
        configs = [c for c in configs if c.name in wanted]
    return configs


def available_model_names() -> Tuple[str, ...]:
    return _AVAILABLE_NAMES


__all__ = ["ModelConfig", "load_model_configs", "available_model_names"]
