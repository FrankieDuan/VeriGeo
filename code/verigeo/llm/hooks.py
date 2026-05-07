"""Integration hook registries for the LLM transport layer.

Replaces the monkey-patching approach used by the original ``pipeline.py``.
External tooling (generation harnesses, token accountants, model-specific
request shapers) registers callbacks here; the engine's HTTP client invokes
them at well-defined points.

Three extension points are exposed:

* **request preprocessors** — run on the outgoing request body (list order)
  before the HTTP call; can mutate or replace the body. Typical use: strip
  fields that a given provider rejects.
* **response observers** — run on each ``(request_body, response_dict, model)``
  triple after a successful call. Read-only in spirit; used for token
  accounting, latency tracking, etc.
* **cache policy** — a single function ``(model_name) -> bool`` that decides
  whether Anthropic-style ``cache_control`` breakpoints should be applied on
  a per-request basis. If unset, the engine falls back to a structural check
  on the body (any ``cache_control`` key already present).
"""
from __future__ import annotations

from threading import RLock
from typing import Callable, Dict, List, Optional

RequestPreprocessor = Callable[[Dict, str], Dict]
ResponseObserver = Callable[[Dict, Dict, str], None]
CachePolicy = Callable[[str], bool]

_lock = RLock()
_request_preprocessors: List[RequestPreprocessor] = []
_response_observers: List[ResponseObserver] = []
_cache_policy: Optional[CachePolicy] = None


def register_request_preprocessor(fn: RequestPreprocessor) -> RequestPreprocessor:
    """Append ``fn`` to the request preprocessor chain; return ``fn`` for decorator use."""
    with _lock:
        _request_preprocessors.append(fn)
    return fn


def register_response_observer(fn: ResponseObserver) -> ResponseObserver:
    """Append ``fn`` to the response observer chain; return ``fn`` for decorator use."""
    with _lock:
        _response_observers.append(fn)
    return fn


def set_cache_policy(fn: Optional[CachePolicy]) -> None:
    """Install (or clear, if ``None``) the global cache policy."""
    global _cache_policy
    with _lock:
        _cache_policy = fn


def reset() -> None:
    """Drop all registered hooks — intended for tests."""
    global _cache_policy
    with _lock:
        _request_preprocessors.clear()
        _response_observers.clear()
        _cache_policy = None


# --- engine-side invocation helpers --------------------------------------------------

def run_request_preprocessors(body: Dict, model: str) -> Dict:
    with _lock:
        chain = list(_request_preprocessors)
    for fn in chain:
        body = fn(body, model)
    return body


def run_response_observers(body: Dict, response: Dict, model: str) -> None:
    with _lock:
        chain = list(_response_observers)
    for fn in chain:
        try:
            fn(body, response, model)
        except Exception:  # observers must never break the main call path
            pass


def should_apply_cache(model: str) -> Optional[bool]:
    with _lock:
        policy = _cache_policy
    if policy is None:
        return None
    try:
        return bool(policy(model))
    except Exception:
        return None


__all__ = [
    "RequestPreprocessor",
    "ResponseObserver",
    "CachePolicy",
    "register_request_preprocessor",
    "register_response_observer",
    "set_cache_policy",
    "reset",
    "run_request_preprocessors",
    "run_response_observers",
    "should_apply_cache",
]
