"""Central LLM configuration.

All secrets are sourced from environment variables (populated via ``.env``).
No API keys are embedded in code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from verigeo.prompts import load_prompt

# --------------------------------------------------------------------------- env

def _load_dotenv(path: str) -> None:
    """Minimal .env loader — does not override already-set env vars."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


# Load .env from project root (one level up from verigeo/).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))


def _env_api_key() -> str:
    """Resolve the engine API key from env."""
    return os.environ.get("VERIGEO_API_KEY", "").strip()


def _env_endpoint() -> str:
    """Resolve the engine API endpoint from env. Empty if not set."""
    return os.environ.get("VERIGEO_API_ENDPOINT", "").strip()


# --------------------------------------------------------------------------- LLMConfig

@dataclass
class LLMConfig:
    endpoint: str = field(default_factory=_env_endpoint)
    model: str = field(default_factory=lambda: os.environ.get("VERIGEO_MODEL", "claude-sonnet-4-5-20250929"))
    api_key: str = field(default_factory=_env_api_key)
    max_tokens: int = 60000
    timeout_secs: int = 1000
    temperature: float = 1
    difficulty: str = "easy"
    style: str = "exam"
    batch_parallel_jobs: int = 100
    show_single_job_plot: bool = True
    validate_problem_json: bool = False
    llm_max_inflight: Optional[int] = None
    llm_log_timing: bool = True
    llm_cache_enable: bool = True
    llm_cache_dir: Optional[str] = None
    llm_cache_ttl_seconds: int = 7 * 24 * 3600
    llm_cache_max_size_mb: int = 2048
    sub_ids: Optional[List[str]] = None
    author_reflection_prompt: str = field(default_factory=lambda: load_prompt("author_reflection") if os.path.exists(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", "author_reflection.md")) else _AUTHOR_REFLECTION_PROMPT)
    author_reflection_rounds_pass: int = 1
    author_reflection_rounds_fail: int = 1
    solver_reflection_prompt: str = field(default_factory=lambda: _SOLVER_REFLECTION_PROMPT)
    solver_reflection_rounds_pass: int = 1
    solver_reflection_rounds_fail: int = 1
    enable_teacher_agent: bool = False
    author_enable_enforce: bool = False
    blueprint_enable: bool = True
    blueprint_seed: Optional[int] = None
    blueprint_library_path: str = "blueprint_library.json"
    blueprint_library_reload: bool = False
    blueprint_family: Optional[str] = "euclid"
    output_root: Optional[str] = None
    matrix_enable: bool = True
    matrix_num_per_combo: Optional[int] = None
    matrix_counts: Optional[Dict[str, Dict[str, int]]] = field(
        default_factory=lambda: {"vector_coord": {"easy": 100, "medium": 100, "hard": 100}}
    )
    matrix_combo_counts: Optional[List[Tuple[str, str, int]]] = None
    matrix_families: Optional[List[str]] = None
    matrix_difficulties: Optional[List[str]] = None
    matrix_base_seed: int = 5656
    matrix_max_workers: Optional[int] = None
    analytic_fill_free_params: bool = True
    analytic_fill_constant: float = 0.35
    analytic_tol: float = 1e-6
    analytic_max_unknowns: Optional[int] = None
    analytic_max_equations: Optional[int] = None
    analytic_timeout_seconds: Optional[float] = 60.0


# --------------------------------------------------------------------------- reflection prompts
# These short reflection prompts stay inline (they're part of the engine's control-flow
# grammar, not large prose assets).

_AUTHOR_REFLECTION_PROMPT = """
You are a geometry problem logic reviewer.

Your responsibility is to ensure that the problem design and the solution logic are correct and coherent.

1) Logical consistency check
- Check that the problem statement, diagram construction (actions), givens, proof_outline, and goal are logically consistent.
- Verify that every step in proof_outline follows from previous steps, the givens, or standard geometric theorems.
- Ensure that no step relies on hidden assumptions or unstated conditions.
- Ensure that every given is actually used in the reasoning.
- Ensure that the goal is genuinely implied by the givens and the constructed diagram.

2) Corrective revision (allowed and encouraged)
- If you find any logical flaw, missing justification, incorrect theorem usage, or mismatch between the diagram and the reasoning:
  * You SHOULD revise the problem, proof_outline, and/or construction actions as needed.
  * You MAY significantly modify the structure of the problem or the construction if necessary.
  * You do NOT need to discard the entire problem unless it is impossible to fix.
- Your objective is to make the problem logically correct, not to preserve the original form.

3) VeriScript rewrite triggered by execution logs
- If execution logs indicate failures such as:
  * intersection not found,
  * numeric solving failure,
  * invalid or unstable construction,
- you MUST rewrite the Author-VeriScript actions according to the logs.
- The rewritten VeriScript must be consistent with the problem statement and proof_outline.

4) Scope limitation
- You do NOT need to verify numerical correctness; numeric validation is handled automatically.
- Focus only on logical soundness, structural correctness, and VeriScript executability.

Rules:
- Do not explain your review process.
- Do not output warnings, comments, or analysis.
- Always output a valid problem JSON.
- The final output must represent a logically correct and executable problem–solution pair.
- Set problem.meta.reflection_modified = true if you modified anything in this reflection; otherwise set it to false.

This reflection stage exists to guarantee logical correctness and VeriScript executability, not numerical verification.
"""

_SOLVER_REFLECTION_PROMPT = """
Self-check rules (must be strictly followed):

1) Logical correctness
- Check whether the problem, proof, and final_answer are logically correct and coherent.
- Ensure the solution contains no skipped steps, no concept swapping, and no reliance on unstated assumptions.

2) Log-driven VeriScript revision
- If execution logs indicate intersection failure, numeric solving failure, or unstable construction,
  you MUST rewrite the VeriScript actions based on the logs
  (including adjusting constructions, adding auxiliary points/lines, or reordering actions).
- The revised VeriScript must still support the reasoning expressed in the proof.

3) Strict alignment between VeriScript and proof
- Every key reasoning step in the proof must have a clear and unique corresponding VeriScript action.
- It is forbidden for the proof to contain conclusions not encoded in VeriScript,
  or for VeriScript to introduce essential constraints not used in the proof.

Output requirements:
- You must output a complete and valid JSON (proof / actions / final_answer).
- Do not output any checking explanations, analysis, or extra text.
- The solution is acceptable only if the logic is correct, the VeriScript is executable,
  and the VeriScript and proof are in strict correspondence.
- Include a boolean field reflection_modified in the JSON; set it to true if you modified any part in this reflection round, otherwise false.
"""

# --------------------------------------------------------------------------- singleton

CONFIG = LLMConfig()

# Honor a few legacy env overrides that existed in the original file.
if os.environ.get("LLM_CACHE_ENABLE", "").strip().lower() in ("0", "false", "no", "off"):
    CONFIG.llm_cache_enable = False
if os.environ.get("LLM_CACHE_DIR", "").strip():
    CONFIG.llm_cache_dir = os.environ["LLM_CACHE_DIR"].strip()
if os.environ.get("LLM_CACHE_TTL", "").strip():
    try:
        CONFIG.llm_cache_ttl_seconds = int(os.environ["LLM_CACHE_TTL"])
    except ValueError:
        pass

AUTHOR_WORKFLOW = os.environ.get("AUTHOR_WORKFLOW", "unified")
AUTHOR_FROM_SCRATCH = os.environ.get("AUTHOR_FROM_SCRATCH", "1").strip().lower() in {"1", "true", "yes", "on"}
CORE_SHAPE = "trapezoid"

PROJECT_ROOT = _PROJECT_ROOT

# Kept as an empty list for back-compat with call sites that use
# ``construction_targets or _DEFAULT_SCRATCH_CONSTRUCTION_TARGETS`` as a
# default sentinel. Originally non-empty themes that were disabled upstream.
_DEFAULT_SCRATCH_CONSTRUCTION_TARGETS: List[str] = []


def require_api_key() -> str:
    """Read the API key, raising a clear error if it's missing.

    Call this at the boundary where we actually issue an HTTP request, so the
    module stays importable without credentials (for tests, tooling, docs).
    """
    key = CONFIG.api_key or _env_api_key()
    if not key:
        raise RuntimeError(
            "VERIGEO_API_KEY is not set. Populate .env or export the env var before "
            "running generation / solver LLM calls."
        )
    CONFIG.api_key = key
    return key


__all__ = [
    "LLMConfig",
    "CONFIG",
    "AUTHOR_WORKFLOW",
    "AUTHOR_FROM_SCRATCH",
    "CORE_SHAPE",
    "PROJECT_ROOT",
    "require_api_key",
]
