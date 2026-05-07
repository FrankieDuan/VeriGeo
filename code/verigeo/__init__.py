"""VeriGeo — geometry problem generation + solver pipeline.

Public API. Import from here rather than reaching into :mod:`verigeo.engine`.

Typical usage::

    from verigeo import CONFIG, generate_problem_from_VeriGraph, solve_problem_with_llm

    graph = GeometryGraph()
    problem = generate_problem_from_VeriGraph(graph, ...)
    solution = solve_problem_with_llm(problem, ...)
"""
from __future__ import annotations

from verigeo.config import (
    AUTHOR_FROM_SCRATCH,
    AUTHOR_WORKFLOW,
    CONFIG,
    CORE_SHAPE,
    LLMConfig,
    PROJECT_ROOT,
    require_api_key,
)
from verigeo.llm import hooks
from verigeo.prompts import load_prompt

# Core engine symbols — imported lazily via submodule to avoid forcing heavy
# matplotlib / sympy imports at ``import verigeo`` time if a caller only wants
# config.
from verigeo.engine import (
    GeometryGraph,
    VeriScriptAction,
    _apply_VeriScript_on_VeriGraph as apply_veriscript,
    _detect_VeriScript_and_VeriSolve_failures as detect_veriscript_failures,
    _set_blueprint_family_override as set_blueprint_family_override,
    generate_problem_from_VeriGraph,
    run_generation_from_config,
    run_parallel_author_solver_teacher,
    run_parallel_author_solver_teacher_matrix,
    solve_problem_with_llm,
)

# Legacy aliases (underscored names) for legacy back-compat.
_apply_VeriScript_on_VeriGraph = apply_veriscript
_detect_VeriScript_and_VeriSolve_failures = detect_veriscript_failures
_set_blueprint_family_override = set_blueprint_family_override


__all__ = [
    # config
    "CONFIG",
    "LLMConfig",
    "AUTHOR_WORKFLOW",
    "AUTHOR_FROM_SCRATCH",
    "CORE_SHAPE",
    "PROJECT_ROOT",
    "require_api_key",
    # hooks + prompts
    "hooks",
    "load_prompt",
    # engine entry points
    "GeometryGraph",
    "VeriScriptAction",
    "apply_veriscript",
    "detect_veriscript_failures",
    "set_blueprint_family_override",
    "generate_problem_from_VeriGraph",
    "solve_problem_with_llm",
    "run_generation_from_config",
    "run_parallel_author_solver_teacher",
    "run_parallel_author_solver_teacher_matrix",
    # legacy aliases
    "_apply_VeriScript_on_VeriGraph",
    "_detect_VeriScript_and_VeriSolve_failures",
    "_set_blueprint_family_override",
]
