"""Per-attempt execution + top-level generation loop.

Replaces the original pipeline.py's monkey-patch wiring with hook-based
integration (see :mod:`legacy.hooks`). Dead imports / variables from the
original have been removed.
"""
from __future__ import annotations

import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import verigeo
from verigeo import (
    CONFIG,
    GeometryGraph,
    VeriScriptAction,
    apply_veriscript,
    detect_veriscript_failures,
    generate_problem_from_VeriGraph,
    set_blueprint_family_override,
    solve_problem_with_llm,
)

from generate.constants import (
    BASE_SEED,
    DEFAULT_ATTEMPTS_PER_CELL,
    DEFAULT_MAX_WORKERS,
    DIFFICULTIES,
    DOMAINS,
    RESULTS_DIR,
    load_knowledge_points,
)
from generate.hooks import install as install_hooks
from generate.models import ModelConfig, load_model_configs
from generate.stats import compute_and_save_statistics
from generate.tokens import TokenAccumulator


# --------------------------------------------------------------------------- dataclass

@dataclass
class GenerationResult:
    attempt_id: str
    task_type: str
    model_name: str
    domain: str
    difficulty: str
    knowledge_points: List[str]
    outcome: str
    prompt_tokens: int
    completion_tokens: int
    wall_secs: float
    cost_usd: float
    num_llm_calls: int
    seed: int
    started_at: str
    finished_at: str
    cache_read_tokens: int = 0
    cache_create_tokens: int = 0
    error_message: Optional[str] = None


# --------------------------------------------------------------------------- helpers

def _now_local() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _compute_cost(prompt_tokens: int, completion_tokens: int, cfg: ModelConfig,
                  cache_read: int, cache_create: int) -> float:
    base_input = max(0, prompt_tokens - cache_read - cache_create)
    cost_in = base_input * cfg.cost_per_1m_input / 1_000_000
    cost_cache_read = cache_read * cfg.cost_per_1m_cache_read / 1_000_000
    cost_cache_create = cache_create * cfg.cost_per_1m_input * 1.25 / 1_000_000
    cost_out = completion_tokens * cfg.cost_per_1m_output / 1_000_000
    return round(cost_in + cost_cache_read + cost_cache_create + cost_out, 6)


def _sample_knowledge_points(domain: str, rng: random.Random, pool: List[str], k: int = 2) -> List[str]:
    if not pool:
        return []
    return rng.sample(pool, min(k, len(pool)))


def _make_attempt_id(model_name: str, domain: str, difficulty: str, idx: int) -> str:
    return f"{model_name}_{domain}_{difficulty}_{idx:03d}"


def _apply_model_config(model_cfg: ModelConfig) -> None:
    """Point the global engine CONFIG at the given model for this run."""
    CONFIG.endpoint = model_cfg.endpoint
    CONFIG.api_key = model_cfg.api_key
    CONFIG.model = model_cfg.model
    CONFIG.max_tokens = model_cfg.max_tokens
    CONFIG.temperature = model_cfg.temperature
    CONFIG.timeout_secs = model_cfg.timeout_secs


# --------------------------------------------------------------------------- single attempt

def run_single_attempt(model_cfg: ModelConfig, domain: str, difficulty: str,
                       kps: List[str], seed: int, attempt_id: str, out_dir: str) -> GenerationResult:
    TokenAccumulator.start()
    started_at = _now_local()
    start_wall = time.time()
    outcome = "final_reject"
    error_message: Optional[str] = None
    problem_json = None
    solver_output = None
    stage_outputs: dict = {}

    set_blueprint_family_override(domain)

    try:
        g = GeometryGraph()
        try:
            problem_json = generate_problem_from_VeriGraph(
                g,
                endpoint=model_cfg.endpoint,
                api_key=model_cfg.api_key,
                model=model_cfg.model,
                difficulty=difficulty,
                max_tokens=model_cfg.max_tokens,
                temperature=model_cfg.temperature,
                timeout_secs=model_cfg.timeout_secs,
                strict_validate=False,
                allow_author_edits=True,
                stage_outputs=stage_outputs,
                from_scratch=True,
            )

            solver_output = solve_problem_with_llm(
                g, problem_json,
                endpoint=model_cfg.endpoint,
                api_key=model_cfg.api_key,
                model=model_cfg.model,
                max_tokens=model_cfg.max_tokens,
                temperature=model_cfg.temperature,
                timeout_secs=model_cfg.timeout_secs,
            )

            try:
                actions = [VeriScriptAction(**a) for a in solver_output.get("actions", [])]
                solver_logs = apply_veriscript(g, actions, verify_only=True)
                if solver_logs:
                    solver_output = dict(solver_output)
                    solver_output["execution_logs"] = solver_logs
            except Exception:
                pass

            veriscript_failed, verisolve_failed = detect_veriscript_failures(stage_outputs, solver_output)

            author_modified = bool(((problem_json or {}).get("meta") or {}).get("reflection_modified"))
            solver_modified = bool((solver_output or {}).get("reflection_modified"))
            has_intercept = veriscript_failed or verisolve_failed

            if not has_intercept and not author_modified and not solver_modified:
                outcome = "direct_pass"
            else:
                outcome = "accepted_after_repair"

        except Exception as exc:
            outcome = "final_reject"
            error_message = str(exc)[:500]
    finally:
        set_blueprint_family_override(None)

    wall_secs = time.time() - start_wall
    prompt_tokens, completion_tokens, _, num_calls = TokenAccumulator.get_totals()
    cache_read, cache_create = TokenAccumulator.get_cache_totals()
    cost = _compute_cost(prompt_tokens, completion_tokens, model_cfg, cache_read, cache_create)

    if outcome != "final_reject" and isinstance(problem_json, dict):
        _persist_attempt_artifacts(
            out_dir, domain, difficulty, attempt_id, g, problem_json, stage_outputs, solver_output
        )

    return GenerationResult(
        attempt_id=attempt_id,
        task_type="basic",
        model_name=model_cfg.name,
        domain=domain,
        difficulty=difficulty,
        knowledge_points=kps,
        outcome=outcome,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        wall_secs=round(wall_secs, 2),
        cost_usd=cost,
        num_llm_calls=num_calls,
        seed=seed,
        started_at=started_at,
        finished_at=_now_local(),
        cache_read_tokens=cache_read,
        cache_create_tokens=cache_create,
        error_message=error_message,
    )


def _persist_attempt_artifacts(out_dir: str, domain: str, difficulty: str, attempt_id: str,
                               graph, problem_json: dict, stage_outputs: dict,
                               solver_output: Optional[dict]) -> None:
    job_dir = os.path.join(out_dir, "problems", domain, difficulty, attempt_id)
    os.makedirs(job_dir, exist_ok=True)
    try:
        with open(os.path.join(job_dir, "problem.json"), "w", encoding="utf-8") as fh:
            json.dump(problem_json, fh, ensure_ascii=False, indent=2)
        with open(os.path.join(job_dir, "author_output.json"), "w", encoding="utf-8") as fh:
            json.dump(stage_outputs or {"note": "no stage outputs"}, fh, ensure_ascii=False, indent=2)
        if isinstance(solver_output, dict):
            with open(os.path.join(job_dir, "solver_output.json"), "w", encoding="utf-8") as fh:
                json.dump(solver_output, fh, ensure_ascii=False, indent=2)

        qa_problem = None
        if isinstance(stage_outputs, dict):
            qa_problem = stage_outputs.get("problem")
            if isinstance(qa_problem, dict):
                qa_problem = qa_problem.get("problem")
        if qa_problem is None:
            inner = problem_json.get("problem")
            if isinstance(inner, dict):
                qa_problem = inner.get("problem")
        qa_proof = solver_output.get("proof") if isinstance(solver_output, dict) else None
        with open(os.path.join(job_dir, "QA.json"), "w", encoding="utf-8") as fh:
            json.dump({"problem": qa_problem, "proof": qa_proof}, fh, ensure_ascii=False, indent=2)

        try:
            graph.render(save_path=os.path.join(job_dir, "graph_author.png"), show=False)
            graph.render(save_path=os.path.join(job_dir, "graph_solver.png"), show=False)
        except Exception:
            pass
    except Exception:
        pass


# --------------------------------------------------------------------------- results IO

_RESULTS_LOCK = threading.Lock()


def _append_result(result: GenerationResult, path: str) -> None:
    line = json.dumps(asdict(result), ensure_ascii=False)
    with _RESULTS_LOCK:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def _load_completed_ids(path: str) -> set:
    completed: set = set()
    if not os.path.exists(path):
        return completed
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                completed.add(json.loads(raw)["attempt_id"])
            except Exception:
                pass
    return completed


# --------------------------------------------------------------------------- top-level loop

def run_generation(model_names: Optional[List[str]] = None,
                  attempts_per_cell: int = DEFAULT_ATTEMPTS_PER_CELL,
                  max_workers: int = DEFAULT_MAX_WORKERS,
                  domains: Optional[List[str]] = None,
                  difficulties: Optional[List[str]] = None) -> None:
    install_hooks()

    run_domains = domains or DOMAINS
    run_difficulties = difficulties or DIFFICULTIES
    configs = load_model_configs(model_names)
    if not configs:
        print("[ERROR] No valid model configurations found. Check your environment variables.")
        return

    kp_pool = load_knowledge_points()

    for cfg in configs:
        os.makedirs(os.path.join(RESULTS_DIR, cfg.model), exist_ok=True)

    total_jobs = len(configs) * len(run_domains) * len(run_difficulties) * attempts_per_cell
    print("[GEN] Starting generation run")
    print(f"[GEN] Models: {[c.name for c in configs]}")
    print(f"[GEN] Domains: {run_domains}")
    print(f"[GEN] Difficulties: {run_difficulties}")
    print(f"[GEN] Attempts per cell: {attempts_per_cell}")
    print(f"[GEN] Total jobs: {total_jobs}")
    print(f"[GEN] Concurrent workers: {max_workers}")

    for model_cfg in configs:
        model_dir = os.path.join(RESULTS_DIR, model_cfg.model)
        results_path = os.path.join(model_dir, "results.jsonl")

        print(f"\n{'='*60}")
        start_utc = _now_utc()
        print(f"[GEN] Begin model: {model_cfg.name} (model={model_cfg.model})")
        print(f"[GEN] Started at (UTC): {start_utc}")
        print(f"{'='*60}")

        _apply_model_config(model_cfg)
        completed_ids = _load_completed_ids(results_path)
        if completed_ids:
            print(f"[GEN] {len(completed_ids)} attempts already completed; will skip.")

        run_config = {
            "models": [asdict(c) for c in configs],
            "domains": DOMAINS,
            "difficulties": DIFFICULTIES,
            "attempts_per_cell": attempts_per_cell,
            "max_workers": max_workers,
            "base_seed": BASE_SEED,
            "started_at": _now_local(),
            "started_at_utc": start_utc,
        }
        for m in run_config["models"]:
            if len(m["api_key"]) > 8:
                m["api_key"] = m["api_key"][:8] + "..."
            else:
                m["api_key"] = "***"
        config_path = os.path.join(model_dir, "run_config.json")
        with open(config_path, "w", encoding="utf-8") as fh:
            json.dump(run_config, fh, ensure_ascii=False, indent=2)

        jobs = []
        for domain in run_domains:
            for difficulty in run_difficulties:
                for idx in range(attempts_per_cell):
                    attempt_id = _make_attempt_id(model_cfg.name, domain, difficulty, idx)
                    if attempt_id in completed_ids:
                        continue
                    seed = BASE_SEED + hash((model_cfg.name, domain, difficulty, idx)) % 100000
                    rng = random.Random(seed)
                    kps = _sample_knowledge_points(domain, rng, kp_pool.get(domain, []))
                    jobs.append((attempt_id, domain, difficulty, kps, seed))

        if not jobs:
            print(f"[GEN] All jobs already done for model {model_cfg.name}; skipping.")
            continue

        print(f"[GEN] Pending jobs: {len(jobs)}")
        success = fails = 0

        def _run_job(job_tuple):
            attempt_id, domain, difficulty, kps, seed = job_tuple
            return run_single_attempt(model_cfg, domain, difficulty, kps, seed, attempt_id, model_dir)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_job = {pool.submit(_run_job, job): job for job in jobs}
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                attempt_id = job[0]
                try:
                    result = future.result()
                except Exception as exc:
                    result = GenerationResult(
                        attempt_id=attempt_id,
                        task_type="basic",
                        model_name=model_cfg.name,
                        domain=job[1], difficulty=job[2], knowledge_points=job[3],
                        outcome="final_reject",
                        prompt_tokens=0, completion_tokens=0, wall_secs=0.0, cost_usd=0.0,
                        num_llm_calls=0, seed=job[4],
                        started_at=_now_local(), finished_at=_now_local(),
                        error_message=str(exc)[:500],
                    )
                    _append_result(result, results_path)
                    print(f"  [ERROR] {attempt_id}: {str(exc)[:100]}")
                    fails += 1
                    continue

                _append_result(result, results_path)
                if result.outcome == "final_reject":
                    fails += 1
                    status = "REJECT"
                elif result.outcome == "direct_pass":
                    success += 1
                    status = "PASS"
                else:
                    success += 1
                    status = "REPAIR"
                print(
                    f"  [{status}] {attempt_id}"
                    f" | tokens={result.prompt_tokens + result.completion_tokens}"
                    f" | time={result.wall_secs:.1f}s"
                    f" | cost=${result.cost_usd:.4f}"
                )

        end_utc = _now_utc()
        print(f"\n[GEN] Model {model_cfg.name} done: accepted={success}, rejected={fails}")
        print(f"[GEN] Started at (UTC): {start_utc}")
        print(f"[GEN] Finished at (UTC): {end_utc}")

        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                rc = json.load(fh)
            rc["finished_at_utc"] = end_utc
            rc["started_at_utc"] = start_utc
            with open(config_path, "w", encoding="utf-8") as fh:
                json.dump(rc, fh, ensure_ascii=False, indent=2)
        except Exception:
            pass

    print("\n[GEN] All models finished. Computing statistics...")
    for cfg in configs:
        model_dir = os.path.join(RESULTS_DIR, cfg.model)
        compute_and_save_statistics(os.path.join(model_dir, "results.jsonl"), model_dir)


__all__ = ["GenerationResult", "run_single_attempt", "run_generation"]
