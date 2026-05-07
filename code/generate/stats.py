"""Benchmark statistics aggregation."""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from generate.constants import DIFFICULTIES, DOMAINS, load_knowledge_points


def _parse_cell_key(cell_key: str) -> tuple:
    """Split ``<domain>_<difficulty>`` — domain may itself contain underscores."""
    for diff in DIFFICULTIES:
        suffix = "_" + diff
        if cell_key.endswith(suffix):
            return cell_key[: -len(suffix)], diff
    # fallback
    head, _, tail = cell_key.rpartition("_")
    return head, tail


def _load_results(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError:
                pass
    return rows


def _categorize_error(msg: str) -> str:
    if not msg:
        return "unknown"
    lower = msg.lower()
    if "401" in msg or "403" in msg or "invalid token" in lower:
        return "auth_error"
    if "429" in msg or "rate" in lower:
        return "rate_limit"
    if any(code in msg for code in ("500", "502", "503")):
        return "server_error"
    if "timeout" in lower or "timed out" in lower:
        return "timeout"
    if "VeriScript" in msg:
        return "veriscript_error"
    if "VeriSolve" in msg:
        return "verisolve_error"
    return "other_error"


def _compute_cell_stats(cell_rows: List[dict]) -> dict:
    total = len(cell_rows)
    by_outcome = {"direct_pass": [], "accepted_after_repair": [], "final_reject": []}
    for r in cell_rows:
        by_outcome.setdefault(r.get("outcome", ""), []).append(r)
    accepted = by_outcome["direct_pass"] + by_outcome["accepted_after_repair"]

    avg_tokens = (
        round(sum(r.get("prompt_tokens", 0) + r.get("completion_tokens", 0) for r in accepted) / len(accepted), 1)
        if accepted else 0
    )
    avg_time = round(sum(r.get("wall_secs", 0) for r in accepted) / len(accepted), 2) if accepted else 0
    avg_cost = round(sum(r.get("cost_usd", 0) for r in accepted) / len(accepted), 6) if accepted else 0

    stat = {
        "total": total,
        "direct_pass": len(by_outcome["direct_pass"]),
        "accepted_after_repair": len(by_outcome["accepted_after_repair"]),
        "final_reject": len(by_outcome["final_reject"]),
        "avg_tokens": avg_tokens,
        "avg_time": avg_time,
        "avg_cost": avg_cost,
    }

    if cell_rows:
        times_sorted = sorted(r.get("wall_secs", 0) for r in cell_rows)
        n = len(times_sorted)
        stat["time_p25"] = round(times_sorted[int(n * 0.25)], 2)
        stat["time_p50"] = round(times_sorted[int(n * 0.50)], 2)
        stat["time_p75"] = round(times_sorted[int(n * 0.75)], 2)
        stat["time_p95"] = round(times_sorted[min(int(n * 0.95), n - 1)], 2)

    total_prompt = sum(r.get("prompt_tokens", 0) for r in cell_rows)
    stat["total_cache_read_tokens"] = sum(r.get("cache_read_tokens", 0) for r in cell_rows)
    stat["total_cache_create_tokens"] = sum(r.get("cache_create_tokens", 0) for r in cell_rows)
    stat["cache_read_rate"] = round(stat["total_cache_read_tokens"] / total_prompt, 4) if total_prompt else 0

    errors: Dict[str, int] = {}
    for r in by_outcome["final_reject"]:
        cat = _categorize_error(r.get("error_message", "") or "")
        errors[cat] = errors.get(cat, 0) + 1
    stat["error_categories"] = errors

    return stat


def _compute_model_stats(rows: List[dict], kp_pool: Dict[str, List[str]]) -> dict:
    total = len(rows)
    by_outcome = {"direct_pass": [], "accepted_after_repair": [], "final_reject": []}
    for r in rows:
        by_outcome.setdefault(r.get("outcome", ""), []).append(r)
    accepted = by_outcome["direct_pass"] + by_outcome["accepted_after_repair"]

    stats = {
        "total_attempts": total,
        "direct_pass_count": len(by_outcome["direct_pass"]),
        "direct_pass_rate": round(len(by_outcome["direct_pass"]) / total, 4) if total else 0,
        "accepted_after_repair_count": len(by_outcome["accepted_after_repair"]),
        "accepted_after_repair_rate": round(len(by_outcome["accepted_after_repair"]) / total, 4) if total else 0,
        "final_reject_count": len(by_outcome["final_reject"]),
        "final_reject_rate": round(len(by_outcome["final_reject"]) / total, 4) if total else 0,
    }

    if accepted:
        stats["avg_tokens_per_accepted"] = round(
            sum(r.get("prompt_tokens", 0) + r.get("completion_tokens", 0) for r in accepted) / len(accepted), 1
        )
        stats["avg_time_per_accepted"] = round(sum(r.get("wall_secs", 0) for r in accepted) / len(accepted), 2)
        stats["avg_cost_per_accepted"] = round(sum(r.get("cost_usd", 0) for r in accepted) / len(accepted), 6)
    else:
        stats["avg_tokens_per_accepted"] = 0
        stats["avg_time_per_accepted"] = 0
        stats["avg_cost_per_accepted"] = 0

    cells = {}
    for domain in DOMAINS:
        for difficulty in DIFFICULTIES:
            key = f"{domain}_{difficulty}"
            cell_rows = [r for r in rows if r.get("domain") == domain and r.get("difficulty") == difficulty]
            cells[key] = _compute_cell_stats(cell_rows)
    stats["cells"] = cells

    kp_coverage = {}
    for domain in DOMAINS:
        pool = set(kp_pool.get(domain, []))
        covered: set = set()
        for r in rows:
            if r.get("domain") == domain and r.get("outcome") != "final_reject":
                for kp in r.get("knowledge_points", []):
                    covered.add(kp)
        kp_coverage[domain] = {
            "total_pool": len(pool),
            "covered": len(covered & pool),
            "coverage_rate": round(len(covered & pool) / len(pool), 4) if pool else 0,
            "uncovered": sorted(pool - covered),
            "covered_list": sorted(covered & pool),
        }
    stats["knowledge_point_coverage"] = kp_coverage

    total_prompt_all = sum(r.get("prompt_tokens", 0) for r in rows)
    stats["total_cache_read_tokens"] = sum(r.get("cache_read_tokens", 0) for r in rows)
    stats["total_cache_create_tokens"] = sum(r.get("cache_create_tokens", 0) for r in rows)
    stats["cache_read_rate"] = (
        round(stats["total_cache_read_tokens"] / total_prompt_all, 4) if total_prompt_all else 0
    )
    stats["total_prompt_tokens"] = total_prompt_all
    stats["total_completion_tokens"] = sum(r.get("completion_tokens", 0) for r in rows)
    stats["total_cost_usd"] = round(sum(r.get("cost_usd", 0) for r in rows), 4)
    return stats


def _write_csv(stats: Dict[str, dict], path: str) -> None:
    header = "model,domain,difficulty,total,direct_pass,accepted_after_repair,final_reject,avg_tokens,avg_time_sec,avg_cost_usd"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(header + "\n")
        for model_name, ms in stats.items():
            for cell_key, cell in ms.get("cells", {}).items():
                domain, difficulty = _parse_cell_key(cell_key)
                row = [
                    model_name, domain, difficulty,
                    str(cell.get("total", "")),
                    str(cell.get("direct_pass", "")),
                    str(cell.get("accepted_after_repair", "")),
                    str(cell.get("final_reject", "")),
                    str(cell.get("avg_tokens", "")),
                    str(cell.get("avg_time", "")),
                    str(cell.get("avg_cost", "")),
                ]
                fh.write(",".join(row) + "\n")
            fh.write(",".join([
                model_name, "ALL", "ALL",
                str(ms.get("total_attempts", "")),
                str(ms.get("direct_pass_count", "")),
                str(ms.get("accepted_after_repair_count", "")),
                str(ms.get("final_reject_count", "")),
                "", "", "",
            ]) + "\n")


def _print_summary(stats: Dict[str, dict]) -> None:
    print("\n" + "=" * 70)
    print("VeriGeo generation statistics")
    print("=" * 70)
    for model_name, ms in stats.items():
        print(f"\n  Model: {model_name}")
        print(f"    Total attempts: {ms['total_attempts']}")
        print(f"    Direct pass: {ms['direct_pass_count']} ({ms['direct_pass_rate']*100:.1f}%)")
        print(f"    Accepted after repair: {ms['accepted_after_repair_count']} ({ms['accepted_after_repair_rate']*100:.1f}%)")
        print(f"    Final reject: {ms['final_reject_count']} ({ms['final_reject_rate']*100:.1f}%)")
        print(f"    Avg tokens / accepted problem: {ms['avg_tokens_per_accepted']:.0f}")
        print(f"    Avg time / accepted problem: {ms['avg_time_per_accepted']:.1f}s")
        print(f"    Avg cost / accepted problem: ${ms['avg_cost_per_accepted']:.4f}")
        print(f"    Total cost: ${ms['total_cost_usd']:.4f}")
        for dom, info in ms.get("knowledge_point_coverage", {}).items():
            print(f"    Knowledge-point coverage ({dom}): {info['covered']}/{info['total_pool']} ({info['coverage_rate']*100:.1f}%)")


def compute_and_save_statistics(results_path: str, out_dir: str) -> Optional[Dict[str, dict]]:
    rows = _load_results(results_path)
    if not rows:
        print("[STATS] No result data found; skipping statistics.")
        return None

    kp_pool = load_knowledge_points()
    by_model: Dict[str, List[dict]] = {}
    for r in rows:
        by_model.setdefault(r.get("model_name", "unknown"), []).append(r)

    stats = {name: _compute_model_stats(model_rows, kp_pool) for name, model_rows in by_model.items()}

    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "statistics.json")
    csv_path = os.path.join(out_dir, "statistics.csv")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)
    _write_csv(stats, csv_path)
    print(f"[STATS] Statistics JSON saved: {json_path}")
    print(f"[STATS] Statistics CSV saved: {csv_path}")
    _print_summary(stats)
    return stats


__all__ = ["compute_and_save_statistics"]
