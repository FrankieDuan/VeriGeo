"""``python -m generate`` entry point."""
from __future__ import annotations

import argparse
import os

from generate.constants import (
    DEFAULT_ATTEMPTS_PER_CELL,
    DEFAULT_MAX_WORKERS,
    DIFFICULTIES,
    DOMAINS,
    RESULTS_DIR,
)
from generate.models import available_model_names, load_model_configs
from generate.runner import run_generation
from generate.stats import compute_and_save_statistics


def main() -> int:
    available = ", ".join(available_model_names())
    parser = argparse.ArgumentParser(
        description="VeriGeo data generation: multi-model geometry-problem pipeline."
    )
    parser.add_argument("--run", action="store_true", help="run the generation pipeline")
    parser.add_argument("--stats", action="store_true",
                        help="recompute statistics from an existing results.jsonl")
    parser.add_argument("--models", nargs="+",
                        help=f"models to run; choose from: {available}")
    parser.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS_PER_CELL,
                        help=f"attempts per cell (default {DEFAULT_ATTEMPTS_PER_CELL})")
    parser.add_argument("--workers", type=int, default=DEFAULT_MAX_WORKERS,
                        help=f"concurrent workers (default {DEFAULT_MAX_WORKERS})")
    parser.add_argument("--domains", nargs="+",
                        help=f"domains to run; choose from: {', '.join(DOMAINS)}")
    parser.add_argument("--difficulties", nargs="+",
                        help=f"difficulties to run; choose from: {', '.join(DIFFICULTIES)}")
    args = parser.parse_args()

    if args.stats:
        configs = load_model_configs(args.models)
        if not configs:
            print("[STATS] No model configs found.")
            return 1
        for cfg in configs:
            model_dir = os.path.join(RESULTS_DIR, cfg.model)
            results_path = os.path.join(model_dir, "results.jsonl")
            if os.path.exists(results_path):
                compute_and_save_statistics(results_path, model_dir)
            else:
                print(f"[STATS] Not found: {results_path}")
        return 0

    if args.run:
        run_generation(
            model_names=args.models,
            attempts_per_cell=args.attempts,
            max_workers=args.workers,
            domains=args.domains,
            difficulties=args.difficulties,
        )
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
