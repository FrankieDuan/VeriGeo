"""Shared constants + knowledge-point loading."""
from __future__ import annotations

import json
import os
from typing import Dict, List

from verigeo.config import PROJECT_ROOT

DOMAINS = ["euclid", "vector_coord", "function_calculus"]
DIFFICULTIES = ["easy", "medium", "hard"]

DEFAULT_ATTEMPTS_PER_CELL = 50
DEFAULT_MAX_WORKERS = 10
BASE_SEED = 20260420

RESULTS_DIR = os.path.join(PROJECT_ROOT, "generated_data")
BLUEPRINT_LIBRARY_PATH = os.path.join(PROJECT_ROOT, "blueprint_library.json")


def load_knowledge_points() -> Dict[str, List[str]]:
    """Load motif pool from ``blueprint_library.json`` keyed by domain."""
    try:
        with open(BLUEPRINT_LIBRARY_PATH, "r", encoding="utf-8") as fh:
            lib = json.load(fh)
        return {fam: lib.get("motifs", {}).get(fam, []) for fam in DOMAINS}
    except Exception:
        return {fam: [] for fam in DOMAINS}


__all__ = [
    "DOMAINS",
    "DIFFICULTIES",
    "DEFAULT_ATTEMPTS_PER_CELL",
    "DEFAULT_MAX_WORKERS",
    "BASE_SEED",
    "RESULTS_DIR",
    "BLUEPRINT_LIBRARY_PATH",
    "load_knowledge_points",
]
