"""Multi-model geometry-problem generation pipeline.

Runs the VeriGeo author->solver->verify pipeline over a (model, domain, difficulty)
grid and writes one folder per accepted problem (problem.json / QA.json /
graph_*.png) for downstream training and educational use.
"""
from generate.runner import GenerationResult, run_generation, run_single_attempt
from generate.stats import compute_and_save_statistics

__all__ = [
    "GenerationResult",
    "run_generation",
    "run_single_attempt",
    "compute_and_save_statistics",
]
