# VeriGeo

A geometry-problem **data generation system** built around a multi-stage
LLM pipeline (author → solver → verifier). Each accepted problem ships with
its statement, a step-by-step proof, the underlying VeriScript actions, and
a rendered diagram, so the output is directly usable for downstream model
training, evaluation, and educational content authoring.

## Overview

VeriGeo executes the following pipeline per (model, domain, difficulty)
slot:

1. **Author** — an LLM extends a base figure (`VeriGraph`) with new
   geometric structure via a small DSL (`VeriScript`) and writes a
   problem statement, givens, goal, and proof outline.
2. **Solver** — a (possibly different) LLM proposes a VeriScript that
   reproduces the proof on the figure.
3. **Verifier** — the engine numerically executes the solver's
   VeriScript on the diagram and flags intersection / numeric failures.
4. **Repair** — failed attempts are returned to author/solver for one
   round of self-correction; final outcomes are recorded.

Outputs (per accepted attempt):

```
generated_data/<model>/problems/<domain>/<difficulty>/<attempt_id>/
├── problem.json          # full problem record
├── QA.json               # condensed (problem, proof) pair for training
├── author_output.json    # Author-stage trace
├── solver_output.json    # Solver-stage trace
├── graph_author.png      # diagram as drawn by Author
└── graph_solver.png      # diagram as redrawn by Solver
```

## Install

```bash
python -c "import sys; assert sys.version_info >= (3, 9)"
pip install -r requirements.txt
```

## Configure

Copy `.env.example` to `.env` and fill in the API key + endpoint of the
chat-completions service you want the engine to call. The engine speaks
the OpenAI-style `/v1/chat/completions` schema; Anthropic-style prompt
caching is auto-enabled for `claude-*` models via the hook layer.

```bash
cp .env.example .env
$EDITOR .env       # set VERIGEO_API_KEY and VERIGEO_API_ENDPOINT
```

For multi-model dataset generation, additionally populate the
`LLM_<NAME>_*` variables for whichever subset of models you want to
run (see `.env.example`).

> **Difficulty calibration.** `_DIFFICULTY_SHOTS` in `verigeo/engine.py`
> ships empty by default. To bias the author toward your own difficulty
> distribution, drop 2–4 reference problems into the `easy` / `medium` /
> `hard` lists; they will be inlined into the author prompt as
> in-context examples.

## Single-task generation

```bash
python -m verigeo
```

Reads `LLMConfig.matrix_counts` from `verigeo/config.py` and writes
problems under `<family>_<difficulty>/problem_<idx>/`.

## Multi-model dataset generation

```bash
# Full grid
python -m generate --run

# Subset
python -m generate --run --models qwen claude-opus \
    --attempts 5 --workers 4 \
    --domains vector_coord --difficulties easy medium

# Recompute statistics from an existing results.jsonl
python -m generate --stats
```

Results land in `generated_data/<model>/`:
- `results.jsonl` — one line per attempt (jobs are resumable).
- `run_config.json` — frozen run parameters (API keys redacted).
- `statistics.json` / `statistics.csv` — per-cell + per-model aggregates.
- `problems/<domain>/<difficulty>/<attempt_id>/` — accepted artifacts.

## Hook API

The engine exposes three integration points for harnesses that want to
adapt request bodies, observe responses, or gate prompt caching, without
monkey-patching engine internals:

```python
from verigeo.llm import hooks

hooks.register_request_preprocessor(my_fn)   # fn(body: dict, model: str) -> dict
hooks.register_response_observer(my_fn)      # fn(body, response, model) -> None
hooks.set_cache_policy(lambda model: "claude" in model.lower())
```

`generate/hooks.py` uses this API to (1) strip provider-incompatible
fields (`strict`, `cache_control`, image blocks) for non-Anthropic
backends, (2) feed token-usage into a thread-local accumulator, and
(3) only enable Anthropic prompt-caching for Claude models.

## Layout

```
verigeo/         # engine: graph + author + solver + pipeline
  engine.py      # consolidated core
  config.py      # LLMConfig + .env loader
  cli.py         # python -m verigeo entry point
  prompts/       # author / solver / review prompt resources
  llm/hooks.py   # request/response hook registry
generate/        # multi-model generation harness
  runner.py      # per-attempt + top-level loop
  models.py      # ModelConfig + LLM_<NAME>_* env loading
  stats.py       # cell/model aggregates + CSV
  hooks.py       # provider-compatibility wiring
  tokens.py      # thread-local token accumulator
blueprint_library.json
```

## License

MIT. See [LICENSE](LICENSE).
