"""VeriGeo engine core.

This module contains the consolidated generation / solver / author pipeline.
The public API is re-exported from :mod:`verigeo` — import from there rather
than reaching into this module directly.
"""
# flake8: noqa  — large monolithic module; stylistic cleanup is out of scope.

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple, Callable
from collections import defaultdict, Counter
from itertools import combinations
import json, uuid, random, math, os, copy, ast, base64, tempfile, traceback, hashlib, time, threading, shutil
import multiprocessing as mp
from fractions import Fraction
import numpy as np
import matplotlib
matplotlib.rcParams["path.simplify"] = False
matplotlib.rcParams["path.simplify_threshold"] = 0.0
import sys
sys.stdout.reconfigure(encoding='utf-8')

DEFAULT_MPL_BACKEND = matplotlib.get_backend()
HEADLESS_MPL_BACKEND = os.environ.get("MPL_HEADLESS_BACKEND", "Agg")
GUI_MPL_BACKEND = os.environ.get("MPL_GUI_BACKEND", DEFAULT_MPL_BACKEND)
try:
    if HEADLESS_MPL_BACKEND and HEADLESS_MPL_BACKEND.lower() != DEFAULT_MPL_BACKEND.lower():
        matplotlib.use(HEADLESS_MPL_BACKEND)
except Exception as _mpl_backend_err:
    print(f"[WARN] Matplotlib backend switch failed; will continue using {DEFAULT_MPL_BACKEND}: {_mpl_backend_err}")
HEADLESS_ACTIVE_BACKEND = matplotlib.get_backend()
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.patches import Arc, Circle as MPLCircle
# ==== [A] LLM authoring: minimal-dependency implementation (no extra installs needed) ====
import requests
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait, FIRST_COMPLETED
# ==== LLM config (centralized in verigeo.config) ====
from verigeo.config import (
    CONFIG,
    LLMConfig,
    AUTHOR_WORKFLOW,
    AUTHOR_FROM_SCRATCH,
    CORE_SHAPE,
    PROJECT_ROOT,
    require_api_key,
    _DEFAULT_SCRATCH_CONSTRUCTION_TARGETS,
)
from verigeo.prompts import load_prompt
from verigeo.llm import hooks

# ===== Blueprint generation grammar: program-side random sampling (injected into prompt) =====
_BLUEPRINT_LIBRARY_DEFAULT = {
    "families": [
        "segment",
    ],
    "motifs": {

        "segment": ["Pythagorean Theorem"],

    },
    "goal_types": [
        "concyclic", "collinear", "concurrent", "equal-angle", "equal-ratio", "area-ratio", "power-equation",
        "invariant", "similar", "congruent", "vector-relation",
    ],
}

_BLUEPRINT_LIBRARY_DISABLED = {
    "families": [],
    "motifs": {},
    "goal_types": [],
}

_BLUEPRINT_LIBRARY_CACHE: Optional[Dict[str, Any]] = None
_BLUEPRINT_RNG: Optional[random.Random] = None
_BLUEPRINT_FAMILY_OVERRIDE = threading.local()
_BLUEPRINT_STATS = {
    "family": Counter(),
    "motif": Counter(),
    "goal": Counter(),
}


def _strip_code_fence(text: str) -> str:
    """Strip surrounding ``` / ```json markdown fences from a model response."""
    s = text.strip()
    if not s.startswith("```"):
        return s
    lines = s.split("\n")
    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def _extract_first_json_object(text: str) -> str:
    """Return the first balanced ``{...}`` block in ``text``.

    LLMs sometimes emit trailing prose after the JSON payload, which makes
    ``json.loads`` raise ``Extra data: line N column 1``. This walks the
    string respecting string literals and backslash escapes, returning the
    substring covering the first top-level object. Raises ``ValueError``
    if no balanced object is found.
    """
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start:i + 1]
    raise ValueError("no balanced JSON object found in text")


def _loads_lenient(text: str) -> Any:
    """``json.loads`` with fence stripping + tolerant trailing-text handling."""
    cleaned = _strip_code_fence(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return json.loads(_extract_first_json_object(cleaned))


def _load_blueprint_library() -> Dict[str, Any]:
    path = (CONFIG.blueprint_library_path or "").strip()
    if not path:
        if CONFIG.blueprint_enable:
            raise FileNotFoundError("Blueprint library path is empty; cannot read blueprint library.")
        return copy.deepcopy(_BLUEPRINT_LIBRARY_DEFAULT)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        if CONFIG.blueprint_enable:
            raise RuntimeError(f"Blueprint library read failed: {exc}") from exc
        print(f"[WARN] Blueprint library read failed; using built-in default library: {exc}")
        return copy.deepcopy(_BLUEPRINT_LIBRARY_DEFAULT)
    if isinstance(data, dict) and data.get("families") and data.get("motifs"):
        return data
    if CONFIG.blueprint_enable:
        raise ValueError("Blueprint library format error: must contain families and motifs.")
    print("[WARN] Blueprint library format error; using built-in default library.")
    return copy.deepcopy(_BLUEPRINT_LIBRARY_DEFAULT)


def _get_blueprint_library() -> Dict[str, Any]:
    global _BLUEPRINT_LIBRARY_CACHE
    if not CONFIG.blueprint_enable:
        return copy.deepcopy(_BLUEPRINT_LIBRARY_DISABLED)
    if _BLUEPRINT_LIBRARY_CACHE is None or CONFIG.blueprint_library_reload:
        _BLUEPRINT_LIBRARY_CACHE = _load_blueprint_library()
    return _BLUEPRINT_LIBRARY_CACHE


def _get_blueprint_rng():
    global _BLUEPRINT_RNG
    if CONFIG.blueprint_seed is None:
        return random
    if _BLUEPRINT_RNG is None:
        _BLUEPRINT_RNG = random.Random(CONFIG.blueprint_seed)
    return _BLUEPRINT_RNG


def _get_blueprint_family_override() -> Optional[str]:
    return getattr(_BLUEPRINT_FAMILY_OVERRIDE, "value", None)


def _set_blueprint_family_override(value: Optional[str]) -> None:
    _BLUEPRINT_FAMILY_OVERRIDE.value = value


def _resolve_blueprint_family(requested: str, families: List[str], motifs_map: Dict[str, Any]) -> Optional[str]:
    if not requested:
        return None
    trimmed = requested.strip()
    if not trimmed:
        return None
    if trimmed in families or trimmed in motifs_map:
        return trimmed
    lowered = trimmed.lower()
    for fam in families:
        if fam.lower() == lowered:
            return fam
    for fam in motifs_map.keys():
        if fam.lower() == lowered:
            return fam
    return None


def _pick_blueprint_family(lib: Dict[str, Any], rng) -> str:
    families = list(lib.get("families") or [])
    motifs_map = lib.get("motifs") or {}
    override = _get_blueprint_family_override()
    requested_raw = (override if override is not None else (CONFIG.blueprint_family or "")).strip()
    requested = _resolve_blueprint_family(requested_raw, families, motifs_map)
    if requested:
        return requested
    if requested_raw:
        print(f"[WARN] blueprint_family '{requested_raw}' not found; fallback to random family.")
    if families:
        return _balanced_pick(families, _BLUEPRINT_STATS["family"], rng, k=1)[0]
    if motifs_map:
        return rng.choice(list(motifs_map.keys()))
    return "unspecified"


def _pick_family_motifs(family: str, lib: Dict[str, Any], rng, k: int = 2) -> List[str]:
    motifs_map = lib.get("motifs") or {}
    motif_list = motifs_map.get(family)
    if not motif_list:
        motif_list = ["basic motif A", "basic motif B"]
    if not isinstance(motif_list, list):
        motif_list = list(motif_list)
    key_prefix = f"{family}::"
    motif_counts = Counter({m: _BLUEPRINT_STATS["motif"].get(key_prefix + m, 0) for m in motif_list})
    picks = _balanced_pick(motif_list, motif_counts, rng, k=k)
    if motif_list and len(picks) < k:
        while len(picks) < k:
            picks.append(motif_list[len(picks) % len(motif_list)])
    return picks


def _select_family_and_motifs(*, k: int = 2, update_stats: bool = True) -> Tuple[str, List[str]]:
    rng = _get_blueprint_rng()
    lib = _get_blueprint_library()
    family = _pick_blueprint_family(lib, rng)
    motifs = _pick_family_motifs(family, lib, rng, k=k)
    if update_stats and family:
        _BLUEPRINT_STATS["family"][family] += 1
        for motif in motifs:
            _BLUEPRINT_STATS["motif"][f"{family}::{motif}"] += 1
    return family, motifs


def _extract_blueprint_motifs(blueprint: Dict[str, Any]) -> List[str]:
    motifs = blueprint.get("motifs")
    if isinstance(motifs, dict):
        collected: List[str] = []
        for value in motifs.values():
            if isinstance(value, list):
                collected.extend(value)
            elif value:
                collected.append(str(value))
        return collected
    if isinstance(motifs, list):
        return [str(item) for item in motifs if item]
    if isinstance(motifs, str):
        return [motifs]
    return []


def _pick_required_geometry_topic(blueprint: Optional[Dict[str, Any]] = None) -> str:
    if blueprint:
        motifs = _extract_blueprint_motifs(blueprint)
        if motifs:
            return ", ".join(motifs)
    family, motifs = _select_family_and_motifs(k=2, update_stats=True)
    if motifs:
        return ", ".join(motifs)
    return family or "unspecified"


_DIFFICULTY_SHOTS: Dict[str, List[str]] = {
    # Users: fill in 2-4 reference problems per difficulty level. Each string
    # should be a complete problem statement; they are joined into the author
    # prompt as in-context calibration examples. If left empty, the prompt
    # falls back to a plain "Difficulty level: {level}." label.
    "easy": [],
    "medium": [],
    "hard": [],
}


def _format_difficulty_shots(difficulty: str) -> str:
    level = (difficulty or "").strip().lower()
    if level not in _DIFFICULTY_SHOTS:
        level = "medium"
    shots = _DIFFICULTY_SHOTS[level]
    if not shots:
        return f"Difficulty level: {level}."
    examples = "\n\n".join(f"Example {i+1}:\n{s}" for i, s in enumerate(shots))
    return (
        f"Difficulty level: {level}.\n\n"
        "Here are reference problems at this difficulty level:\n\n"
        f"{examples}\n\n"
        "Generate a problem whose difficulty is comparable to the examples above."
    )


def _balanced_pick(items: List[str], counter: Counter, rng, k: int = 1) -> List[str]:
    if not items:
        return []
    pool = list(items)
    picks: List[str] = []
    for _ in range(min(k, len(pool))):
        min_count = min(counter.get(item, 0) for item in pool)
        candidates = [item for item in pool if counter.get(item, 0) == min_count]
        pick = rng.choice(candidates)
        picks.append(pick)
        pool.remove(pick)
    return picks


def _sample_blueprint() -> Dict[str, Any]:
    rng = _get_blueprint_rng()
    family, motifs_list = _select_family_and_motifs(k=2, update_stats=True)
    blueprint_id = f"BP-{rng.randint(1000, 9999)}"
    return {
        "id": blueprint_id,
        "families": [family] if family else [],
        "motifs": {family: motifs_list},
    }


def _format_blueprint_text(blueprint: Dict[str, Any]) -> str:
    families = blueprint.get("families") or []
    fam_text = " + ".join(families)
    motifs = blueprint.get("motifs") or {}
    if isinstance(motifs, dict):
        motif_text = "; ".join(
            f"{fam}:{', '.join(motifs.get(fam, [])) if isinstance(motifs.get(fam), list) else motifs.get(fam, '')}"
            for fam in families
        )
    elif isinstance(motifs, list):
        motif_text = ", ".join(str(item) for item in motifs if item)
    else:
        motif_text = str(motifs) if motifs else ""
    return "\n".join([
        f"Blueprint ID: {blueprint.get('id')}",
        f"Family: {fam_text}",
        f"Motif: {motif_text}",
    ])

# ======================================================================


# ==== Numeric utilities: quantized sampling + rational/sqrt formatting ==================================
_QUANTIZED_STEPS = (1.0, 0.5, 0.25, 0.125, 0.0625)
_PYTHAGOREAN_DIRECTIONS = [
    (1, 0), (0, 1), (-1, 0), (0, -1),
    (3, 4), (-3, 4), (4, 3), (-4, 3)
]


def _render_graph_to_data_url(VeriGraph, prefix: str) -> Optional[str]:
    """Render a graph to PNG and encode as base64 data URL."""
    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, f"{prefix}_{uuid.uuid4().hex}.png")
    try:
        VeriGraph.render(save_path=tmp_path, show=False)
        with open(tmp_path, "rb") as fh:
            data = base64.b64encode(fh.read()).decode("ascii")
        return "data:image/png;base64," + data
    except Exception:
        return None
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _build_prompt_image_block(VeriGraph, prefix: str) -> Optional[Dict[str, Any]]:
    """Small helper for building vision content payload."""
    data_url = _render_graph_to_data_url(VeriGraph, prefix)
    if not data_url:
        return None
    return {"type": "image_url", "image_url": {"url": data_url}}


def _quantized_sample(range_tuple, steps=_QUANTIZED_STEPS, *, rng=None):
    """In the given interval prefer integers/simple fractions; fall back to uniform distribution and rationalize when necessary"""
    rng = rng or random
    lo, hi = range_tuple
    if lo > hi:
        lo, hi = hi, lo

    candidates: List[Fraction] = []
    seen = set()
    for step in steps:
        if step <= 0:
            continue
        denom = int(round(1.0 / step))
        if denom <= 0:
            continue
        start = math.ceil(lo * denom - 1e-9)
        end = math.floor(hi * denom + 1e-9)
        if start > end:
            continue
        for num in range(start, end + 1):
            frac = Fraction(num, denom)
            key = (frac.numerator, frac.denominator)
            if key in seen:
                continue
            val = float(frac)
            if lo - 1e-9 <= val <= hi + 1e-9:
                seen.add(key)
                candidates.append(frac)

    if candidates:
        ints = [f for f in candidates if f.denominator == 1]
        if ints:
            chosen = rng.choice(ints)
        else:
            best_denom = min(f.denominator for f in candidates)
            pool = [f for f in candidates if f.denominator == best_denom]
            chosen = rng.choice(pool)
        return float(chosen)

    approx = Fraction(rng.uniform(lo, hi)).limit_denominator(128)
    return float(approx)


def _format_fraction(value: float, max_den: int = 24, tol: float = 1e-6) -> str:
    """Express a float as integer/fraction when possible; widen denominator and reconstruct via decimal string when needed"""

    def _fmt(frac: Fraction) -> str:
        if frac.denominator == 1:
            return str(frac.numerator)
        sign = "-" if frac.numerator * frac.denominator < 0 else ""
        num = abs(frac.numerator)
        den = abs(frac.denominator)
        return f"{sign}{num}/{den}"

    # Try smaller denominators first
    frac = Fraction(value).limit_denominator(max_den)
    if abs(float(frac) - value) <= tol:
        return _fmt(frac)

    # Then widen the denominator range, recovering exact decimal via string precision
    widened_den = max_den * 10
    dec_txt = format(value, ".6f")
    frac = Fraction(dec_txt).limit_denominator(widened_den)
    return _fmt(frac)


def _format_sqrt_from_square(square: float, max_den: int = 48, tol: float = 1e-6) -> str:
    """Given a squared value, output a √n/rational form, preferring integer, fraction, or surd expressions"""
    if square <= tol:
        return "0"
    rounded = round(square)
    if abs(square - rounded) <= tol:
        root = int(math.isqrt(max(0, rounded)))
        if root * root == rounded:
            return str(root)
        return f"√{rounded}"

    frac = Fraction(square).limit_denominator(max_den)
    if abs(float(frac) - square) <= tol:
        num = abs(frac.numerator)
        den = frac.denominator
        num_root = int(math.isqrt(num))
        den_root = int(math.isqrt(den))
        sign = "-" if frac.numerator < 0 else ""
        if num_root * num_root == num and den_root * den_root == den:
            if den_root == 1:
                return f"{sign}{num_root}"
            return f"{sign}{num_root}/{den_root}"
        if den == 1:
            return f"{sign}√{num}"
        return f"{sign}√({num}/{den})"

    approx = _format_fraction(square, max_den=max_den, tol=tol)
    return f"√({approx})"


def _format_coord_xy(x: float, y: float) -> Tuple[str, str]:
    return _format_fraction(x), _format_fraction(y)


# ======================================================================


def _coerce_to_schema(obj: dict) -> dict:
    """
    Try to coerce common gateway response formats (wrapped/synonym keys) into the target schema:
    {problem, givens[], goal, answer, proof_outline[], used_objects[]}
    """
    if not isinstance(obj, dict):
        return obj

    # 1) Unwrap common outer fields first
    for wrapper in ("data", "output", "result", "response"):
        if isinstance(obj.get(wrapper), dict):
            obj = obj[wrapper]

    # 2) Some gateways stuff the JSON into a message.content string
    if isinstance(obj.get("message"), dict) and isinstance(obj["message"].get("content"), str):
        txt = obj["message"]["content"]
        try:
            maybe = json.loads(txt)
            if isinstance(maybe, dict):
                obj = maybe
        except Exception:
            pass

    # 3) Synonym key mapping
    synonym_map = {
        "question": "problem",
        "statement": "problem",
        "prompt": "problem",
        "given": "givens",
        "givens_list": "givens",
        "goal_statement": "goal",
        "solution": "answer",
        "answer_text": "answer",
        "proof": "proof_outline",
        "outline": "proof_outline",
        "objects": "used_objects",
        "entities": "used_objects",
    }
    for src, dst in synonym_map.items():
        if src in obj and dst not in obj:
            obj[dst] = obj[src]

    # 4) Normalize types to list
    if isinstance(obj.get("givens"), str):
        obj["givens"] = [obj["givens"]]
    if isinstance(obj.get("proof_outline"), str):
        obj["proof_outline"] = [obj["proof_outline"]]
    if isinstance(obj.get("used_objects"), str):
        # When string, try splitting by space/comma
        s = obj["used_objects"].replace(",", " ")
        obj["used_objects"] = [x for x in s.split() if x]

    return obj

# Simple validator: ensure all fields are present + used_objects ⊆ allowed_ids
import re
from typing import List, Dict, Any


def _validate_problem_json(obj: dict, allowed_ids: list, VeriGraph, *, strict: bool = True):
    meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}
    problem_type = (meta or {}).get("problem_type") or obj.get("problem_type")
    required = ["problem", "givens", "goal", "answer", "proof_outline", "used_objects"]
    for k in required:
        if k not in obj:
            raise ValueError(f"Missing field: {k}")
    if not isinstance(obj["problem"], str) or not obj["problem"].strip():
        raise ValueError("problem must be a non-empty string")
    # Prevent circular reasoning: check that goal does not appear directly in givens
    goal = obj["goal"].strip()
    for given in obj["givens"]:
        if goal in given or given in goal:
            raise ValueError(f"Circular reasoning: goal '{goal}' cannot serve as given '{given}'")
    if not isinstance(obj["givens"], list) or not all(isinstance(x, str) for x in obj["givens"]):
        raise ValueError("givens must be a list of strings")
    if not isinstance(obj["goal"], str) or not obj["goal"].strip():
        raise ValueError("goal must be a non-empty string")
    if not isinstance(obj["answer"], str) or not obj["answer"].strip():
        raise ValueError("answer must be a non-empty string")
    if not isinstance(obj["proof_outline"], list) or not all(isinstance(x, str) for x in obj["proof_outline"]):
        raise ValueError("proof_outline must be a list of strings")
    if not isinstance(obj["used_objects"], list) or not all(isinstance(x, str) for x in obj["used_objects"]):
        raise ValueError("used_objects must be a list of strings")

    # Auto-append angle entities present in text to avoid omissions
    if obj["used_objects"] is not None:
        dedup = []
        seen = set()
        for item in obj["used_objects"]:
            if item not in seen:
                dedup.append(item)
                seen.add(item)
        obj["used_objects"] = dedup
    angle_tokens = set()
    texts = [obj.get("problem", ""), obj.get("goal", ""), obj.get("answer", "")]
    texts.extend(obj.get("givens") or [])
    texts.extend(obj.get("proof_outline") or [])
    angle_pattern = re.compile(r"∠([A-Za-z]{3})")
    ang_name_pattern = re.compile(r"ANG_([A-Za-z]{3})", re.IGNORECASE)
    for txt in texts:
        if not isinstance(txt, str):
            continue
        for match in angle_pattern.findall(txt):
            angle_tokens.add("∠" + match.upper())
        for match in ang_name_pattern.findall(txt):
            angle_tokens.add("ANG_" + match.upper())
    used_set = set(obj["used_objects"])
    for ang in sorted(angle_tokens):
        if ang in allowed_ids and ang not in used_set:
            obj["used_objects"].append(ang)
            used_set.add(ang)

    extra = [x for x in obj["used_objects"] if x not in set(allowed_ids)]
    if extra:
        raise ValueError(f"used_objects contains unauthorized entities: {extra}")
    return obj  # valid

# —— Extract allowed_ids and clean_constraints from your GeometryGraph ——
def _collect_alias_map(VeriGraph) -> Dict[str, str]:
    alias_map: Dict[str, str] = {}
    for eid, ent in VeriGraph.entities.items():
        if ent.type == "Segment":
            labels = set()
            attrs_name = ent.attrs.data.get("name")
            if isinstance(attrs_name, str) and attrs_name:
                labels.add(attrs_name)
            aliases = ent.attrs.data.get("aliases") or ent.attrs.data.get("alias")
            if isinstance(aliases, str) and aliases:
                labels.add(aliases)
            elif isinstance(aliases, (list, tuple, set)):
                for alias in aliases:
                    if isinstance(alias, str) and alias:
                        labels.add(alias)
            for token in list(labels):
                labels.add(token.upper())
                labels.add(token.lower())
            for token in labels:
                if isinstance(token, str) and token:
                    alias_map.setdefault(token, eid)
            if eid.upper().startswith("SEG_"):
                name = eid.upper()
                body = name.split("_", 1)[1] if "_" in name else name[3:]
                if len(body) == 2 and body.isalpha():
                    a, b = body[0], body[1]
                    rev = f"SEG_{b}{a}"
                    if rev not in alias_map:
                        alias_map[rev] = eid
                    tokens = {body, body[::-1], body.lower(), body[::-1].lower()}
                    for token in tokens:
                        alias_map.setdefault(token, eid)
        elif ent.type == "Triangle" and eid.upper().startswith("TRI_") and len(eid) >= 7:
            canonical = eid
            body = eid.split("_", 1)[1] if "_" in eid else eid[3:]
            body = body.upper()
            if len(body) >= 3:
                letters = list(body[:3])
                rotations = {
                    f"TRI_{letters[i%3]}{letters[(i+1)%3]}{letters[(i+2)%3]}"
                    for i in range(3)
                }
                for token in rotations:
                    alias_map.setdefault(token, canonical)
        elif ent.type == "Angle":
            name = ent.attrs.data.get("name")
            candidates = set()
            for tok in (name, eid):
                if isinstance(tok, str) and tok:
                    candidates.add(tok)
            aliases = ent.attrs.data.get("aliases") or ent.attrs.data.get("alias")
            if isinstance(aliases, str) and aliases:
                candidates.add(aliases)
            elif isinstance(aliases, (list, tuple, set)):
                for alias in aliases:
                    if isinstance(alias, str) and alias:
                        candidates.add(alias)
            if isinstance(name, str) and len(name) >= 3:
                body = name[-3:]
                candidates.update({f"??{body}", f"?{body}", f"ANG_{body}", body})
            if eid.upper().startswith("ANG_") and len(eid) >= 7:
                body = eid.split("_", 1)[1]
                if len(body) >= 3:
                    candidates.update({f"??{body[:3]}", f"?{body[:3]}", body[:3]})
            for tok in candidates:
                if not isinstance(tok, str) or not tok:
                    continue
                alias_map.setdefault(tok, eid)
                alias_map.setdefault(tok.upper(), eid)
                alias_map.setdefault(tok.lower(), eid)
        elif ent.type == "Circle":
            labels = set()
            attrs_name = ent.attrs.data.get("name")
            if isinstance(attrs_name, str) and attrs_name:
                labels.add(attrs_name)
            labels.add(eid)
            eid_upper = eid.upper()
            if eid_upper.startswith("CIR_"):
                labels.add(eid.split("_", 1)[1])
            for token in list(labels):
                labels.add(token.upper())
                labels.add(token.lower())
            for token in labels:
                if isinstance(token, str) and token:
                    alias_map.setdefault(token, eid)
        elif ent.type == "Function":
            labels = {eid}
            eid_upper = eid.upper()
            if eid_upper.startswith("FUNC_"):
                base = eid.split("_", 1)[1]
                if base:
                    labels.add(base)
            for token in list(labels):
                if isinstance(token, str) and token:
                    labels.add(token.upper())
                    labels.add(token.lower())
            for token in labels:
                if isinstance(token, str) and token:
                    alias_map.setdefault(token, eid)
    return alias_map


def _extract_allowed_and_constraints(VeriGraph, sub_ids=None):
    all_ids = list(VeriGraph.entities.keys())
    allowed = set(sub_ids) if sub_ids else set(all_ids)
    alias_map = _collect_alias_map(VeriGraph)
    for alias, target in alias_map.items():
        if target in allowed:
            allowed.add(alias)
    clean_constraints = []
    for c in VeriGraph.constraints.values():
        sc = list(c.scope)
        if all(s in allowed for s in sc):
            clean_constraints.append({
                "id": c.id, "type": c.type, "scope": sc,
                "params": dict(c.params or {}), "source": c.source
            })
    # Stable sort by source priority: problem/given first
    def key_fn(d):
        pri = 0 if d["source"] in ("problem", "given") else 1
        return (pri, d["type"], d["id"])
    clean_constraints.sort(key=key_fn)
    VeriGraph.meta.setdefault("alias_map", {}).update(alias_map)
    return list(allowed), clean_constraints, alias_map

def _canonical_segment_id(u: str, v: str) -> str:
    """Return canonical segment ID: SEG_uv (alphabetical), and require distinct endpoints."""
    if u == v:
        raise ValueError("Segment endpoints must be distinct")
    return f"SEG_{u}{v}" if u < v else f"SEG_{v}{u}"


def _describe_relation_snippet(ctype: str, scope: List[str]) -> str:
    """Convert a key constraint into a short English description, providing high-value cues for the prompt."""
    label_map = {
        "Perpendicular": "perpendicular",
        "Parallel": "parallel",
        "EqualLength": "equal-length",
        "EqualAngle": "equal-angle",
        "AngleBisector": "angle bisector",
        "Midpoint": "midpoint",
        "PowerEquality": "power-of-a-point",
        "SimilarTriangles": "similar triangles",
        "CongruentTriangles": "congruent triangles",
        "Concyclic": "concyclic",
        "VectorDotZero": "perpendicular vectors",
    }
    if not scope:
        return ""
    if ctype == "Perpendicular" and len(scope) >= 2:
        return f"{scope[0]} ⟂ {scope[1]}"
    if ctype == "Parallel" and len(scope) >= 2:
        return f"{scope[0]} ∥ {scope[1]}"
    if ctype == "EqualLength" and len(scope) >= 2:
        return f"{scope[0]} = {scope[1]}"
    if ctype == "EqualAngle" and len(scope) >= 2:
        return f"∠{scope[0]} = ∠{scope[1]}"
    if ctype == "AngleBisector" and len(scope) >= 2:
        return f"{scope[0]} is the angle bisector of ∠{scope[1]}"
    if ctype == "Midpoint" and len(scope) >= 3:
        return f"{scope[0]} is the midpoint of {scope[1]}{scope[2]}"
    if ctype == "Concyclic" and len(scope) >= 3:
        return f"{', '.join(scope)} are concyclic"
    label = label_map.get(ctype, ctype)
    return f"{label}: {', '.join(scope)}"


def _summarize_clean_constraints(clean_constraints: List[Dict[str, Any]], *, max_key_relations: int = 48) -> Dict[str, Any]:
    """Distill clean_constraints to present high-value structure and candidate goals to the LLM."""
    circle_points: Dict[str, set] = defaultdict(set)
    segment_points: Dict[str, set] = defaultdict(set)
    line_points: Dict[str, set] = defaultdict(set)
    counts: Counter = Counter()
    key_relations: List[str] = []
    type_samples: Dict[str, List[str]] = {}
    key_types = {
        "Perpendicular", "Parallel", "EqualLength", "EqualAngle", "AngleBisector", "Midpoint",
        "PowerEquality", "SimilarTriangles", "CongruentTriangles", "Concyclic", "VectorDotZero",
    }

    for cons in clean_constraints:
        ctype = (cons.get("type") or "").strip()
        scope = list(cons.get("scope") or [])
        counts[ctype] += 1
        if ctype in ("OnCircle", "PointOnCircle") and len(scope) >= 2:
            point, circle = scope[0], scope[1]
            circle_points[circle].add(point)
        elif ctype == "PointOnSegment" and len(scope) >= 2:
            point, seg = scope[0], scope[1]
            segment_points[seg].add(point)
        elif ctype == "PointOnLine" and len(scope) >= 2:
            point, line = scope[0], scope[1]
            line_points[line].add(point)

        if ctype in key_types:
            if ctype not in type_samples:
                type_samples[ctype] = scope
            snippet = _describe_relation_snippet(ctype, scope)
            if snippet:
                key_relations.append(snippet)

    if len(key_relations) > max_key_relations:
        key_relations = key_relations[:max_key_relations]

    circle_groups = [{"id": cid, "points": sorted(list(points))} for cid, points in circle_points.items()]
    segment_groups = [{"id": sid, "points": sorted(list(points))} for sid, points in segment_points.items()]
    line_groups = [{"id": lid, "points": sorted(list(points))} for lid, points in line_points.items()]
    circle_groups.sort(key=lambda item: item["id"])
    segment_groups.sort(key=lambda item: item["id"])
    line_groups.sort(key=lambda item: item["id"])

    relation_counts = [{"type": typ, "count": cnt} for typ, cnt in counts.most_common()]

    goal_hints: List[str] = []
    if "PowerEquality" in type_samples:
        goal_hints.append(f"Use the power-of-a-point theorem ({', '.join(type_samples['PowerEquality'])}) together with tangent/secant constructions to build length or angle ratios.")
    if "AngleBisector" in type_samples or "EqualAngle" in type_samples:
        goal_hints.append("Combining angle bisection or equal-angle relations, try to prove the incenter/circumcenter or triangle similarity.")
    if "Concyclic" in type_samples and "Perpendicular" in type_samples:
        goal_hints.append("Concyclic + perpendicular-chord can set up inscribed-angle / tangent-chord theorem chains, from which goals can be derived.")
    if "SimilarTriangles" in type_samples:
        goal_hints.append(f"Similar triangles available ({', '.join(type_samples['SimilarTriangles'])}); extend to proportional or equal-angle conclusions.")
    if "Midpoint" in type_samples and "Parallel" in type_samples:
        goal_hints.append("Combining midpoints with parallel lines can build parallelogram/trapezoid properties, suitable for advancing multi-step reasoning.")

    return {
        "circle_groups": circle_groups,
        "segment_groups": segment_groups,
        "line_groups": line_groups,
        "key_relations": key_relations,
        "relation_counts": relation_counts,
        "goal_hints": goal_hints,
    }


def _format_constraint_summary(summary: Dict[str, Any]) -> str:

    """Convert the summary structure to compact text for direct concatenation into prompts."""

    if not summary:

        return ""

    lines: List[str] = []



    def _append_groups(title: str, groups: List[Dict[str, Any]], limit: int = 8):

        if not groups:

            return

        lines.append(f"[{title}]")

        for idx, item in enumerate(groups):

            if idx >= limit:

                lines.append("…(remaining of same kind omitted)")

                break

            pts = ", ".join(item.get("points") or [])

            lines.append(f"- {item.get('id')}: {pts}")



    _append_groups("Points on circle", summary.get("circle_groups") or [])

    _append_groups("Points on segment", summary.get("segment_groups") or [])

    _append_groups("Points on line", summary.get("line_groups") or [], limit=6)



    key_relations = summary.get("key_relations") or []

    if key_relations:

        lines.append("[Key Relations (excerpt)]")

        for idx, rel in enumerate(key_relations):

            if idx >= 40:

                lines.append("…(more key relations omitted)")

                break

            lines.append(f"- {rel}")



    relation_counts = summary.get("relation_counts") or []

    if relation_counts:

        lines.append("[Constraint Count]")

        for item in relation_counts[:12]:

            lines.append(f"- {item['type']}: {item['count']}")



    goal_hints = summary.get("goal_hints") or []

    if goal_hints:

        lines.append("[Candidate Goal Hints]")

        for hint in goal_hints[:6]:

            lines.append(f"- {hint}")



    return "\n".join(lines).strip()





def _filter_author_constraints(clean_constraints: List[Dict[str, Any]], segment_catalog: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Filter noise constraints for LLM input: if PointOnSegment points to a segment endpoint, treat it as pure naming information."""
    if not clean_constraints:
        return []
    seg_map: Dict[str, set] = {}
    for item in segment_catalog or []:
        sid = item.get("id")
        points = item.get("points")
        if sid and isinstance(points, list):
            seg_map[sid] = {str(p) for p in points}

    filtered: List[Dict[str, Any]] = []
    for cons in clean_constraints:
        ctype = cons.get("type")
        scope = list(cons.get("scope") or [])
        if ctype == "PointOnSegment" and len(scope) >= 2:
            point, seg_id = scope[0], scope[1]
            pts = seg_map.get(seg_id)
            if pts and point in pts:
                continue
            if _segment_name_matches_point(seg_id, point):
                continue
        filtered.append(cons)
    return filtered


def _segment_name_matches_point(seg_id: str, point_id: str) -> bool:
    """If the segment name directly contains the point name, treat it as endpoint information."""
    if not isinstance(seg_id, str) or not isinstance(point_id, str):
        return False
    seg_id = seg_id.upper()
    point_token = point_id.upper()
    if not seg_id.startswith("SEG_"):
        return False
    body = seg_id.split("_", 1)[1]
    if not body:
        return False
    return body.startswith(point_token) or body.endswith(point_token)







# —— Build the LLM's system + user payload ——
# (removed legacy _SYSTEM_PROMPT — unused since unified author workflow)
# Backward-compatible alias for legacy name






_AUTHOR_COMBINED_PROMPT = load_prompt("author_combined")
_AUTHOR_COMBINED_PROMPT_FROM_SCRATCH = load_prompt("author_from_scratch")
_SYSTEM_PROMPT_SOLVER = load_prompt("solver_system")
# ===== Strict VeriScript JSON Schema (solver side) =====
SOLVER_CONSTRAINT_TYPES = [
    "Perpendicular", "Parallel",
    "EqualLength", "EqualAngle",
    "SimilarTriangles", "CongruentTriangles",
    "PointOnLine", "PointOnSegment", "OnCircle",
    "AngleBisector", "Midpoint", "Collinear", "Cyclic",
    "VectorSum", "VectorDiff", "VectorDotZero", "VectorLinear",
    "ExprConstraint",
]

# ==== VeriScript definitions and controlled actions ====
VeriScript_ALLOWED_EDGE_TYPES = {
    "Perpendicular",
    "Parallel",
    "EqualLength",
    "EqualAngle",
    "Collinear",
    "OnCircle",
    "AngleBisector",
    "Midpoint",
    "SimilarTriangles",
    "CongruentTriangles",
    "VectorSum",
    "VectorDiff",
    "VectorDotZero",
    "VectorLinear",
    "ExprConstraint",
}
VeriScript_ALLOWED_OPS = {"AddEdge", "AddPoint", "AddAuxLine", "AddFunction", "VerifyPoint", "VerifyFunction"}
VeriScript_POINT_TYPES = {"Free", "OnSegment", "OnCircle", "PointOnCircle"}
VeriScript_AUX_LINE_TYPES = {"Segment"}
VeriScript_VERIFY_POINT_TYPES = {"Cartesian"}
VeriScript_VERIFY_FUNCTION_TYPES = {"PointsOnFunction", "DerivativeAt", "IntegralOn"}

VeriScript_JSON_SCHEMA = {
    "name": "VeriScript_schema",
    "schema": {
        "type": "object",
        "properties": {
            "actions": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "op":   {"type": "string", "enum": sorted(list(VeriScript_ALLOWED_OPS))},
                        "type": {"type": "string"},
                        "args": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1
                        }
                    },
                    "required": ["op", "type", "args"],
                    "additionalProperties": False
                }
            },
            "final_answer": {"type": "string", "minLength": 1},
            "proof": {"type": "string", "minLength": 1},
            "reflection_modified": {"type": "boolean"}
        },
        "required": ["actions", "final_answer", "proof"],
        "additionalProperties": False
    },
    "strict": True
}
# ================================================






# ==== VeriScript definitions and controlled actions ====
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

@dataclass
class VeriScriptAction:
    op: str             # AddEdge / AddPoint / AddAuxLine / AddFunction / VerifyPoint / VerifyFunction
    type: str           # Constraint, point-construction method, or auxiliary-line type
    args: List[str]     # Parameters: entity IDs / supplementary parameters (e.g. ratios); must follow allowed rules

def _parse_VeriScript_json(obj: Dict[str, Any]) -> Tuple[List[VeriScriptAction], str]:
    """Parse the action sequence and final conclusion from the JSON returned by the LLM"""
    if not isinstance(obj, dict) or "actions" not in obj:
        raise ValueError("VeriScript JSON must contain an 'actions' field.")
    acts = []
    for i, a in enumerate(obj["actions"]):
        op  = a.get("op", "")
        typ = a.get("type", "")
        raw_args = a.get("args", [])
        if op not in VeriScript_ALLOWED_OPS:
            raise ValueError(f"Step {i+1} op invalid: {op}")
        if not isinstance(raw_args, list):
            raise ValueError(f"Step {i+1} args format error; expected array.")
        # Normalize numbers to strings for downstream processing
        args: List[str] = []
        for tok in raw_args:
            if isinstance(tok, str):
                args.append(tok.strip())
            elif isinstance(tok, (int, float)):
                args.append(str(tok))
            else:
                raise ValueError(f"Step {i+1} args only support strings or numbers; received {type(tok).__name__}")

        if op == "AddEdge":
            if typ not in VeriScript_ALLOWED_EDGE_TYPES:
                raise ValueError(f"Step {i+1} type invalid: {typ}")
            if not args:
                raise ValueError(f"Step {i+1} AddEdge requires at least 1 argument")
            if typ == "VectorLinear":
                if len(args) < 5 or (len(args) - 2) % 3 != 0:
                    raise ValueError(f"Step {i+1} VectorLinear must be of form [U,V,k1,P1,Q1,...] with at least one linear-combination term")
                for k_idx in range(2, len(args), 3):
                    try:
                        _author_parse_float(args[k_idx], "coeff")
                    except Exception as exc:
                        raise ValueError(f"Step {i+1} VectorLinear coefficient must be numeric: {args[k_idx]}") from exc
        elif op == "AddPoint":
            if typ not in VeriScript_POINT_TYPES:
                raise ValueError(f"Step {i+1} AddPoint type invalid: {typ}")
            if not args:
                raise ValueError(f"Step {i+1} AddPoint must provide at least the new point ID")
            if typ == "OnSegment" and len(args) < 2:
                raise ValueError(f"Step {i+1} AddPoint-OnSegment must provide segment ID")
            if typ in ("OnCircle", "PointOnCircle") and len(args) < 2:
                raise ValueError(f"Step {i+1} AddPoint-OnCircle must provide circle ID")
        elif op == "AddAuxLine":
            if typ not in VeriScript_AUX_LINE_TYPES:
                raise ValueError(f"Step {i+1} AddAuxLine type invalid: {typ}")
            if len(args) < 2:
                raise ValueError(f"Step {i+1} AddAuxLine requires at least two points")
        elif op == "AddFunction":
            if len(args) < 2:
                raise ValueError(f"Step {i+1} AddFunction requires function ID and expression")
        elif op == "VerifyPoint":
            if typ not in VeriScript_VERIFY_POINT_TYPES:
                raise ValueError(f"Step {i+1} VerifyPoint type invalid: {typ}")
            if len(args) < 3:
                raise ValueError(f"Step {i+1} VerifyPoint must provide point ID and coordinates (x,y)")
            try:
                _author_parse_float(args[1], "x")
                _author_parse_float(args[2], "y")
            except Exception as exc:
                raise ValueError(f"Step {i+1} VerifyPoint coordinates must be numeric: {exc}") from exc
        elif op == "VerifyFunction":
            if typ not in VeriScript_VERIFY_FUNCTION_TYPES:
                raise ValueError(f"Step {i+1} VerifyFunction type invalid: {typ}")
            if typ == "PointsOnFunction" and len(args) < 2:
                raise ValueError(f"Step {i+1} VerifyFunction-PointsOnFunction must provide function and at least one point")
            if typ == "DerivativeAt" and len(args) < 3:
                raise ValueError(f"Step {i+1} VerifyFunction-DerivativeAt must provide function, x, and derivative value")
            if typ == "IntegralOn" and len(args) < 4:
                raise ValueError(f"Step {i+1} VerifyFunction-IntegralOn must provide function, interval, and integral value")
        else:
            raise ValueError(f"Step {i+1} op not supported: {op}")

        acts.append(VeriScriptAction(op=op, type=typ, args=[arg for arg in args if arg]))
    final_answer = obj.get("final_answer", "")
    if not isinstance(final_answer, str) or not final_answer.strip():
        raise ValueError("final_answer must be a non-empty string.")
    return acts, final_answer.strip()

def _validate_VeriScript_actions(actions: List[VeriScriptAction], allowed_ids: List[str]):
    """Controlled decoding: ensure actions only use allowed_ids, with newly created entities added to the whitelist step by step per rules."""
    allowed = set(allowed_ids)

    def _require(token: str, step: int):
        if token not in allowed:
            raise ValueError(f"Step {step} argument {token} is not in allowed_ids.")

    for i, a in enumerate(actions, start=1):
        if a.op == "AddEdge":
            if a.type not in VeriScript_ALLOWED_EDGE_TYPES:
                raise ValueError(f"Step {i} AddEdge type invalid: {a.type}")
            if a.type == "ExprConstraint":
                if len(a.args) < 3:
                    raise ValueError(f"Step {i} ExprConstraint requires at least lhs, relation, rhs")
            elif a.type == "VectorLinear":
                if len(a.args) < 5 or (len(a.args) - 2) % 3 != 0:
                    raise ValueError(f"Step {i} VectorLinear must be of form [U,V,k1,P1,Q1,...] with at least one linear-combination term")
                _require(a.args[0], i)
                _require(a.args[1], i)
                for idx in range(2, len(a.args), 3):
                    try:
                        _author_parse_float(a.args[idx], "coeff")
                    except Exception:
                        raise ValueError(f"Step {i} VectorLinear coefficient must be numeric: {a.args[idx]}")
                    _require(a.args[idx + 1], i)
                    _require(a.args[idx + 2], i)
            else:
                for tok in a.args:
                    _require(tok, i)
        elif a.op == "AddPoint":
            if not a.args:
                raise ValueError(f"Step {i} AddPoint missing new point ID")
            pid = a.args[0]
            if pid in allowed:
                raise ValueError(f"Step {i} AddPoint: {pid} already exists; cannot recreate.")
            if a.type == "OnSegment":
                if len(a.args) < 2:
                    raise ValueError(f"Step {i} AddPoint-OnSegment must provide segment ID")
                seg_id = a.args[1]
                _require(seg_id, i)
            allowed.add(pid)
        elif a.op == "AddAuxLine":
            if len(a.args) < 2:
                raise ValueError(f"Step {i} AddAuxLine must provide two points")
            p, q = a.args[:2]
            _require(p, i)
            _require(q, i)
            try:
                seg_id = _canonical_segment_id(p, q)
            except ValueError as exc:
                raise ValueError(f"Step {i} AddAuxLine: {exc}") from exc
            allowed.add(seg_id)
        elif a.op == "AddFunction":
            if len(a.args) < 2:
                raise ValueError(f"Step {i} AddFunction must provide function ID and expression")
            fid = a.args[0]
            if fid in allowed:
                raise ValueError(f"Step {i} AddFunction: {fid} already exists; cannot recreate.")
            allowed.add(fid)
        elif a.op == "VerifyPoint":
            if len(a.args) < 3:
                raise ValueError(f"Step {i} VerifyPoint must provide point ID and coordinates (x,y)")
            pid = a.args[0]
            _require(pid, i)
            try:
                _author_parse_float(a.args[1], "x")
                _author_parse_float(a.args[2], "y")
            except Exception as exc:
                raise ValueError(f"Step {i} VerifyPoint coordinates must be numeric: {exc}") from exc
        elif a.op == "VerifyFunction":
            if len(a.args) < 2:
                raise ValueError(f"Step {i} VerifyFunction must provide enough arguments")
            solve_type = a.type
            if solve_type not in VeriScript_VERIFY_FUNCTION_TYPES:
                raise ValueError(f"Step {i} VerifyFunction type invalid: {solve_type}")
            if solve_type == "PointsOnFunction":
                if len(a.args) < 2:
                    raise ValueError(f"Step {i} VerifyFunction-PointsOnFunction must provide function and at least one point")
                tokens = a.args[1:]
                if len(tokens) >= 2:
                    try:
                        _author_parse_float(tokens[-1], "tol")
                        if len(tokens) % 2 == 1 or tokens[-1] not in allowed:
                            tokens = tokens[:-1]
                    except Exception:
                        pass
                idx = 0
                while idx < len(tokens):
                    tok = tokens[idx]
                    if tok in allowed:
                        _require(tok, i)
                        idx += 1
                        continue
                    if idx + 1 >= len(tokens):
                        raise ValueError(f"Step {i} VerifyFunction-PointsOnFunction must provide coordinate pairs")
                    _author_parse_float(tokens[idx], "x")
                    _author_parse_float(tokens[idx + 1], "y")
                    idx += 2
            elif solve_type == "DerivativeAt":
                if len(a.args) < 3:
                    raise ValueError(f"Step {i} VerifyFunction-DerivativeAt must provide function, x, and derivative value")
                x_tok = a.args[1]
                if x_tok in allowed:
                    _require(x_tok, i)
                else:
                    _author_parse_float(x_tok, "x")
                _author_parse_float(a.args[2], "derivative")
                if len(a.args) >= 5:
                    _parse_derivative_order(a.args[3])
                    _author_parse_float(a.args[4], "tol")
                elif len(a.args) == 4:
                    tok = str(a.args[3]).strip().lower()
                    if tok.startswith("n="):
                        _parse_derivative_order(a.args[3])
                    else:
                        _author_parse_float(a.args[3], "tol")
            elif solve_type == "IntegralOn":
                if len(a.args) < 4:
                    raise ValueError(f"Step {i} VerifyFunction-IntegralOn must provide function, interval, and integral value")
                a_tok = a.args[1]
                b_tok = a.args[2]
                def _is_inf(tok: str) -> bool:
                    t = str(tok).strip().lower()
                    return t in _INF_TOKENS_POS or t in _INF_TOKENS_NEG
                if a_tok in allowed:
                    _require(a_tok, i)
                elif not _is_inf(a_tok):
                    _author_parse_float(a_tok, "a")
                if b_tok in allowed:
                    _require(b_tok, i)
                elif not _is_inf(b_tok):
                    _author_parse_float(b_tok, "b")
                _author_parse_float(a.args[3], "integral")
                rest = a.args[4:]
                if rest:
                    head = str(rest[0]).strip().lower()
                    if head in ("breaks", "breakpoints"):
                        rest = rest[1:]
                    else:
                        try:
                            _author_parse_float(rest[0], "tol")
                            rest = rest[1:]
                        except Exception:
                            pass
                        if rest and str(rest[0]).strip().lower() in ("breaks", "breakpoints"):
                            rest = rest[1:]
                    for tok in rest:
                        if tok in allowed:
                            _require(tok, i)
                        elif not _is_inf(tok):
                            _author_parse_float(tok, "break")
        else:
            raise ValueError(f"Step {i} op not supported: {a.op}")
    return True


# ==== Author Review Prompt & Schema ====
_AUTHOR_REVIEW_PROMPT = load_prompt("author_review")
_AUTHOR_SELF_REVIEW_GUIDE = (
    "Author self-check instructions: you have just completed the graph edit (VeriScript) and problem generation; now the rendered figure and JSON are returned to you as-is. "
    "Within the same dialogue turn, please verify on your own whether the figure and problem statement contain any contradictions, omissions, or unreasonable aspects. "
    "If there are issues, directly modify the actions and problem statement and return a new complete JSON; if you confirm everything is correct, return the complete JSON as-is and note \"self-check passed\" in the comment. "
    "Do not role-play as an independent review agent, and do not only give suggestions or critique. actions must be a complete covering sequence from the input graph to the final graph, not an incremental patch."
)


def _build_author_self_review_blocks(
    VeriGraph,
    combo_obj: Dict[str, Any],
    round_idx: int,
    total_rounds: int,
    attach_image: bool = True,
    feedback: Optional[str] = None,
):
    guide = CONFIG.author_reflection_prompt or _AUTHOR_SELF_REVIEW_GUIDE
    prefix = f"Author self-check round {round_idx}/{total_rounds}: please directly output the revised complete JSON (actions/problem/comment must all be provided)."
    blocks = [{"type": "text", "text": prefix + "\n" + guide}]
    if feedback:
        blocks.append({"type": "text", "text": "Issues and log summary to address in this round:\n" + str(feedback)})
    try:
        prob_txt = json.dumps(combo_obj.get("problem", {}), ensure_ascii=False, indent=2)
        blocks.append({"type": "text", "text": "Current problem:\n" + prob_txt})
    except Exception:
        pass
    try:
        actions_txt = json.dumps(combo_obj.get("actions", []), ensure_ascii=False, indent=2)
        blocks.append({"type": "text", "text": "Current actions (complete covering sequence):\n" + actions_txt})
    except Exception:
        pass
    comment = combo_obj.get("comment")
    if comment:
        blocks.append({"type": "text", "text": "Previous round notes: " + str(comment)})
    logs = combo_obj.get("logs") or []
    if logs:
        try:
            logs_txt = json.dumps(logs[:20], ensure_ascii=False, indent=2)
            blocks.append({"type": "text", "text": "VeriScript execution logs (truncated):\n" + logs_txt})
        except Exception:
            pass
    try:
        nl_text = NaturalLanguageDescriber.describe(VeriGraph, verify=None, only_true=False)
        blocks.append({"type": "text", "text": "Current graph summary:\n" + _redact_nl_coordinates(nl_text)})
    except Exception:
        pass
    if attach_image:
        block = _build_prompt_image_block(VeriGraph, f"author_self_review_{round_idx}")
        if block:
            blocks.append(block)
    return blocks

AUTHOR_REVIEW_JSON_SCHEMA = {
    "name": "author_review_schema",
    "schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "needs_fix"]},
            "issues": {
                "type": "array",
                "items": {"type": "string"},
                "default": []
            },
            "fix_suggestions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "focus": {
                            "type": "string",
                            "enum": ["VeriScript", "problem", "givens", "goal", "proof_outline", "graph", "other"]
                        },
                        "detail": {"type": "string"},
                        "severity": {
                            "type": "string",
                            "enum": ["blocking", "major", "minor"],
                            "default": "blocking"
                        },
                        "recommendation": {"type": "string", "default": ""}
                    },
                    "required": ["focus", "detail", "severity", "recommendation"],
                    "additionalProperties": False
                },
                "default": []
            },
            "repair_plan": {"type": "string", "default": ""},
            "needs_regeneration": {"type": "boolean", "default": False}
        },
        "required": ["status", "issues", "fix_suggestions", "repair_plan", "needs_regeneration"],
        "additionalProperties": False
    },
    "strict": True
}


# ==== Pre-authoring edits: extended VeriScript (applied only to the authoring LLM) ====
AUTHOR_VeriScript_ALLOWED_OPS = {
    "AddPoint",
    "MovePoint",
    "AddAuxLine",
    "AddEdge",
    "AddCircle",
    "AddAngle",
    "AddTriangle",
    "AddQuadrilateral",
    "AddFunction",
}
AUTHOR_VeriScript_POINT_TYPES = {"Free", "OnSegment", "OnCircle", "PointOnCircle", "Cartesian"}
AUTHOR_VeriScript_AUX_LINE_TYPES = {"Segment"}
AUTHOR_VeriScript_JSON_SCHEMA = {
    "name": "author_VeriScript_schema",
    "schema": {
        "type": "object",
        "properties": {
            "actions": {
                "type": "array",
                "minItems": 0,
                "items": {
                    "type": "object",
                    "properties": {
                        "op": {"type": "string", "enum": sorted(list(AUTHOR_VeriScript_ALLOWED_OPS))},
                        "type": {"type": "string"},
                        "args": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1
                        }
                    },
                    "required": ["op", "type", "args"],
                    "additionalProperties": False
                }
            },
            "comment": {"type": "string"}
        },
        "required": ["actions", "comment"],
        "additionalProperties": False
    },
    "strict": True
}

AUTHOR_COMBINED_JSON_SCHEMA = {
    "name": "author_combined_schema",
    "schema": {
        "type": "object",
        "properties": {
            "actions": AUTHOR_VeriScript_JSON_SCHEMA["schema"]["properties"]["actions"],
            "problem": {
                "type": "object",
                "properties": {
                    "problem": {"type": "string"},
                    "givens": {"type": "array", "items": {"type": "string"}},
                    "goal": {"type": "string"},
                    "answer": {"type": "string"},
                    "proof_outline": {"type": "array", "items": {"type": "string"}},
                    "used_objects": {"type": "array", "items": {"type": "string"}},
                    "meta": {
                        "type": "object",
                        "properties": {
                            "problem_type": {"type": "string"},
                            "reflection_modified": {"type": "boolean"}
                        },
                        "required": ["problem_type"],
                        "additionalProperties": False
                    }
                },
                "required": ["problem", "givens", "goal", "answer", "proof_outline", "used_objects", "meta"],
                "additionalProperties": False
            },
            "comment": {"type": "string"}
        },
        "required": ["actions", "problem", "comment"],
        "additionalProperties": False
    },
    "strict": True
}


@dataclass
class AuthorVeriScriptAction:
    op: str
    type: str
    args: List[str]


def _parse_author_VeriScript_json(obj: Dict[str, Any]) -> Tuple[List[AuthorVeriScriptAction], str]:
    if not isinstance(obj, dict):
        raise ValueError("Author-VeriScript output must be an object")
    raw_actions = obj.get("actions")
    if raw_actions is None:
        raise ValueError("Author-VeriScript result missing actions field")
    if not isinstance(raw_actions, list):
        raise ValueError("actions must be an array")
    acts: List[AuthorVeriScriptAction] = []
    for idx, action in enumerate(raw_actions):
        if not isinstance(action, dict):
            raise ValueError(f"actions[{idx}] is not an object")
        op = str(action.get("op", "")).strip()
        typ = str(action.get("type", "")).strip()
        args_raw = action.get("args", [])
        if op not in AUTHOR_VeriScript_ALLOWED_OPS:
            raise ValueError(f"actions[{idx}].op not supported: {op}")
        if not typ:
            raise ValueError(f"actions[{idx}] missing type")
        if not isinstance(args_raw, list) or not args_raw:
            raise ValueError(f"actions[{idx}].args must be a non-empty array")
        args: List[str] = []
        for tok in args_raw:
            if isinstance(tok, str):
                args.append(tok.strip())
            elif isinstance(tok, (int, float)):
                args.append(str(tok))
            else:
                raise ValueError(f"actions[{idx}].args contains invalid type: {type(tok).__name__}")
        if op == "AddAuxLine" and len(args) >= 3:
            head = (args[0] or "").upper()
            if (head.startswith("SEG_") and len(head) >= 5 and head[4:].isalpha()) or (len(head) == 2 and head.isalpha()):
                args = args[1:]
        if op == "AddAuxLine" and typ not in AUTHOR_VeriScript_AUX_LINE_TYPES:
            if args and args[0] in AUTHOR_VeriScript_AUX_LINE_TYPES:
                typ = args[0]
                args = args[1:]
        acts.append(AuthorVeriScriptAction(op=op, type=typ, args=args))
    comment = ""
    if isinstance(obj.get("comment"), str):
        comment = obj["comment"].strip()
    return acts, comment


def _print_author_actions(logs: List[Dict[str, Any]]):
    if not logs:
        return
    print("\n[AUTHOR] VeriScript action replay:")
    for rec in logs:
        status = "OK" if rec.get("ok") else "FAIL"
        msg = rec.get("msg", "")
        args = ", ".join(rec.get("args") or [])
        print(f"  Step {rec.get('step', '?'):>2}: {rec.get('op')}({rec.get('type')}, [{args}]) -> {status} {msg}")


def _print_author_problem(problem_json: Dict[str, Any]):
    if not isinstance(problem_json, dict):
        return
    print("\n[AUTHOR] Problem statement (natural language)")
    problem_text = (problem_json.get("problem") or "").strip()
    if problem_text:
        print(f"  Problem: {problem_text}")
    givens = problem_json.get("givens") or []
    if givens:
        print("  Givens:")
        for g in givens:
            print(f"    - {g}")
    goal = (problem_json.get("goal") or "").strip()
    if goal:
        print(f"  Goal: {goal}")
    answer = (problem_json.get("answer") or "").strip()
    if answer:
        print(f"  Answer: {answer}")
    outline = problem_json.get("proof_outline") or []
    if outline:
        print("  Proof outline:")
        for step in outline:
            print(f"    - {step}")
    used_objs = problem_json.get("used_objects") or []
    if used_objs:
        print("  Entities used: ", ", ".join(used_objs))
    print("\n[AUTHOR] Problem JSON:\n" + json.dumps(problem_json, ensure_ascii=False, indent=2))


def _author_require_point(VeriGraph, pid: str):
    if pid not in VeriGraph.entities or VeriGraph.entities[pid].type != "Point":
        raise ValueError(f"Point {pid} does not exist")


def _author_parse_float(value: str, label: str, default: Optional[float] = None) -> float:
    """
    Accepts three input forms: decimal/integer, fraction a/b, surd sqrt(n) (n may also be fraction or integer),
    and simple arithmetic combinations of these (e.g. 2*sqrt(6)/3).
    """
    if value is None:
        if default is not None:
            return default
        raise ValueError(f"{label} is not a valid number: {value}")
    text = str(value).strip()
    if text == "":
        if default is not None:
            return default
        raise ValueError(f"{label} is not a valid number: {value}")
    if "√" in text:
        text = text.replace("√(", "sqrt(")
        text = re.sub(r"√\s*([0-9]+(?:\.[0-9]+)?(?:/[0-9]+(?:\.[0-9]+)?)?)", r"sqrt(\1)", text)
        text = text.replace("√", "sqrt")
        text = re.sub(r"(?<=\d)\s*sqrt", "*sqrt", text)
        text = re.sub(r"(?<=\))\s*sqrt", "*sqrt", text)
    try:
        return float(text)
    except Exception:
        pass

    # Fraction form
    if "/" in text:
        # First try whole fraction like "-60/29" or "sqrt(2)/3"
        try:
            return float(Fraction(text))
        except Exception:
            pass
        # Fallback: parse numerator/denominator separately to allow nested forms
        num, den = text.split("/", 1)
        try:
            num_val = _author_parse_float(num.strip(), label)
            den_val = _author_parse_float(den.strip(), label)
            return num_val / den_val
        except Exception:
            pass

    # Surd form, possibly signed
    sign = 1.0
    inner = text
    if inner[0] in "+-":
        if inner[0] == "-":
            sign = -1.0
        inner = inner[1:].strip()
    if inner.lower().startswith("sqrt(") and inner.endswith(")"):
        inner_expr = inner[5:-1].strip()
        if inner_expr == "":
            raise ValueError(f"{label} is not a valid number: {value}")
        inner_val = _author_parse_float(inner_expr, label)
        if inner_val < 0:
            raise ValueError(f"{label} value under the sqrt must be non-negative: {value}")
        return sign * math.sqrt(inner_val)

    # Simple arithmetic expression (addition/subtraction/multiplication/division combined with sqrt()); avoids the security risks of raw eval
    def _safe_eval_expr(expr: str) -> Optional[float]:
        try:
            tree = ast.parse(expr, mode="eval")
        except Exception:
            return None
        allowed_nodes = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Num,
            ast.Constant,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Pow,
            ast.USub,
            ast.UAdd,
            ast.Load,
            ast.Call,
            ast.Name,
        )
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                return None
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name) or node.func.id != "sqrt":
                    return None
                if node.keywords:
                    return None
            if isinstance(node, ast.Name) and node.id != "sqrt":
                return None
        try:
            return float(eval(compile(tree, "<author_expr>", "eval"), {"__builtins__": {}, "sqrt": math.sqrt}, {}))
        except Exception:
            return None

    expr_val = _safe_eval_expr(text)
    if expr_val is not None:
        return expr_val

    if default is not None:
        return default
    raise ValueError(f"{label} is not a valid number: {value}")


_ALLOWED_FUNCTION_EVAL_FUNCS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "ln": math.log,
    "exp": math.exp,
    "pow": math.pow,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "arcsin": math.asin,
    "arccos": math.acos,
    "arctan": math.atan,
}
_ALLOWED_FUNCTION_CONSTANTS = {"pi": math.pi, "e": math.e}


def _normalize_function_expr(expr: str) -> str:
    if expr is None:
        return ""
    text = str(expr).strip()
    if not text:
        return ""
    if "=" in text:
        text = text.split("=")[-1].strip()
    return text


def _compile_function_expr(expr: str) -> Optional[Callable[[float], Optional[float]]]:
    text = _normalize_function_expr(expr)
    if not text:
        return None
    text = text.replace("^", "**")
    try:
        tree = ast.parse(text, mode="eval")
    except Exception:
        return None

    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Num,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Load,
        ast.Call,
        ast.Name,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            return None
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCTION_EVAL_FUNCS:
                return None
            if node.keywords:
                return None
        if isinstance(node, ast.Name):
            if node.id not in ("x", *(_ALLOWED_FUNCTION_CONSTANTS.keys()), *(_ALLOWED_FUNCTION_EVAL_FUNCS.keys())):
                return None

    def _eval_node(node: ast.AST, x_value: float) -> float:
        if isinstance(node, ast.Expression):
            return _eval_node(node.body, x_value)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError("const not numeric")
        if isinstance(node, ast.Num):
            return float(node.n)
        if isinstance(node, ast.Name):
            if node.id == "x":
                return float(x_value)
            if node.id in _ALLOWED_FUNCTION_CONSTANTS:
                return float(_ALLOWED_FUNCTION_CONSTANTS[node.id])
            raise ValueError("invalid name")
        if isinstance(node, ast.UnaryOp):
            val = _eval_node(node.operand, x_value)
            if isinstance(node.op, ast.UAdd):
                return val
            if isinstance(node.op, ast.USub):
                return -val
            raise ValueError("invalid unary")
        if isinstance(node, ast.BinOp):
            left = _eval_node(node.left, x_value)
            right = _eval_node(node.right, x_value)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left ** right
            raise ValueError("invalid binop")
        if isinstance(node, ast.Call):
            func_name = node.func.id
            func = _ALLOWED_FUNCTION_EVAL_FUNCS[func_name]
            if func_name == "pow":
                if len(node.args) != 2:
                    raise ValueError("invalid call args")
                arg1 = _eval_node(node.args[0], x_value)
                arg2 = _eval_node(node.args[1], x_value)
                return func(arg1, arg2)
            if func_name == "log" and len(node.args) == 2:
                arg1 = _eval_node(node.args[0], x_value)
                arg2 = _eval_node(node.args[1], x_value)
                return func(arg1, arg2)
            if len(node.args) != 1:
                raise ValueError("invalid call args")
            arg_val = _eval_node(node.args[0], x_value)
            return func(arg_val)
        raise ValueError("invalid expr")

    def _eval(x_value: float) -> Optional[float]:
        try:
            val = _eval_node(tree.body, x_value)
        except Exception:
            return None
        if val is None or not math.isfinite(val):
            return None
        return float(val)

    return _eval


_INF_TOKENS_POS = {"+inf", "inf", "+infty", "infty", "+infinity", "infinity", "+oo", "oo"}
_INF_TOKENS_NEG = {"-inf", "-infty", "-infinity", "-oo"}


def _parse_bound_token(VeriGraph: "GeometryGraph", token: str, axis: str = "x") -> float:
    if isinstance(token, str) and token in VeriGraph.entities:
        ent = VeriGraph.entities.get(token)
        if ent and ent.type == "Point":
            x_val, y_val = VeriGraph.get_point_xy(token)
            return x_val if axis == "x" else y_val
    text = str(token).strip().lower()
    if text in _INF_TOKENS_POS:
        return math.inf
    if text in _INF_TOKENS_NEG:
        return -math.inf
    return _author_parse_float(token, "bound")


def _resolve_function_spec(VeriGraph: "GeometryGraph", token: str) -> Tuple[Optional[Callable[[float], Optional[float]]], str]:
    expr = token
    ent = VeriGraph.entities.get(token)
    if ent and ent.type == "Function":
        expr = ent.attrs.data.get("expr") or ent.attrs.data.get("expression") or token
    expr = _normalize_function_expr(expr)
    fn = _compile_function_expr(expr)
    return fn, expr


def _safe_eval_function(fn: Callable[[float], Optional[float]], x_val: float) -> Optional[float]:
    if fn is None:
        return None
    try:
        y_val = fn(float(x_val))
    except Exception:
        return None
    if y_val is None or not math.isfinite(y_val):
        return None
    return float(y_val)


def _parse_point_or_value(VeriGraph: "GeometryGraph", token: str, axis: str, label: str) -> float:
    if isinstance(token, str) and token in VeriGraph.entities:
        ent = VeriGraph.entities.get(token)
        if ent and ent.type == "Point":
            x_val, y_val = VeriGraph.get_point_xy(token)
            return x_val if axis == "x" else y_val
    return _author_parse_float(token, label)

def _parse_derivative_order(token: str) -> int:
    text = str(token).strip().lower()
    if text.startswith("n="):
        text = text[2:].strip()
    val = int(float(text))
    if val < 1:
        raise ValueError("derivative order must be >= 1")
    return val


def _approx_nth_derivative(fn: Callable[[float], Optional[float]], x0: float, order: int) -> Optional[float]:
    if order < 0:
        return None
    if order == 0:
        return _safe_eval_function(fn, x0)
    h_base = max(1e-4, 1e-6 * max(1.0, abs(x0)))
    h = h_base / max(1.0, order)
    denom = h ** order
    total = 0.0
    for k in range(order + 1):
        coeff = (-1) ** k * math.comb(order, k)
        shift = (order / 2.0 - k) * h
        fx = _safe_eval_function(fn, x0 + shift)
        if fx is None:
            return None
        total += coeff * fx
    return total / denom


def _approx_integral_simpson(func: Callable[[float], Optional[float]], a: float, b: float, steps: int) -> Optional[float]:
    if b < a:
        a, b = b, a
    if abs(b - a) < 1e-12:
        return 0.0
    steps = max(50, min(int(steps), 6000))
    if steps % 2 == 1:
        steps += 1
    h = (b - a) / steps
    total = 0.0
    for i in range(steps + 1):
        x = a + i * h
        fx = func(x)
        if fx is None:
            return None
        if i == 0 or i == steps:
            coef = 1.0
        elif i % 2 == 1:
            coef = 4.0
        else:
            coef = 2.0
        total += coef * fx
    return total * h / 3.0


def _approx_integral(fn: Callable[[float], Optional[float]], a: float, b: float,
                     breaks: Optional[List[float]] = None, steps: int = 1200) -> Optional[float]:
    if breaks is None:
        breaks = []
    sign = 1.0
    def _bound_key(val: float) -> float:
        if val == math.inf:
            return float("inf")
        if val == -math.inf:
            return float("-inf")
        return float(val)
    if _bound_key(a) > _bound_key(b):
        a, b = b, a
        sign = -1.0
    finite_breaks = [x for x in breaks if math.isfinite(x)]
    if not math.isfinite(a) and not math.isfinite(b):
        if 0.0 not in finite_breaks:
            finite_breaks.append(0.0)
    finite_breaks = sorted(finite_breaks)
    points = [a]
    for br in finite_breaks:
        if math.isfinite(a) and br <= a:
            continue
        if math.isfinite(b) and br >= b:
            continue
        points.append(br)
    points.append(b)
    open_points = set(finite_breaks)

    def _integrate_finite(l: float, r: float) -> Optional[float]:
        eps = max(1e-8, 1e-6 * max(1.0, abs(r - l)))
        l_adj = l + eps if l in open_points else l
        r_adj = r - eps if r in open_points else r
        if r_adj <= l_adj:
            return 0.0
        return _approx_integral_simpson(lambda x: _safe_eval_function(fn, x), l_adj, r_adj, steps)

    def _integrate_to_inf(l: float) -> Optional[float]:
        eps = 1e-8
        def _g(t: float) -> Optional[float]:
            if t >= 1.0:
                return None
            x = l + t / (1.0 - t)
            fx = _safe_eval_function(fn, x)
            if fx is None:
                return None
            return fx / (1.0 - t) ** 2
        return _approx_integral_simpson(_g, eps, 1.0 - eps, steps)

    def _integrate_from_neg_inf(r: float) -> Optional[float]:
        eps = 1e-8
        def _g(t: float) -> Optional[float]:
            if t >= 1.0:
                return None
            x = r - t / (1.0 - t)
            fx = _safe_eval_function(fn, x)
            if fx is None:
                return None
            return fx / (1.0 - t) ** 2
        return _approx_integral_simpson(_g, eps, 1.0 - eps, steps)

    total = 0.0
    for i in range(len(points) - 1):
        l = points[i]
        r = points[i + 1]
        if l == r:
            continue
        if math.isfinite(l) and math.isfinite(r):
            part = _integrate_finite(l, r)
        elif math.isfinite(l) and not math.isfinite(r):
            part = _integrate_to_inf(l)
        elif not math.isfinite(l) and math.isfinite(r):
            part = _integrate_from_neg_inf(r)
        else:
            left = _integrate_from_neg_inf(0.0)
            right = _integrate_to_inf(0.0)
            if left is None or right is None:
                return None
            part = left + right
        if part is None:
            return None
        total += part
    return sign * total


def _parse_points_on_function_args(VeriGraph: "GeometryGraph", tokens: List[str]) -> Tuple[float, List[Tuple[str, Any]]]:
    if not tokens:
        raise ValueError("VerifyFunction-PointsOnFunction requires at least one point or coordinate pair")
    work = list(tokens)
    tol = 1e-4
    if len(work) >= 2:
        try:
            tol_candidate = _author_parse_float(work[-1], "tol")
            if len(work) % 2 == 1 or work[-1] not in VeriGraph.entities:
                tol = tol_candidate
                work = work[:-1]
        except Exception:
            pass
    items: List[Tuple[str, Any]] = []
    idx = 0
    while idx < len(work):
        tok = work[idx]
        ent = VeriGraph.entities.get(tok)
        if ent and ent.type == "Point":
            items.append(("point", tok))
            idx += 1
            continue
        if idx + 1 >= len(work):
            raise ValueError("VerifyFunction-PointsOnFunction requires coordinate values to be paired")
        x_val = _author_parse_float(work[idx], "x")
        y_val = _author_parse_float(work[idx + 1], "y")
        items.append(("coord", (x_val, y_val)))
        idx += 2
    if not items:
        raise ValueError("VerifyFunction-PointsOnFunction requires at least one point or coordinate pair")
    return tol, items

def _author_create_angle(VeriGraph, angle_id: str, verts: Tuple[str, str, str]) -> str:
    for pid in verts:
        _author_require_point(VeriGraph, pid)
    raw_id = (angle_id or "").strip()
    if not raw_id or not raw_id.startswith("∠"):
        angle_id = f"∠{''.join(verts)}"
    else:
        angle_id = raw_id
    name = angle_id
    ent = VeriGraph.entities.get(angle_id)
    if ent and ent.type != "Angle":
        raise ValueError(f"{angle_id} already exists but is not an angle")
    attrs = Attrs.angle(name, provenance="author")
    aliases: List[str] = []
    if raw_id and raw_id != angle_id:
        aliases.append(raw_id)
    if ent:
        existing_aliases = ent.attrs.data.get("aliases") or ent.attrs.data.get("alias")
        if isinstance(existing_aliases, str):
            aliases.append(existing_aliases)
        elif isinstance(existing_aliases, (list, tuple, set)):
            for alias in existing_aliases:
                if isinstance(alias, str):
                    aliases.append(alias)
    if aliases:
        seen = set()
        clean: List[str] = []
        for alias in aliases:
            alias = alias.strip()
            if not alias or alias in seen:
                continue
            seen.add(alias)
            clean.append(alias)
        if clean:
            attrs.data["aliases"] = clean
    if ent is None:
        VeriGraph.add_entity("Angle", id=angle_id, attrs=attrs)
    else:
        VeriGraph.entities[angle_id].attrs = attrs
    return angle_id


def _author_create_triangle(VeriGraph, tri_id: str, verts: Tuple[str, str, str]) -> str:
    for pid in verts:
        _author_require_point(VeriGraph, pid)
    if not tri_id:
        tri_id = f"TRI_{''.join(verts)}"
    ent = VeriGraph.entities.get(tri_id)
    if ent and ent.type != "Triangle":
        raise ValueError(f"{tri_id} already exists but is not a triangle")
    attrs = Attrs.triangle(list(verts), provenance="author")
    if ent is None:
        VeriGraph.add_entity("Triangle", id=tri_id, attrs=attrs)
    else:
        VeriGraph.entities[tri_id].attrs = attrs
    # auto-create edge segments for triangle sides
    try:
        _ensure_segment_between_points(VeriGraph, verts[0], verts[1])
        _ensure_segment_between_points(VeriGraph, verts[1], verts[2])
        _ensure_segment_between_points(VeriGraph, verts[2], verts[0])
    except Exception:
        pass
    return tri_id


def _author_create_quadrilateral(VeriGraph, quad_id: str, verts: Tuple[str, str, str, str], kind: str) -> str:
    kind_map = {
        "trapezoid": "Trapezoid",
        "parallelogram": "Parallelogram",
        "rectangle": "Rectangle",
        "square": "Square",
        "rhombus": "Rhombus",
    }
    kind_norm = kind_map.get(str(kind).strip().lower())
    if not kind_norm:
        raise ValueError(f"AddQuadrilateral type not supported: {kind}")
    for pid in verts:
        _author_require_point(VeriGraph, pid)
    if len(set(verts)) < 4:
        raise ValueError("AddQuadrilateral requires 4 distinct points")
    if not quad_id:
        prefix = {
            "Trapezoid": "TRAP",
            "Parallelogram": "PARA",
            "Rectangle": "RECT",
            "Square": "SQR",
            "Rhombus": "RHOM",
        }.get(kind_norm, "QUAD")
        quad_id = f"{prefix}_{''.join(verts)}"
    ent = VeriGraph.entities.get(quad_id)
    if ent and ent.type != "Quadrilateral":
        raise ValueError(f"{quad_id} already exists but has type {ent.type}")
    attrs = Attrs.quad(list(verts), kind=kind_norm, provenance="author")
    if ent is None:
        VeriGraph.add_entity("Quadrilateral", id=quad_id, attrs=attrs)
    else:
        VeriGraph.entities[quad_id].attrs = attrs

    segs: List[str] = []
    edges = [
        (verts[0], verts[1]),
        (verts[1], verts[2]),
        (verts[2], verts[3]),
        (verts[3], verts[0]),
    ]
    for u, v in edges:
        segs.append(_ensure_segment_between_points(VeriGraph, u, v))

    def _regularize_quad_points():
        # If C, D are on the same side of AB and the projection order is opposite to the expected ABCD order, swap C/D coordinates to avoid the common arrangement error of "C ending up to the left of D" (applies to all quadrilaterals)
        try:
            ax, ay = VeriGraph.get_point_xy(verts[0]); bx, by = VeriGraph.get_point_xy(verts[1])
            cx, cy = VeriGraph.get_point_xy(verts[2]); dx, dy = VeriGraph.get_point_xy(verts[3])
        except Exception:
            return
        ab = (bx - ax, by - ay)
        ab_len = math.hypot(*ab)
        if ab_len < 1e-9:
            return
        cross_c = (cx - ax) * ab[1] - (cy - ay) * ab[0]
        cross_d = (dx - ax) * ab[1] - (dy - ay) * ab[0]
        if cross_c == 0 or cross_d == 0 or cross_c * cross_d <= 0:
            return
        proj_c = (cx - ax) * ab[0] + (cy - ay) * ab[1]
        proj_d = (dx - ax) * ab[0] + (dy - ay) * ab[1]
        if proj_c < proj_d:
            VeriGraph.entities[verts[2]].attrs.data["coord"] = {"x": float(dx), "y": float(dy)}
            VeriGraph.entities[verts[3]].attrs.data["coord"] = {"x": float(cx), "y": float(cy)}

    _regularize_quad_points()

    def _add_constraint_once(cons_type: str, pair: Tuple[str, str]) -> Optional[str]:
        scope_set = set(pair)
        for cid, cons in VeriGraph.constraints.items():
            if cons.type == cons_type and set(cons.scope) == scope_set:
                return cid
        return VeriGraph.add_constraint(cons_type, list(pair), source="author", apply_immediate=False)

    def _segments_parallel(seg_a: str, seg_b: str) -> bool:
        for cons in VeriGraph.constraints.values():
            if cons.type == "Parallel" and set(cons.scope) == {seg_a, seg_b}:
                return True
        try:
            a1, a2 = _segment_endpoints_for_VeriScript(VeriGraph, seg_a)
            b1, b2 = _segment_endpoints_for_VeriScript(VeriGraph, seg_b)
            ax1, ay1 = VeriGraph.get_point_xy(a1); ax2, ay2 = VeriGraph.get_point_xy(a2)
            bx1, by1 = VeriGraph.get_point_xy(b1); bx2, by2 = VeriGraph.get_point_xy(b2)
        except Exception:
            return False
        va = (ax2 - ax1, ay2 - ay1)
        vb = (bx2 - bx1, by2 - by1)
        na = math.hypot(*va); nb = math.hypot(*vb)
        if na < 1e-9 or nb < 1e-9:
            return False
        cross = va[0] * vb[1] - va[1] * vb[0]
        return abs(cross) <= 1e-9 * na * nb

    base_pair = (segs[0], segs[2])  # AB ∥ CD
    other_pair = (segs[1], segs[3])  # BC and AD
    if kind_norm == "Trapezoid":
        if _segments_parallel(*other_pair):
            raise ValueError("Trapezoid degenerates to parallelogram: the other pair of opposite sides is also parallel")
        _add_constraint_once("Parallel", base_pair)
    else:
        _add_constraint_once("Parallel", base_pair)
        _add_constraint_once("Parallel", other_pair)
        if kind_norm in ("Rectangle", "Square"):
            _add_constraint_once("Perpendicular", (segs[0], segs[1]))
        if kind_norm in ("Rhombus", "Square"):
            _add_constraint_once("EqualLength", (segs[0], segs[1]))
            _add_constraint_once("EqualLength", (segs[1], segs[2]))
            _add_constraint_once("EqualLength", (segs[2], segs[3]))
    return quad_id


def _apply_author_VeriScript_actions(VeriGraph, actions: List[AuthorVeriScriptAction]) -> List[Dict[str, Any]]:
    logs: List[Dict[str, Any]] = []
    derived_records: List[Dict[str, Any]] = []
    VeriGraph.meta["author_strict_mode"] = True

    def _as_ratio(tokens: List[str]) -> Optional[float]:
        """Parse ratio; return None if not provided or invalid, meaning "on the segment but with position not locked"."""
        if len(tokens) < 3:
            return None
        try:
            return float(Fraction(str(tokens[2])))
        except Exception:
            return None

    def _normalize_existing_id(eid: str) -> str:
        """Only normalize alias for existing entities; do not create new ones."""
        if not isinstance(eid, str):
            return eid
        if eid in VeriGraph.entities:
            return eid
        alias_map = _collect_alias_map(VeriGraph)
        return alias_map.get(eid, eid)

    for idx, action in enumerate(actions, start=1):
        ok, msg = True, "ok"
        try:
            if action.op == "AddPoint":
                if not action.args:
                    raise ValueError("AddPoint missing new point ID")
                pid = action.args[0]
                if pid in VeriGraph.entities:
                    raise ValueError(f"Point {pid} already exists")
                if action.type == "Free":
                    x, y = _suggest_free_point_xy(VeriGraph)
                    VeriGraph.add_entity("Point", id=pid, attrs=Attrs.point(x, y, provenance="author"))
                    try:
                        VeriGraph.entities[pid].attrs.data["origin"] = "free"
                    except Exception:
                        pass
                    msg = f"Added free point {pid}"
                elif action.type == "OnSegment":
                    if len(action.args) < 2:
                        raise ValueError("AddPoint-OnSegment must provide segment ID")
                    seg_id_raw = action.args[1]
                    seg_id = _normalize_existing_id(seg_id_raw)
                    seg = VeriGraph.entities.get(seg_id)
                    if not seg or seg.type != "Segment":
                        raise ValueError(f"{seg_id_raw} is not a valid segment (normalized to {seg_id})")
                    u, v = _segment_endpoints_for_VeriScript(VeriGraph, seg_id)
                    ux, uy = VeriGraph.get_point_xy(u)
                    vx, vy = VeriGraph.get_point_xy(v)
                    ratio_val = _as_ratio(action.args)
                    # Use the midpoint as the initial coordinate, but do not hard-code ratio, so later constraints like foot-of-perpendicular can determine the exact position
                    seed_ratio = 0.5 if ratio_val is None else max(0.0, min(1.0, ratio_val))
                    x = ux + (vx - ux) * seed_ratio
                    y = uy + (vy - uy) * seed_ratio
                    VeriGraph.add_entity("Point", id=pid, attrs=Attrs.point(x, y, provenance="author"))
                    _ensure_incidence(VeriGraph, pid, seg_id)
                    try:
                        params = {"ratio": max(0.0, min(1.0, ratio_val))} if ratio_val is not None else None
                        VeriGraph.add_constraint("PointOnSegment", [pid, seg_id], params=params or {}, source="author")
                    except Exception:
                        pass
                    msg = f"Placed point {pid} on segment {seg_id}"
                elif action.type in ("OnCircle", "PointOnCircle"):
                    if len(action.args) < 2:
                        raise ValueError("AddPoint-OnCircle must provide circle ID")
                    circle_raw = action.args[1]
                    circle_id = _normalize_existing_id(circle_raw)
                    circle_ent = VeriGraph.entities.get(circle_id)
                    if not circle_ent or circle_ent.type != "Circle":
                        raise ValueError(f"{circle_raw} is not a valid circle (normalized to {circle_id})")
                    params = VeriGraph._circle_params(circle_id)
                    if not params:
                        raise ValueError(f"{circle_id} circle parameters missing")
                    cx, cy, r, _ = params
                    if r <= 1e-9:
                        r = 1.0
                    x = cx + r
                    y = cy
                    VeriGraph.add_entity("Point", id=pid, attrs=Attrs.point(x, y, provenance="author"))
                    _ensure_on_circle(VeriGraph, pid, circle_id, source="author")
                    msg = f"Placed point {pid} on circle {circle_id}"
                elif action.type == "Cartesian":
                    if len(action.args) < 3:
                        raise ValueError("AddPoint-Cartesian requires x,y")
                    x = _author_parse_float(action.args[1], "x")
                    y = _author_parse_float(action.args[2], "y")
                    VeriGraph.add_entity("Point", id=pid, attrs=Attrs.point(x, y, provenance="given"))
                    msg = f"Set point {pid} = ({x:.3f}, {y:.3f})"
                else:
                    raise ValueError(f"Unknown AddPoint type: {action.type}")
            elif action.op == "MovePoint":
                if len(action.args) < 3:
                    raise ValueError("MovePoint requires point ID and new coordinates")
                pid = action.args[0]
                _author_require_point(VeriGraph, pid)
                x = _author_parse_float(action.args[1], "x")
                y = _author_parse_float(action.args[2], "y")
                ent = VeriGraph.entities[pid]
                ent.attrs.data.setdefault("coord", {})
                ent.attrs.data["coord"]["x"] = float(x)
                ent.attrs.data["coord"]["y"] = float(y)
                ent.attrs.data["provenance"] = "author_move"
                msg = f"Moved point {pid} to ({x:.3f},{y:.3f})"
            elif action.op == "AddAuxLine":
                args = list(action.args)
                seg_alias = None
                point_lookup = _build_point_lookup(VeriGraph)
                # Fault tolerance: if mistakenly written as ["SEG_AB","A","B"] / ["AB","A","B"], drop the segment token and keep only the endpoints
                if len(args) >= 3:
                    head_raw = (args[0] or "").strip()
                    head = head_raw.upper()
                    if head_raw:
                        if (head.startswith("SEG_") and len(head) >= 5 and head[4:].isalpha()) or (len(head) == 2 and head.isalpha()):
                            seg_alias = head_raw
                            args = args[1:]
                        else:
                            head_is_point = head in point_lookup
                            p1 = point_lookup.get(str(args[1]).upper()) or args[1]
                            p2 = point_lookup.get(str(args[2]).upper()) or args[2]
                            if (not head_is_point) and _is_point_entity(VeriGraph, p1) and _is_point_entity(VeriGraph, p2):
                                seg_alias = head_raw
                                args = args[1:]
                elif len(args) == 2:
                    head = (args[0] or "").upper()
                    if (head.startswith("SEG_") and len(head) >= 5 and head[4:].isalpha()) or (len(head) == 2 and head.isalpha()):
                        args = args[1:]
                if len(args) < 2:
                    raise ValueError("AddAuxLine must provide two points")
                p_raw, q_raw = args[:2]
                p = point_lookup.get(p_raw.upper()) or p_raw
                q = point_lookup.get(q_raw.upper()) or q_raw
                _author_require_point(VeriGraph, p)
                _author_require_point(VeriGraph, q)
                seg_id = _ensure_segment_between_points(VeriGraph, p, q)
                if seg_alias and seg_alias != seg_id:
                    ent = VeriGraph.entities.get(seg_id)
                    if ent:
                        existing = ent.attrs.data.get("aliases") or ent.attrs.data.get("alias")
                        aliases: List[str] = []
                        if isinstance(existing, str):
                            aliases.append(existing)
                        elif isinstance(existing, (list, tuple, set)):
                            for alias in existing:
                                if isinstance(alias, str):
                                    aliases.append(alias)
                        seg_alias = seg_alias.strip()
                        if seg_alias and seg_alias not in aliases:
                            aliases.append(seg_alias)
                        if aliases:
                            ent.attrs.data["aliases"] = aliases
                msg = f"Created/reused segment {seg_id}"
            elif action.op == "AddEdge":
                if not action.args:
                    raise ValueError("AddEdge must provide target entities")
                scope: Tuple[str, ...]
                if action.type == "AngleBisector":
                    normalized_scope = _normalize_angle_bisector_scope(VeriGraph, tuple(action.args))
                    scope = tuple(_normalize_named_entity_token(VeriGraph, tok) for tok in normalized_scope)
                elif action.type == "Midpoint":
                    scope = _normalize_midpoint_scope(VeriGraph, list(action.args))
                elif action.type in _SEGMENT_EDGE_TYPES:
                    scope = _normalize_segment_edge_scope(VeriGraph, list(action.args), action.type)
                else:
                    point_lookup = _build_point_lookup(VeriGraph)
                    scope = tuple(_normalize_named_entity_token(VeriGraph, tok, point_lookup) for tok in action.args)
                VeriGraph.add_constraint(action.type, list(scope), source="author", apply_immediate=False)
                msg = f"Added constraint {action.type}"
            elif action.op == "AddCircle":
                if action.type not in ("CenterRadius", "CenterPoint"):
                    raise ValueError(f"AddCircle type not supported: {action.type}")
                if len(action.args) < 3:
                    raise ValueError("AddCircle requires at least 3 arguments")
                circle_id = action.args[0]
                center_id = action.args[1]
                through_id = None
                _author_require_point(VeriGraph, center_id)
                if circle_id in VeriGraph.entities and VeriGraph.entities[circle_id].type != "Circle":
                    raise ValueError(f"{circle_id} already exists but is not a circle")
                if action.type == "CenterRadius":
                    radius = _author_parse_float(action.args[2], "radius")
                else:
                    through_id = action.args[2]
                    _author_require_point(VeriGraph, through_id)
                    cxy = VeriGraph.get_point_xy(center_id)
                    txy = VeriGraph.get_point_xy(through_id)
                    if cxy and txy:
                        radius = max(1e-6, _dist(cxy, txy))
                    else:
                        radius = 1.0
                attrs = Attrs.circle(center_id, radius, provenance="author")
                attrs.data["center_id"] = center_id
                if through_id:
                    attrs.data["through_id"] = through_id
                if circle_id in VeriGraph.entities:
                    VeriGraph.entities[circle_id].attrs = attrs
                else:
                    VeriGraph.add_entity("Circle", id=circle_id, attrs=attrs)
                msg = f"Adjusted/created circle {circle_id}"
            elif action.op == "AddFunction":
                if len(action.args) < 2:
                    raise ValueError("AddFunction requires function ID and expression")
                func_id = action.args[0]
                expr_text = str(action.args[1]).strip()
                if not _normalize_function_expr(expr_text):
                    raise ValueError("AddFunction expression is empty")
                if _compile_function_expr(expr_text) is None:
                    raise ValueError(f"AddFunction expression cannot be parsed: {expr_text}")
                xmin_raw = action.args[2] if len(action.args) >= 3 else "-5"
                xmax_raw = action.args[3] if len(action.args) >= 4 else "5"
                samples_raw = action.args[4] if len(action.args) >= 5 else "240"
                xmin_val = _author_parse_float(xmin_raw, "x_min")
                xmax_val = _author_parse_float(xmax_raw, "x_max")
                if xmax_val < xmin_val:
                    xmin_val, xmax_val = xmax_val, xmin_val
                if abs(xmax_val - xmin_val) < 1e-9:
                    xmax_val = xmin_val + 1.0
                try:
                    samples_val = int(str(samples_raw).strip())
                except Exception:
                    samples_val = 240
                samples_val = max(20, min(samples_val, 2000))
                ent = VeriGraph.entities.get(func_id)
                if ent and ent.type != "Function":
                    raise ValueError(f"{func_id} already exists but is not a function")
                attrs = Attrs({
                    "expr": expr_text,
                    "xmin": float(xmin_val),
                    "xmax": float(xmax_val),
                    "samples": int(samples_val),
                    "provenance": "author",
                })
                if ent:
                    VeriGraph.entities[func_id].attrs = attrs
                else:
                    VeriGraph.add_entity("Function", id=func_id, attrs=attrs)
                msg = f"Created function {func_id}: y={_normalize_function_expr(expr_text)}"
            elif action.op == "AddAngle":
                if len(action.args) < 4:
                    raise ValueError("AddAngle requires angle ID + 3 points")
                angle_id = action.args[0]
                verts = tuple(action.args[1:4])
                for pid in verts:
                    _author_require_point(VeriGraph, pid)
                angle_id = _author_create_angle(VeriGraph, angle_id, verts)
                msg = f"Registered angle {angle_id}"
            elif action.op == "AddTriangle":
                if len(action.args) < 4:
                    raise ValueError("AddTriangle requires triangle ID and 3 points")
                tri_id = action.args[0]
                verts = tuple(action.args[1:4])
                tri_id = _author_create_triangle(VeriGraph, tri_id=tri_id, verts=verts)
                msg = f"Registered triangle {tri_id}"
            elif action.op == "AddQuadrilateral":
                if len(action.args) < 5:
                    raise ValueError("AddQuadrilateral requires quadrilateral ID and 4 points")
                quad_id = action.args[0]
                verts = tuple(action.args[1:5])
                quad_id = _author_create_quadrilateral(VeriGraph, quad_id=quad_id, verts=verts, kind=action.type)
                msg = f"Registered {action.type} {quad_id}"
            else:
                raise ValueError(f"Unknown operation: {action.op}")
        except Exception as exc:
            ok, msg = False, f"{action.op} failed: {exc}"
        finally:
            after_action_constraints = set(VeriGraph.constraints.keys())
            try:
                DerivedRuleEngine.apply(VeriGraph)
            except Exception:
                pass
            else:
                final_constraints = set(VeriGraph.constraints.keys())
                new_ids = final_constraints - after_action_constraints
                for nid in new_ids:
                    cons = VeriGraph.constraints[nid]
                    derived_records.append({"id": nid, "type": cons.type, "scope": list(cons.scope), "source": cons.source})
        logs.append({
            "step": idx,
            "op": action.op,
            "type": action.type,
            "args": action.args,
            "ok": ok,
            "msg": msg,
        })
    post_step = len(actions) + 1
    try:
        VeriGraph.VeriSolve_incidence(ignore_protection=True)
        logs.append({
            "step": post_step,
            "op": "VeriSolve",
            "type": "PostProcess",
            "args": [],
            "ok": True,
            "msg": "Hard-constraint resolution complete: line/segment/circle intersections placed; order-independent",
        })
    except Exception as exc:
        logs.append({
            "step": post_step,
            "op": "VeriSolve",
            "type": "PostProcess",
            "args": [],
            "ok": False,
            "msg": f"VeriSolve_incidence failed: {exc}",
        })
    post_step += 1

    if CONFIG.author_enable_enforce:
        try:
            VeriGraph.polish_soft_constraints(ignore_protection=True)
            logs.append({
                "step": post_step,
                "op": "SoftPolish",
                "type": "Author",
                "args": [],
                "ok": True,
                "msg": "Soft polish complete (heuristic fine-tuning of former enforce)",
            })
        except Exception as exc:
            logs.append({
                "step": post_step,
                "op": "SoftPolish",
                "type": "Author",
                "args": [],
                "ok": False,
                "msg": f"soft polish failed: {exc}",
            })
        post_step += 1

    try:
        snap_stats = VeriGraph.snap_free_rays_first_hit()
        logs.append({
            "step": post_step,
            "op": "SnapFreeRays",
            "type": "PostProcess",
            "args": [],
            "ok": True,
            "msg": f"Free-ray snap complete: snapped={snap_stats.get('snapped', 0)}, hidden={snap_stats.get('hidden', 0)}, total={snap_stats.get('total', 0)}",
        })
    except Exception as exc:
        logs.append({
            "step": post_step,
            "op": "SnapFreeRays",
            "type": "PostProcess",
            "args": [],
            "ok": False,
            "msg": f"Free-ray snap failed: {exc}",
        })

    return logs


def _analyze_action_logs(logs: Optional[List[Dict[str, Any]]]) -> Tuple[bool, List[str]]:
    numeric_ok = True
    issues: List[str] = []
    for entry in logs or []:
        if not isinstance(entry, dict):
            continue
        step = entry.get("step")
        op = entry.get("op") or ""
        typ = entry.get("type") or ""
        label = f"step {step}: {op}/{typ}".strip()
        msg = entry.get("msg")
        ok_flag = entry.get("ok")
        geom_flag = entry.get("geom_ok")
        if ok_flag is False:
            numeric_ok = False
            issues.append(f"{label} execution failed: {msg}")
        if geom_flag is False:
            numeric_ok = False
            issues.append(f"{label} numeric check did not pass")
        if op == "VeriSolve" and ok_flag is False:
            numeric_ok = False
            issues.append(f"{label} intersection failed: {msg}")
        verification = entry.get("verification")
        if isinstance(verification, dict):
            for k, v in verification.items():
                if isinstance(v, dict) and v.get("ok") is False:
                    numeric_ok = False
                    issues.append(f"{label} validation {k} failed: {v.get('msg') or v}")
    return numeric_ok, issues


def _collect_numeric_hints(VeriGraph, limit: int = 15) -> str:
    """Collect partial metrics usable for numeric problems to help the LLM build computation-style problems."""
    hints: List[str] = []
    try:
        ends = NaturalLanguageDescriber._segment_endpoints(VeriGraph)
    except Exception:
        ends = {}
    for sid in sorted(ends.keys()):
        if len(hints) >= limit:
            break
        u, v = ends[sid]
        try:
            ux, uy = VeriGraph.get_point_xy(u)
            vx, vy = VeriGraph.get_point_xy(v)
        except Exception:
            continue
        sq = (ux - vx) ** 2 + (uy - vy) ** 2
        hints.append(f"{sid}: |{u}{v}| = {_format_sqrt_from_square(sq)}")
    circles = [eid for eid, e in VeriGraph.entities.items() if e.type == "Circle"]
    for cid in sorted(circles):
        if len(hints) >= limit:
            break
        attrs = VeriGraph.entities[cid].attrs.data
        center = attrs.get("center")
        radius = float(attrs.get("radius", 0.0))
        hints.append(f"{cid}: radius = {_format_fraction(radius)} (center {center})")
    return "\n".join(hints)


def _collect_point_catalog(VeriGraph) -> List[Dict[str, Any]]:
    catalog: List[Dict[str, Any]] = []
    for pid, ent in sorted(VeriGraph.entities.items()):
        if ent.type != "Point":
            continue
        try:
            x, y = VeriGraph.get_point_xy(pid)
        except Exception:
            continue
        catalog.append({
            "id": pid,
            "x": x,
            "y": y,
            "formatted": [_format_fraction(x), _format_fraction(y)],
            "provenance": ent.attrs.data.get("provenance", "")
        })
    return catalog


def _redact_nl_coordinates(nl_text: str) -> str:
    if not nl_text:
        return ""
    lines = str(nl_text).splitlines()
    out: List[str] = []
    in_point_block = False
    for line in lines:
        if line.startswith("Point coordinates"):
            in_point_block = True
            continue
        if in_point_block:
            if line.startswith("  -"):
                continue
            in_point_block = False
        if "center" in line:
            line = re.sub(r"(center\\s*[^\\s\\(,]+)\\([^\\)]*\\)", r"\\1", line)
        out.append(line)
    return "\n".join(out)


def _collect_segment_catalog(VeriGraph) -> List[Dict[str, Any]]:
    catalog: List[Dict[str, Any]] = []
    try:
        ends = NaturalLanguageDescriber._segment_endpoints(VeriGraph)
    except Exception:
        ends = {}
    for sid, (u, v) in sorted(ends.items()):
        catalog.append({"id": sid, "points": [u, v]})
    return catalog


def _collect_angle_catalog(VeriGraph) -> List[Dict[str, Any]]:
    catalog: List[Dict[str, Any]] = []
    for aid, ent in sorted(VeriGraph.entities.items()):
        if ent.type != "Angle":
            continue
        name = ent.attrs.data.get("name") or aid
        label = name if isinstance(name, str) else aid
        token = label.replace("∠", "") if isinstance(label, str) else ""
        verts: List[str] = []
        if len(token) == 3 and token.isalpha():
            verts = list(token.upper())
        catalog.append({"id": aid, "label": label, "verts": verts})
    return catalog



def _build_solver_payload(allowed_ids: List[str], clean_constraints: List[Dict[str, Any]],
                          problem_text: str, goal_text: str,
                          angle_catalog: Optional[List[Dict[str, Any]]] = None):
    """
    Build the Solver input payload (structured); only provides output-format hints.
    """
    schema_hint = {
        "result_schema": {
            "proof": "<proof>",
            "actions": [
                {"op": "<op>", "type": "<type>", "args": ["<arg1>", "<arg2>"]}
            ],
            "final_answer": "<final_answer>"
        }
    }
    return {
        "allowed_ids": allowed_ids,
        "clean_constraints": clean_constraints,
        "angle_catalog": angle_catalog or [],
        "problem": problem_text,
        "goal": goal_text,
        "VeriScript_spec": schema_hint
    }


_LLM_INFLIGHT = 0
_LLM_INFLIGHT_LOCK = threading.Lock()
_LLM_METRICS = {"count": 0, "errors": 0, "total_secs": 0.0, "max_secs": 0.0}


def _now_timestamp_ms() -> str:
    t = time.time()
    base = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))
    ms = int((t - int(t)) * 1000)
    return f"{base}.{ms:03d}"


def _worker_label() -> str:
    try:
        proc_name = mp.current_process().name
    except Exception:
        proc_name = "unknown"
    return f"pid={os.getpid()} proc={proc_name} thread={threading.current_thread().name}"


def _inc_llm_inflight() -> int:
    global _LLM_INFLIGHT
    with _LLM_INFLIGHT_LOCK:
        _LLM_INFLIGHT += 1
        return _LLM_INFLIGHT


def _dec_llm_inflight() -> int:
    global _LLM_INFLIGHT
    with _LLM_INFLIGHT_LOCK:
        _LLM_INFLIGHT = max(0, _LLM_INFLIGHT - 1)
        return _LLM_INFLIGHT


def _record_llm_metrics(duration_secs: float, ok: bool) -> None:
    with _LLM_INFLIGHT_LOCK:
        _LLM_METRICS["count"] += 1
        if not ok:
            _LLM_METRICS["errors"] += 1
        _LLM_METRICS["total_secs"] += float(duration_secs)
        if duration_secs > _LLM_METRICS["max_secs"]:
            _LLM_METRICS["max_secs"] = float(duration_secs)


def _reset_llm_metrics() -> None:
    global _LLM_INFLIGHT
    with _LLM_INFLIGHT_LOCK:
        _LLM_INFLIGHT = 0
        _LLM_METRICS["count"] = 0
        _LLM_METRICS["errors"] = 0
        _LLM_METRICS["total_secs"] = 0.0
        _LLM_METRICS["max_secs"] = 0.0
    with _LLM_CACHE_LOCK:
        _LLM_CACHE_STATS["hits"] = 0
        _LLM_CACHE_STATS["misses"] = 0
        _LLM_CACHE_STATS["evictions"] = 0
        _LLM_CACHE_STATS["errors"] = 0
        _LLM_CACHE_STATS["hit_prompt_tokens"] = 0
        _LLM_CACHE_STATS["hit_completion_tokens"] = 0
        _LLM_CACHE_STATS["miss_prompt_tokens"] = 0
        _LLM_CACHE_STATS["miss_completion_tokens"] = 0
        _LLM_CACHE_STATS["api_cache_read_tokens"] = 0
        _LLM_CACHE_STATS["api_cache_create_tokens"] = 0


def _log_llm_metrics_summary(label: str) -> None:
    if not CONFIG.llm_log_timing:
        return
    with _LLM_INFLIGHT_LOCK:
        count = int(_LLM_METRICS["count"])
        errors = int(_LLM_METRICS["errors"])
        total_secs = float(_LLM_METRICS["total_secs"])
        max_secs = float(_LLM_METRICS["max_secs"])
    if count <= 0:
        return
    avg_secs = total_secs / max(1, count)
    print(
        f"[LLM][SUMMARY] {label} count={count} avg={avg_secs:.2f}s "
        f"max={max_secs:.2f}s errors={errors} {_worker_label()}"
    )
    if CONFIG.llm_cache_enable:
        _log_llm_cache_summary(label)
_LLM_CACHE_LOCK = threading.Lock()
_LLM_CACHE_STATS: Dict[str, int] = {
    "hits": 0, "misses": 0, "evictions": 0, "errors": 0,
    "hit_prompt_tokens": 0, "hit_completion_tokens": 0,
    "miss_prompt_tokens": 0, "miss_completion_tokens": 0,
    "api_cache_read_tokens": 0, "api_cache_create_tokens": 0,
}
_LLM_CACHE_WRITE_COUNT = 0


def _llm_cache_dir() -> str:
    configured = (CONFIG.llm_cache_dir or "").strip()
    if configured:
        return configured
    root = (CONFIG.output_root or "").strip() or os.getcwd()
    return os.path.join(root, ".llm_cache")


def _llm_cache_key(body: dict) -> str:
    canonical = {
        "model": body.get("model", ""),
        "messages": body.get("messages", []),
        "temperature": body.get("temperature"),
        "max_tokens": body.get("max_tokens"),
        "response_format": body.get("response_format"),
    }
    raw = json.dumps(canonical, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _llm_cache_path(cache_key: str) -> str:
    return os.path.join(_llm_cache_dir(), cache_key[:2], cache_key + ".json")


def _llm_cache_read(cache_key: str) -> Optional[dict]:
    path = _llm_cache_path(cache_key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            entry = json.load(f)
        cached_at = entry.get("cached_at", 0)
        if time.time() - cached_at > CONFIG.llm_cache_ttl_seconds:
            try:
                os.remove(path)
            except OSError:
                pass
            return None
        return entry
    except Exception as exc:
        with _LLM_CACHE_LOCK:
            _LLM_CACHE_STATS["errors"] += 1
        print(f"[LLM][CACHE] read error {cache_key[:12]}...: {exc}")
        return None


def _llm_cache_write(cache_key: str, body: dict, response_json: dict, label: str) -> None:
    global _LLM_CACHE_WRITE_COUNT
    path = _llm_cache_path(cache_key)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        entry = {
            "cache_key": cache_key,
            "cached_at": time.time(),
            "model": body.get("model", ""),
            "label": label,
            "temperature": body.get("temperature"),
            "usage": response_json.get("usage", {}),
            "response": response_json,
        }
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False)
        shutil.move(tmp_path, path)
        _LLM_CACHE_WRITE_COUNT += 1
        if _LLM_CACHE_WRITE_COUNT % 50 == 0:
            _llm_cache_enforce_size_limit()
    except Exception as exc:
        with _LLM_CACHE_LOCK:
            _LLM_CACHE_STATS["errors"] += 1
        print(f"[LLM][CACHE] write error {cache_key[:12]}...: {exc}")


def _llm_cache_enforce_size_limit() -> None:
    cache_dir = _llm_cache_dir()
    if not os.path.isdir(cache_dir):
        return
    max_bytes = CONFIG.llm_cache_max_size_mb * 1024 * 1024
    try:
        entries = []
        for dirpath, _, filenames in os.walk(cache_dir):
            for fn in filenames:
                if not fn.endswith(".json"):
                    continue
                fp = os.path.join(dirpath, fn)
                try:
                    st = os.stat(fp)
                    entries.append((fp, st.st_mtime, st.st_size))
                except OSError:
                    pass
        total_size = sum(e[2] for e in entries)
        if total_size <= max_bytes:
            return
        entries.sort(key=lambda e: e[1])
        evicted = 0
        for fp, _, sz in entries:
            if total_size <= max_bytes:
                break
            try:
                os.remove(fp)
                total_size -= sz
                evicted += 1
            except OSError:
                pass
        if evicted:
            with _LLM_CACHE_LOCK:
                _LLM_CACHE_STATS["evictions"] += evicted
            print(f"[LLM][CACHE] evicted {evicted} entries to stay under {CONFIG.llm_cache_max_size_mb}MB")
    except Exception as exc:
        print(f"[LLM][CACHE] size enforcement error: {exc}")


def _log_llm_cache_summary(label: str) -> None:
    with _LLM_CACHE_LOCK:
        hits = _LLM_CACHE_STATS["hits"]
        misses = _LLM_CACHE_STATS["misses"]
        evictions = _LLM_CACHE_STATS["evictions"]
        errors = _LLM_CACHE_STATS["errors"]
        hit_pt = _LLM_CACHE_STATS["hit_prompt_tokens"]
        hit_ct = _LLM_CACHE_STATS["hit_completion_tokens"]
        miss_pt = _LLM_CACHE_STATS["miss_prompt_tokens"]
        miss_ct = _LLM_CACHE_STATS["miss_completion_tokens"]
        api_read = _LLM_CACHE_STATS["api_cache_read_tokens"]
        api_create = _LLM_CACHE_STATS["api_cache_create_tokens"]
    total = hits + misses
    if total <= 0:
        return
    hit_rate = hits / total * 100
    saved_total = hit_pt + hit_ct
    used_total = miss_pt + miss_ct
    print(
        f"[LLM][CACHE][SUMMARY] {label} "
        f"hits={hits} misses={misses} hit_rate={hit_rate:.1f}% | "
        f"disk_cache: create={used_total} read(saved)={saved_total} tokens | "
        f"api_cache: create={api_create} read={api_read} tokens"
    )
def _has_cache_control(body: dict) -> bool:
    for msg in body.get("messages", []):
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "cache_control" in block:
                    return True
    return False


def _openai_to_anthropic_body(body: dict) -> dict:
    system_blocks = []
    messages = []
    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            if isinstance(content, str):
                system_blocks.append({"type": "text", "text": content})
            elif isinstance(content, list):
                system_blocks.extend(content)
            continue
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            converted = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image_url":
                    img_url = block.get("image_url", {})
                    url_val = img_url.get("url", "") if isinstance(img_url, dict) else str(img_url)
                    if url_val.startswith("data:"):
                        parts = url_val.split(",", 1)
                        media_type = parts[0].replace("data:", "").replace(";base64", "")
                        data = parts[1] if len(parts) > 1 else ""
                        converted.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}})
                    else:
                        converted.append({"type": "image", "source": {"type": "url", "url": url_val}})
                else:
                    converted.append(block)
            content = converted
        messages.append({"role": role, "content": content})
    anthropic_body = {
        "model": body.get("model", ""),
        "max_tokens": body.get("max_tokens", 4096),
        "messages": messages,
    }
    if system_blocks:
        anthropic_body["system"] = system_blocks
    temp = body.get("temperature")
    if temp is not None:
        anthropic_body["temperature"] = temp
    rf = body.get("response_format")
    if rf and rf.get("type") == "json_schema":
        schema_obj = rf.get("json_schema", {})
        schema_str = json.dumps(schema_obj.get("schema", schema_obj), ensure_ascii=False, indent=2)
        schema_instruction = (
            "\n\n[IMPORTANT] You MUST respond with ONLY a valid JSON object (no markdown fences, no extra text) "
            "that strictly conforms to this JSON schema:\n```json\n" + schema_str + "\n```"
        )
        sys_blocks = anthropic_body.get("system", [])
        if sys_blocks:
            last = sys_blocks[-1]
            if isinstance(last, dict) and last.get("type") == "text":
                sys_blocks[-1] = {**last, "text": last["text"] + schema_instruction}
            else:
                sys_blocks.append({"type": "text", "text": schema_instruction})
        else:
            anthropic_body["system"] = [{"type": "text", "text": schema_instruction}]
    elif rf and rf.get("type") == "json_object":
        sys_blocks = anthropic_body.get("system", [])
        json_instruction = "\n\n[IMPORTANT] You MUST respond with ONLY a valid JSON object (no markdown fences, no extra text)."
        if sys_blocks:
            last = sys_blocks[-1]
            if isinstance(last, dict) and last.get("type") == "text":
                sys_blocks[-1] = {**last, "text": last["text"] + json_instruction}
            else:
                sys_blocks.append({"type": "text", "text": json_instruction})
        else:
            anthropic_body["system"] = [{"type": "text", "text": json_instruction}]
    return anthropic_body


def _anthropic_to_openai_response(anthropic_json: dict) -> dict:
    content_blocks = anthropic_json.get("content", [])
    text_parts = []
    for block in content_blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            text_parts.append(block.get("text", ""))
    text = "\n".join(text_parts)
    usage = anthropic_json.get("usage", {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    cache_creation = usage.get("cache_creation_input_tokens", 0)
    cache_read = usage.get("cache_read_input_tokens", 0)
    return {
        "id": anthropic_json.get("id", ""),
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "prompt_tokens_details": {"cached_tokens": cache_read},
            "cache_creation_input_tokens": cache_creation,
            "cache_read_input_tokens": cache_read,
        },
    }


def _apply_rolling_cache_breakpoint(body: dict) -> None:
    """Add cache_control to the last non-image block of the last user message.
    First clear stale dynamic breakpoints (preserving the static breakpoints on the system prompt and initial user content)."""
    messages = body.get("messages", [])
    # Find the last user message
    last_user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break
    if last_user_idx is None:
        return
    # Clear dynamic breakpoints on earlier user messages (keep static ones on indices 0, 1)
    for i, msg in enumerate(messages):
        if msg.get("role") != "user":
            continue
        if i == last_user_idx:
            continue
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "cache_control" in block:
                    if i > 1:
                        del block["cache_control"]
    # Add the breakpoint to the last non-image block of the last user message
    content = messages[last_user_idx].get("content")
    if isinstance(content, list):
        for block in reversed(content):
            if isinstance(block, dict) and block.get("type") != "image":
                block["cache_control"] = {"type": "ephemeral"}
                return
    elif isinstance(content, str) and content.strip():
        messages[last_user_idx]["content"] = [
            {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
        ]


def _post_llm_request(url, headers, body, timeout_secs, proxies, *, label: str) -> requests.Response:
    # --- External request preprocessors (replaces legacy monkey-patching) ---
    body = hooks.run_request_preprocessors(body, body.get("model", ""))
    # --- Enforce API key presence at the call boundary ---
    require_api_key()
    # --- Apply rolling cache breakpoint ---
    _apply_rolling_cache_breakpoint(body)

    # --- Cache lookup ---
    cache_key = None
    if CONFIG.llm_cache_enable:
        cache_key = _llm_cache_key(body)
        cached = _llm_cache_read(cache_key)
        if cached is not None:
            cached_usage = cached.get("usage", {})
            pt = cached_usage.get("prompt_tokens", 0)
            ct = cached_usage.get("completion_tokens", 0)
            with _LLM_CACHE_LOCK:
                _LLM_CACHE_STATS["hits"] += 1
                _LLM_CACHE_STATS["hit_prompt_tokens"] += pt
                _LLM_CACHE_STATS["hit_completion_tokens"] += ct
            if CONFIG.llm_log_timing:
                print(f"[LLM][CACHE][HIT] {label} key={cache_key[:12]}... saved_tokens={pt}+{ct}")
            resp = requests.models.Response()
            resp.status_code = 200
            resp._content = json.dumps(cached["response"], ensure_ascii=False).encode("utf-8")
            resp.headers["Content-Type"] = "application/json"
            resp.headers["X-LLM-Cache"] = "hit"
            return resp
        with _LLM_CACHE_LOCK:
            _LLM_CACHE_STATS["misses"] += 1

    # --- Determine if we should use Anthropic Messages API for caching ---
    _cache_policy_hit = hooks.should_apply_cache(body.get("model", ""))
    use_anthropic = _cache_policy_hit if _cache_policy_hit is not None else _has_cache_control(body)
    if use_anthropic:
        anthropic_body = _openai_to_anthropic_body(body)
        anthropic_body["anthropic_beta"] = ["prompt-caching-2024-07-31"]
        anthropic_url = url.replace("/chat/completions", "/messages")
        actual_url, actual_headers, actual_body = anthropic_url, headers, anthropic_body
    else:
        actual_url, actual_headers, actual_body = url, headers, body

    # --- Original request logic ---
    if not CONFIG.llm_log_timing:
        resp = requests.post(actual_url, headers=actual_headers, json=actual_body, timeout=timeout_secs, proxies=proxies)
    else:
        start_ts = _now_timestamp_ms()
        inflight = _inc_llm_inflight()
        worker = _worker_label()
        cache_tag = " (cache-miss)" if cache_key else ""
        print(f"[LLM][START] {start_ts} {label}{cache_tag} inflight={inflight} {worker}")
        resp = None
        err: Optional[Exception] = None
        start = time.time()
        try:
            resp = requests.post(actual_url, headers=actual_headers, json=actual_body, timeout=timeout_secs, proxies=proxies)
        except Exception as exc:
            err = exc
            raise
        finally:
            duration = time.time() - start
            ok = err is None and resp is not None and resp.status_code < 400
            _record_llm_metrics(duration, ok)
            inflight_after = _dec_llm_inflight()
            end_ts = _now_timestamp_ms()
            if err is not None:
                print(
                    f"[LLM][END] {end_ts} {label} inflight={inflight_after} "
                    f"duration={duration:.2f}s status=ERR error={type(err).__name__} {worker}"
                )
            elif resp is not None:
                print(
                    f"[LLM][END] {end_ts} {label} inflight={inflight_after} "
                    f"duration={duration:.2f}s status={resp.status_code} {worker}"
                )
            else:
                print(
                    f"[LLM][END] {end_ts} {label} inflight={inflight_after} "
                    f"duration={duration:.2f}s status=ERR error=resp_none {worker}"
                )

    # --- Extract API cache tokens & convert response if needed ---
    if resp is not None and resp.status_code < 400:
        try:
            resp_json = resp.json()
            usage = resp_json.get("usage", {})
            # Anthropic format fields
            cache_read = usage.get("cache_read_input_tokens", 0)
            cache_create = usage.get("cache_creation_input_tokens", 0)
            # OpenAI format fallback
            if not cache_read:
                cache_read = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
            if not cache_create:
                cache_create = usage.get("claude_cache_creation_5_m_tokens", 0)
            with _LLM_CACHE_LOCK:
                _LLM_CACHE_STATS["api_cache_read_tokens"] += cache_read
                _LLM_CACHE_STATS["api_cache_create_tokens"] += cache_create
            print(f"[LLM][API-CACHE] {label} cache_read={cache_read} cache_create={cache_create}")
        except Exception:
            pass
        if use_anthropic:
            try:
                openai_json = _anthropic_to_openai_response(resp.json())
                resp._content = json.dumps(openai_json, ensure_ascii=False).encode("utf-8")
            except Exception as exc:
                print(f"[LLM][WARN] Anthropic→OpenAI conversion failed: {exc}")

    # --- Cache store on success ---
    if cache_key and resp is not None and resp.status_code < 400:
        try:
            resp_json = resp.json()
            _llm_cache_write(cache_key, body, resp_json, label)
            usage = resp_json.get("usage", {})
            pt = usage.get("prompt_tokens", 0)
            ct = usage.get("completion_tokens", 0)
            with _LLM_CACHE_LOCK:
                _LLM_CACHE_STATS["miss_prompt_tokens"] += pt
                _LLM_CACHE_STATS["miss_completion_tokens"] += ct
        except Exception as exc:
            print(f"[LLM][CACHE] failed to cache response for {label}: {exc}")

    # --- External response observers (replaces legacy monkey-patching) ---
    if resp is not None and resp.status_code < 400:
        try:
            hooks.run_response_observers(body, resp.json(), body.get("model", ""))
        except Exception:
            pass

    return resp

# ==== Debug version: single request + raw-response print (HTTP, no response_format) ====
def _resolve_llm_config(
    endpoint=None,
    api_key=None,
    model=None,
    sub_ids=None,
    difficulty=None,
    style=None,
    max_tokens=None,
    temperature=None,
    timeout_secs=None,
):
    endpoint = endpoint or CONFIG.endpoint
    api_key = api_key or CONFIG.api_key
    model = model or CONFIG.model
    sub_ids = CONFIG.sub_ids if sub_ids is None else sub_ids
    difficulty = difficulty or CONFIG.difficulty
    style = style or CONFIG.style
    max_tokens = max_tokens if max_tokens is not None else CONFIG.max_tokens
    temperature = temperature if temperature is not None else CONFIG.temperature
    timeout_secs = timeout_secs if timeout_secs is not None else CONFIG.timeout_secs
    if not api_key:
        raise RuntimeError("API key not provided; please set the VERIGEO_API_KEY environment variable")
    return endpoint, api_key, model, sub_ids, difficulty, style, max_tokens, temperature, timeout_secs



def _author_combined_edit_and_generate(
    VeriGraph,
    *,
    endpoint,
    api_key,
    model,
    sub_ids,
    difficulty,
    style,
    max_tokens,
    temperature,
    timeout_secs,
    strict_validate: bool = True,
    attach_image: bool = True,
    from_scratch: bool = False,
    construction_targets: Optional[List[str]] = None,
    print_logs_immediately: bool = False,
    review_feedback: Optional[str] = None,
    previous_problem: Optional[Dict[str, Any]] = None,
    previous_actions: Optional[List[Dict[str, Any]]] = None,
    self_review_rounds: Optional[int] = None,
    generation_stats: Optional[Dict[str, Any]] = None,
):
    base_allowed_ids, raw_constraints, _ = _extract_allowed_and_constraints(VeriGraph, sub_ids=sub_ids)
    segment_catalog = _collect_segment_catalog(VeriGraph)
    clean_constraints = _filter_author_constraints(raw_constraints, segment_catalog)
    constraint_summary = _summarize_clean_constraints(clean_constraints)
    constraint_summary_text = _format_constraint_summary(constraint_summary)
    angle_catalog = _collect_angle_catalog(VeriGraph)
    try:
        ver = VeriGraph.verify_constraints()
        nl_text = NaturalLanguageDescriber.describe(VeriGraph, verify=ver, only_true=False)
    except Exception:
        nl_text = ""
    nl_text = _redact_nl_coordinates(nl_text)

    payload = {
        "allowed_ids": base_allowed_ids,
        "clean_constraints": clean_constraints,
        "segment_catalog": segment_catalog,
        "angle_catalog": angle_catalog,
        "constraint_summary": constraint_summary,
    }
    blueprint = None
    if from_scratch:
        payload["from_scratch"] = True
        payload["construction_targets"] = construction_targets or _DEFAULT_SCRATCH_CONSTRUCTION_TARGETS
        if CONFIG.blueprint_enable:
            blueprint = _sample_blueprint()
    difficulty_shots = ""
    required_topic_text = ""
    if from_scratch:
        required_topic = _pick_required_geometry_topic(blueprint)
        difficulty_shots = _format_difficulty_shots(difficulty)
        required_topic_text = (
            f"Required geometry topic(s): [{required_topic}]. "
            "The specified topic(s) only need to play a meaningful role in the problem; "
            "the overall design does not need to be centered on them."
        )

    base_user_content = []
    if required_topic_text:
        base_user_content.append({"type": "text", "text": required_topic_text})
    base_user_content += [
        {"type": "text", "text": "Author Payload:\n" + json.dumps(payload, ensure_ascii=False, indent=2)},
        {"type": "text", "text": "NL Description:\n" + (nl_text or "")},
    ]
    if constraint_summary_text:
        base_user_content.append({"type": "text", "text": "Constraint Summary (compressed):\n" + constraint_summary_text})
    if from_scratch:
        targets = construction_targets or _DEFAULT_SCRATCH_CONSTRUCTION_TARGETS
        base_user_content.append({"type": "text", "text": "Construction Targets:\n" + "\n".join(f"- {item}" for item in targets)})
    if previous_problem:
        try:
            base_user_content.append({"type": "text", "text": "Previous Problem (for revision):\n" + json.dumps(previous_problem, ensure_ascii=False, indent=2)})
        except Exception:
            pass
    if previous_actions:
        try:
            base_user_content.append({"type": "text", "text": "Previous Actions (for revision):\n" + json.dumps(previous_actions, ensure_ascii=False, indent=2)})
        except Exception:
            pass
    if review_feedback:
        base_user_content.append({"type": "text", "text": "Review Feedback (fix these issues):\n" + review_feedback})

    if attach_image:
        try:
            block = _build_prompt_image_block(VeriGraph, "author_combo")
            if block:
                base_user_content.append(block)
        except Exception:
            pass

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    proxies = {"http": None, "https": None}
    url = endpoint.rstrip("/") + "/v1/chat/completions"

    def _build_messages(prev_errors: List[str]):
        prompt = _AUTHOR_COMBINED_PROMPT_FROM_SCRATCH if from_scratch else _AUTHOR_COMBINED_PROMPT
        system_blocks = [
            {"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}},
        ]
        if difficulty_shots:
            system_blocks.append(
                {"type": "text", "text": difficulty_shots, "cache_control": {"type": "ephemeral"}},
            )
        stable_content = list(base_user_content)
        if stable_content:
            last = stable_content[-1]
            if isinstance(last, dict) and last.get("type") == "text":
                stable_content[-1] = {**last, "cache_control": {"type": "ephemeral"}}
        msgs = [
            {"role": "system", "content": system_blocks},
            {"role": "user", "content": stable_content},
        ]
        if prev_errors:
            critique = (
                "The previous combined JSON failed validation; please fix the following issues and retry:\n- "
                + "\n- ".join(prev_errors)
                + "\nOnly output JSON conforming to the schema."
            )
            msgs.append({"role": "user", "content": critique})
        if blueprint:
            bp_text = (
                "Blueprint Spec (must follow):\n"
                + json.dumps(blueprint, ensure_ascii=False, indent=2)
                + "\n\nBlueprint Spec (formatted):\n"
                + _format_blueprint_text(blueprint)
            )
            msgs.append({"role": "user", "content": bp_text})
        return msgs

    base_graph_snapshot = copy.deepcopy(VeriGraph)
    if self_review_rounds is None:
        self_review_rounds = 0
    try:
        self_review_rounds = max(0, int(self_review_rounds))
    except Exception:
        self_review_rounds = 0
    reflection_pass = max(0, int(CONFIG.author_reflection_rounds_pass or 0))
    reflection_fail = max(1, int(CONFIG.author_reflection_rounds_fail or 0))
    reflection_pass = max(reflection_pass, self_review_rounds)
    reflection_fail = max(reflection_fail, self_review_rounds)
    stats = _ensure_author_generation_stats(generation_stats)

    def _materialize_combo(content_text: str):
        raw_obj = None
        try:
            obj = _loads_lenient(content_text)
            raw_obj = obj
            actions_raw = obj.get("actions", [])
            problem_obj = obj.get("problem", {})
            comment = obj.get("comment", "")
            actions, _ = _parse_author_VeriScript_json({"actions": actions_raw, "comment": comment})
        except Exception as exc:
            raise ValueError(f"actions parsing failed: {exc}") from exc

        working = copy.deepcopy(base_graph_snapshot)
        logs: List[Dict[str, Any]] = []
        try:
            logs = _apply_author_VeriScript_actions(working, actions)
            working.normalize_polygons()
            if print_logs_immediately:
                _print_author_actions(logs)
        except Exception as exc:
            raise ValueError(f"Author-VeriScript execution failed: {exc}") from exc

        try:
            problem_obj = _coerce_to_schema(problem_obj)
            allowed_after, _, _ = _extract_allowed_and_constraints(working, sub_ids=sub_ids)
            if CONFIG.validate_problem_json:
                validated_problem = _validate_problem_json(
                    problem_obj,
                    allowed_after,
                    working,
                    strict=strict_validate,
                )
            else:
                validated_problem = problem_obj
        except Exception as exc:
            raise ValueError(f"Problem validation failed: {exc}") from exc

        return working, {
            "actions": [asdict(act) for act in actions],
            "comment": comment,
            "logs": logs,
            "problem": validated_problem,
        }

    def _format_author_reflection_feedback(numeric_ok: bool, issues: List[str]) -> str:
        base_prompt = CONFIG.author_reflection_prompt or _AUTHOR_SELF_REVIEW_GUIDE
        if numeric_ok:
            header = "Numeric / intersection check passed; please perform a logic and problem-statement consistency self-check, explain any modifications if needed, and return the complete JSON."
        else:
            header = "Numeric / intersection check did not pass; please regenerate actions/problem/comment based on the logs to ensure the figure is solvable and the problem statement is correct."
        lines = [header, base_prompt]
        if issues:
            lines.append("Error summary:")
            lines.extend(f"- {item}" for item in issues[:8])
        return "\n".join(lines)

    def _run_author_reflection_loop(base_messages: List[Dict[str, Any]], current_graph, current_combo):
        history = list(base_messages)
        graph_state = current_graph
        combo_state = current_combo
        numeric_ok, issues = _analyze_action_logs(combo_state.get("logs"))
        max_rounds = max(reflection_pass, reflection_fail)
        if numeric_ok and reflection_pass <= 0:
            return graph_state, combo_state
        if not numeric_ok:
            max_rounds = max(max_rounds, 1)
            _record_author_failed_step(
                stats,
                phase="author_combined_initial",
                issues=issues,
                actions=combo_state.get("actions"),
                logs=combo_state.get("logs"),
                comment=combo_state.get("comment"),
            )
        rounds_done = 0
        print(f"[AUTHOR] Entering new reflection chain: numeric_ok={numeric_ok}, pass_rounds={reflection_pass}, fail_rounds={reflection_fail}, issues={issues[:3]}")
        while rounds_done < max_rounds:
            required = reflection_pass if numeric_ok else max(1, reflection_fail)
            if numeric_ok and rounds_done >= required:
                break
            feedback_text = _format_author_reflection_feedback(numeric_ok, issues)
            review_blocks = _build_author_self_review_blocks(
                graph_state,
                combo_state,
                rounds_done + 1,
                required,
                attach_image=attach_image,
                feedback=feedback_text,
            )
            print(f"[AUTHOR] Reflection round {rounds_done + 1}/{required}, numeric_ok={numeric_ok}, issues summary={issues[:3]}")
            history.append({"role": "user", "content": review_blocks})
            body = {
                "model": model,
                "messages": history,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_schema", "json_schema": AUTHOR_COMBINED_JSON_SCHEMA},
            }
            resp = _post_llm_request(
                url,
                headers,
                body,
                timeout_secs,
                proxies,
                label="author_reflection",
            )
            if resp.status_code >= 400:
                err = RuntimeError(f"Author self-check request failed: HTTP {resp.status_code} {resp.text[:200]}")
                _note_author_reflection_round(stats)
                _record_author_failed_step(
                    stats,
                    phase="author_reflection_request",
                    error=err,
                    raw=resp.text,
                )
                raise err
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content") or ""
            if not content.strip():
                err = RuntimeError("Author self-check returned empty")
                _note_author_reflection_round(stats)
                _record_author_failed_step(
                    stats,
                    phase="author_reflection_empty",
                    error=err,
                    raw=data,
                )
                raise err
            history.append({"role": "assistant", "content": content})
            try:
                graph_state, combo_state = _materialize_combo(content)
            except Exception as exc:
                numeric_ok = False
                issues = [f"Parsing failed: {exc}"]
                rounds_done += 1
                _note_author_reflection_round(stats)
                _record_author_failed_step(
                    stats,
                    phase="author_reflection_parse",
                    issues=issues,
                    error=exc,
                    raw=content,
                )
                print(f"[AUTHOR][WARN] Self-check parsing failed; continuing to next round: {exc}")
                continue
            numeric_ok, issues = _analyze_action_logs(combo_state.get("logs"))
            rounds_done += 1
            _note_author_reflection_round(stats)
            if not numeric_ok:
                _record_author_failed_step(
                    stats,
                    phase="author_reflection",
                    issues=issues,
                    actions=combo_state.get("actions"),
                    logs=combo_state.get("logs"),
                    comment=combo_state.get("comment"),
                )
        if not numeric_ok:
            raise RuntimeError(f"Numeric / intersection issues still present after Author self-check: {'; '.join(issues[:3])}")
        return graph_state, combo_state

    prev_errors: List[str] = []
    last_exc: Optional[Exception] = None
    last_messages: Optional[List[Dict[str, Any]]] = None
    final_graph = None
    final_combo = None

    for _ in range(3):
        messages = _build_messages(prev_errors)
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_schema", "json_schema": AUTHOR_COMBINED_JSON_SCHEMA},
        }
        resp = _post_llm_request(
            url,
            headers,
            body,
            timeout_secs,
            proxies,
            label="author_combined",
        )
        if resp.status_code >= 400:
            last_exc = RuntimeError(f"Author-Combined request failed: HTTP {resp.status_code} {resp.text[:200]}")
            prev_errors = [str(last_exc)]
            _record_author_failed_step(
                stats,
                phase="author_combined_request",
                error=last_exc,
                raw=resp.text,
            )
            continue
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content") or ""
        if not content.strip():
            last_exc = RuntimeError("Author-Combined returned empty")
            prev_errors = [str(last_exc)]
            _record_author_failed_step(
                stats,
                phase="author_combined_empty",
                error=last_exc,
                raw=data,
            )
            continue

        try:
            working, combo = _materialize_combo(content)
        except Exception as exc:
            last_exc = exc
            prev_errors = [str(exc)]
            _record_author_failed_step(
                stats,
                phase="author_combined_invalid",
                error=exc,
                raw=content,
            )
            try:
                print("[DEBUG][AUTHOR-UNIFIED] response(raw str):", content[:800])
            except Exception:
                pass
            continue

        final_graph = working
        final_combo = combo
        last_messages = messages + [{"role": "assistant", "content": content}]
        break

    if final_graph is None or final_combo is None:
        if last_exc:
            raise last_exc
        raise RuntimeError("Author-Combined stage failed: multiple retries did not pass validation")

    if last_messages is not None:
        final_graph, final_combo = _run_author_reflection_loop(last_messages, final_graph, final_combo)

    _copy_graph_state(VeriGraph, final_graph)
    if blueprint:
        final_combo = dict(final_combo)
        final_combo["blueprint"] = blueprint
    return final_combo


def _build_author_review_followup(review_obj: Dict[str, Any], round_idx: int, max_rounds: int) -> str:
    status = review_obj.get("status", "needs_fix")
    issues = review_obj.get("issues") or []
    suggestions = review_obj.get("fix_suggestions") or []
    lines = [
        f"Round {round_idx} Review returned status={status}, indicating that figure-text-problem alignment / solvability is not yet fully consistent.",
        "Please continue based on the current payload to verify each item; once you have confirmed there are no conflicts or omissions, return JSON with status=\"ok\"."
    ]
    if issues:
        lines.append("Main issues you currently listed:")
        for issue in issues[:6]:
            lines.append(f"- {issue}")
    if suggestions:
        lines.append("Suggestion summary:")
        for suggestion in suggestions[:4]:
            focus = suggestion.get("focus", "other")
            detail = suggestion.get("detail", "")
            lines.append(f"- [{focus}] {detail}".strip())
    remain = max_rounds - round_idx
    if remain > 0:
        lines.append(f"{remain} round(s) remaining. If you are confident issues are resolved, please directly return status=\"ok\".")
    else:
        lines.append("Reached the maximum configured Review rounds; please provide a final conclusion. If repairs are still needed, clearly identify the most critical issue.")
    return "\n".join(lines)


def _apply_author_review_meta(problem_json: Optional[Dict[str, Any]], review_state: Optional[Dict[str, Any]]) -> Tuple[str, bool, int]:
    """[CODEX-CHANGE] Sync self-check results into problem.meta for later statistics."""
    if not isinstance(problem_json, dict):
        return "skipped", False, 0
    meta = problem_json.setdefault("meta", {})
    state = review_state or {}
    review_obj = state.get("review") if isinstance(state, dict) else None
    if isinstance(review_obj, dict):
        status = (state.get("status") or review_obj.get("status") or "unknown").lower()
        needs_regen = bool(review_obj.get("needs_regeneration"))
        issues_count = len(review_obj.get("issues") or [])
    else:
        status = str(state.get("status") or "skipped").lower()
        needs_regen = False
        issues_count = 0
    meta["review_status"] = status
    meta["review_needs_regeneration"] = needs_regen
    meta["review_issues_count"] = issues_count
    error_msg = state.get("error")
    if error_msg:
        meta["review_error"] = error_msg
    elif "review_error" in meta:
        meta.pop("review_error", None)
    return status, needs_regen, issues_count


def _new_author_generation_stats() -> Dict[str, Any]:
    return {
        "reflection_rounds": 0,
        "failed_attempts": 0,
        "failed_steps": [],
    }


def _ensure_author_generation_stats(stats: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if stats is None:
        return None
    if not isinstance(stats, dict):
        return None
    stats.setdefault("reflection_rounds", 0)
    stats.setdefault("failed_attempts", 0)
    stats.setdefault("failed_steps", [])
    if not isinstance(stats.get("failed_steps"), list):
        stats["failed_steps"] = []
    return stats


def _record_author_failed_step(
    stats: Optional[Dict[str, Any]],
    *,
    phase: str,
    issues: Optional[List[str]] = None,
    error: Optional[Exception] = None,
    actions: Optional[List[Dict[str, Any]]] = None,
    logs: Optional[List[Dict[str, Any]]] = None,
    comment: Optional[str] = None,
    raw: Optional[str] = None,
):
    if not stats:
        return
    entry: Dict[str, Any] = {"phase": phase}
    if issues:
        entry["issues"] = issues
    if error:
        entry["error"] = str(error)
    if actions is not None:
        entry["actions"] = actions
    if logs is not None:
        entry["logs"] = logs
    if comment:
        entry["comment"] = comment
    if raw:
        raw_text = str(raw)
        entry["raw"] = raw_text[:1200]
    stats["failed_steps"].append(entry)
    stats["failed_attempts"] = int(stats.get("failed_attempts", 0)) + 1


def _note_author_reflection_round(stats: Optional[Dict[str, Any]]):
    if not stats:
        return
    stats["reflection_rounds"] = int(stats.get("reflection_rounds", 0)) + 1


def _apply_author_generation_stats(
    problem_json: Optional[Dict[str, Any]],
    stats: Optional[Dict[str, Any]],
    *,
    include_failed_steps: bool = True,
):
    if not isinstance(problem_json, dict) or not stats:
        return
    meta = problem_json.setdefault("meta", {})
    payload = {
        "reflection_rounds": int(stats.get("reflection_rounds", 0)),
        "failed_attempts": int(stats.get("failed_attempts", 0)),
    }
    if include_failed_steps:
        payload["failed_steps"] = stats.get("failed_steps", [])
    meta["author_generation"] = payload


def _author_review_requires_retry(review_state: Optional[Dict[str, Any]]) -> bool:
    return False


def _describe_review_failure(review_state: Optional[Dict[str, Any]]) -> str:
    """[CODEX-CHANGE] Generate a readable self-check failure hint."""
    if not isinstance(review_state, dict):
        return "author review unavailable"
    review_obj = review_state.get("review") or {}
    if not isinstance(review_obj, dict):
        return str(review_state.get("error") or "author review unavailable")
    status = review_state.get("status") or review_obj.get("status") or "unknown"
    needs_regen = bool(review_obj.get("needs_regeneration"))
    issues = review_obj.get("issues") or []
    issue_preview = "; ".join(issues[:3])
    suffix = ""
    if issue_preview:
        suffix = f"; issues: {issue_preview}"
    if needs_regen:
        suffix += "; needs_regeneration=true"
    return f"status={status}{suffix}"

def _format_author_retry_feedback(review_state: Optional[Dict[str, Any]]) -> str:
    """Internal helper (docstring was corrupted in the source; restored as placeholder)."""
    if not isinstance(review_state, dict):
        return "review failed; please fix and regenerate the graph + problem."
    review_obj = review_state.get("review") or {}
    status = str(review_state.get("status") or review_obj.get("status") or "unknown")
    issues = review_obj.get("issues") or []
    suggestions = review_obj.get("fix_suggestions") or []
    repair_plan = review_obj.get("repair_plan")
    lines = [f"review status={status}; fix issues and regenerate actions + problem on the current graph."]
    if repair_plan:
        lines.append(f"review plan: {repair_plan}")
    if issues:
        lines.append("issues:")
        for item in issues[:8]:
            lines.append(f"- {item}")
    if suggestions:
        lines.append("fix_suggestions:")
        for sug in suggestions[:8]:
            if isinstance(sug, dict):
                focus = sug.get("focus", "other")
                detail = sug.get("detail", "")
                lines.append(f"- [{focus}] {detail}".strip())
            else:
                lines.append(f"- {sug}")
    return "\n".join(lines)
def _author_review_problem(
    VeriGraph,
    problem_json: Dict[str, Any],
    *,
    endpoint,
    api_key,
    model,
    sub_ids,
    difficulty,
    style,
    max_tokens,
    temperature,
    timeout_secs,
    author_actions: Optional[List[Dict[str, Any]]] = None,
    author_comment: Optional[str] = "",
    author_logs: Optional[List[Dict[str, Any]]] = None,
    attach_image: bool = True,
    max_rounds: int = 3,
):
    allowed_ids, clean_constraints, _ = _extract_allowed_and_constraints(VeriGraph, sub_ids=sub_ids)
    segment_catalog = _collect_segment_catalog(VeriGraph)
    angle_catalog = _collect_angle_catalog(VeriGraph)
    constraint_summary = _summarize_clean_constraints(clean_constraints)
    constraint_summary_text = _format_constraint_summary(constraint_summary)
    try:
        ver = VeriGraph.verify_constraints()
        nl_text = NaturalLanguageDescriber.describe(VeriGraph, verify=ver, only_true=False)
    except Exception:
        nl_text = ""
    nl_text = _redact_nl_coordinates(nl_text)

    review_payload = {
        "problem_json": problem_json,
        "graph": {
            "allowed_ids": allowed_ids,
            "clean_constraints": clean_constraints,
            "segment_catalog": segment_catalog,
            "angle_catalog": angle_catalog,
            "constraint_summary": constraint_summary,
            "sub_ids": sub_ids or [],
        },
        "author_stage": {
            "actions": author_actions or [],
            "comment": author_comment or "",
            "logs": author_logs or [],
        },
    }

    base_user_content = [
        {"type": "text", "text": "Review Payload:\n" + json.dumps(review_payload, ensure_ascii=False, indent=2)},
        {"type": "text", "text": "NL Description:\n" + (nl_text or "")},
    ]
    if constraint_summary_text:
        base_user_content.append({"type": "text", "text": "Constraint Summary (compressed):\n" + constraint_summary_text})
    if attach_image:
        try:
            block = _build_prompt_image_block(VeriGraph, "author_review")
        except Exception:
            block = None
        if block:
            base_user_content.append(block)

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    proxies = {"http": None, "https": None}
    url = endpoint.rstrip("/") + "/v1/chat/completions"

    messages = [
        {"role": "system", "content": _AUTHOR_REVIEW_PROMPT},
        {"role": "user", "content": list(base_user_content)},
    ]
    max_rounds = max(1, int(max_rounds or 1))
    last_obj: Optional[Dict[str, Any]] = None
    last_exc: Optional[Exception] = None

    for round_idx in range(1, max_rounds + 1):
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_schema", "json_schema": AUTHOR_REVIEW_JSON_SCHEMA},
        }
        resp = _post_llm_request(
            url,
            headers,
            body,
            timeout_secs,
            proxies,
            label="author_review",
        )
        if resp.status_code >= 400:
            last_exc = RuntimeError(f"Author-Review request failed: HTTP {resp.status_code} {resp.text[:200]}")
            break

        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content") or ""
        if not content.strip():
            last_exc = RuntimeError("Author-Review returned empty")
            messages.append({"role": "user", "content": "The previous round's output was empty; please return a JSON satisfying the schema."})
            continue

        messages.append({"role": "assistant", "content": content})
        try:
            obj = _loads_lenient(content)
        except Exception as exc:
            last_exc = exc
            messages.append({"role": "user", "content": f"JSON parsing failed: {exc}. Please output JSON strictly according to the schema, with no additional text."})
            continue

        last_obj = obj
        status = obj.get("status")
        if status == "ok":
            return obj

        follow_up = _build_author_review_followup(obj, round_idx, max_rounds)
        messages.append({"role": "user", "content": follow_up})

    if last_obj is not None:
        return last_obj
    if last_exc:
        raise last_exc
    raise RuntimeError("Author-Review stage failed: multi-turn conversation did not produce a valid response")
def _maybe_run_author_review(
    VeriGraph,
    problem_json: Dict[str, Any],
    *,
    endpoint,
    api_key,
    model,
    sub_ids,
    difficulty,
    style,
    max_tokens,
    temperature,
    timeout_secs,
    stage_outputs: Optional[Dict[str, Any]] = None,
    author_actions: Optional[List[Dict[str, Any]]] = None,
    author_logs: Optional[List[Dict[str, Any]]] = None,
    author_comment: Optional[str] = "",
    attach_image: bool = True,
):
    state = {"status": "skipped", "review": None, "error": None}
    if stage_outputs is not None:
        stage_outputs["author_review_state"] = state
    return state



def generate_problem_from_VeriGraph(
    VeriGraph,
    endpoint=None,
    api_key=None,
    model=None,
    sub_ids=None,
    difficulty=None,
    style=None,
    max_tokens=None,
    temperature=None,
    timeout_secs=None,
    strict_validate: bool = False,
    allow_author_edits: bool = True,
    preview_path: Optional[str] = None,
    stage_outputs: Optional[Dict[str, Any]] = None,
    from_scratch: Optional[bool] = None,
):
    """
    Main authoring entry point: only supports the unified Author-VeriScript (combined) workflow.
    When from_scratch=True, the input figure is ignored and VeriGraph is constructed from scratch.
    """
    endpoint, api_key, model, sub_ids, difficulty, style, max_tokens, temperature, timeout_secs = _resolve_llm_config(
        endpoint, api_key, model, sub_ids, difficulty, style, max_tokens, temperature, timeout_secs
    )
    if from_scratch is None:
        from_scratch = AUTHOR_FROM_SCRATCH
    if not allow_author_edits or AUTHOR_WORKFLOW == "off":
        raise ValueError("Currently only the unified Author-VeriScript workflow is supported")
    original_VeriGraph = VeriGraph
    if from_scratch:
        VeriGraph = GeometryGraph()
    scratch_targets = _DEFAULT_SCRATCH_CONSTRUCTION_TARGETS if from_scratch else None
    author_stats = _new_author_generation_stats()

    def _finalize_author_problem(prob):
        if not prob:
            return None
        target = preview_path
        try:
            if not target:
                preview_dir = os.path.join(os.getcwd(), "author_outputs")
                os.makedirs(preview_dir, exist_ok=True)
                target = os.path.join(preview_dir, "author_latest.png")
            else:
                os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
            VeriGraph.render(save_path=target, show=False)
            print(f"[AUTHOR] Preview image saved: {target}")
        except Exception as render_err:
            print(f"[WARN] Author preview image save failed: {render_err}")
        _print_author_problem(prob)
        if stage_outputs is not None:
            stage_outputs["problem"] = prob
            if target:
                stage_outputs["author_preview_path"] = target
        return prob

    review_regen_limit = 0
    block_on_review_fail = False

    use_author = True
    if use_author:
        last_combo_error = None
        regen_attempts = 0
        prev_review_feedback: Optional[str] = None
        prev_problem_for_retry: Optional[Dict[str, Any]] = None
        prev_actions_for_retry: Optional[List[Dict[str, Any]]] = None
        base_snapshot = copy.deepcopy(VeriGraph)
        while True:
            # regen = restart generation, starting from the original graph snapshot
            VeriGraph = copy.deepcopy(base_snapshot)
            try:
                combo = _author_combined_edit_and_generate(
                    VeriGraph,
                    endpoint=endpoint,
                    api_key=api_key,
                    model=model,
                    sub_ids=sub_ids,
                    difficulty=difficulty,
                    style=style,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout_secs=timeout_secs,
                    strict_validate=strict_validate,
                    from_scratch=from_scratch,
                    construction_targets=scratch_targets,
                    print_logs_immediately=True,
                    review_feedback=prev_review_feedback,
                    previous_problem=prev_problem_for_retry,
                    previous_actions=prev_actions_for_retry,
                    generation_stats=author_stats,
                )
            except Exception as exc:
                last_combo_error = exc
                break
            if not combo:
                break
            print("[INFO] Author unified pipeline succeeded")
            prob = combo.get("problem")
            combo_actions = combo.get("actions", [])
            combo_logs = combo.get("logs")
            combo_comment = combo.get("comment", "")
            finalized_prob = _finalize_author_problem(prob)
            if not finalized_prob:
                break
            review_state = _maybe_run_author_review(
                VeriGraph,
                finalized_prob,
                endpoint=endpoint,
                api_key=api_key,
                model=model,
                sub_ids=sub_ids,
                difficulty=difficulty,
                style=style,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout_secs=timeout_secs,
                stage_outputs=stage_outputs,
                author_actions=combo_actions,
                author_logs=combo_logs,
                author_comment=combo_comment,
                attach_image=True,
            )
            _apply_author_review_meta(finalized_prob, review_state)
            requires_retry = _author_review_requires_retry(review_state)
            if not requires_retry:
                _apply_author_generation_stats(finalized_prob, author_stats, include_failed_steps=True)
                stage_meta = {
                    "VeriScript": combo_actions,
                    "strategy": "unified",
                    "from_scratch": from_scratch,
                }
                problem_type_meta = (finalized_prob.get("meta") or {}).get("problem_type")
                if problem_type_meta:
                    stage_meta["problem_type"] = problem_type_meta
                VeriGraph.meta.setdefault("author_stage", {}).update(stage_meta)
                if stage_outputs is not None:
                    stage_outputs["author"] = {
                        "strategy": "unified",
                        "combo": combo,
                        "from_scratch": from_scratch,
                    }
                    if isinstance(combo, dict):
                        blueprint_info = combo.get("blueprint")
                        if blueprint_info:
                            stage_outputs["blueprint"] = blueprint_info
                    if problem_type_meta:
                        stage_outputs["author"]["mode"] = problem_type_meta
                    stage_outputs["author_stats"] = author_stats
                if from_scratch:
                    _copy_graph_state(original_VeriGraph, VeriGraph)
                return finalized_prob
            regen_attempts += 1
            reason = _describe_review_failure(review_state)
            if regen_attempts <= review_regen_limit:
                prev_review_feedback = _format_author_retry_feedback(review_state)
                prev_problem_for_retry = finalized_prob
                prev_actions_for_retry = combo_actions
                print(f"[REVIEW] Rejection {regen_attempts}/{review_regen_limit}: {reason}; restarting a fresh Author run (carrying Review feedback)")
                continue
            if block_on_review_fail or review_regen_limit > 0:
                raise RuntimeError(f"Author review rejected unified problem: {reason}")
            print(f"[WARN] Author review did not pass but not blocking: {reason} -> using current problem")
            _apply_author_generation_stats(finalized_prob, author_stats, include_failed_steps=True)
            stage_meta = {
                "VeriScript": combo_actions,
                "strategy": "unified",
                "from_scratch": from_scratch,
            }
            problem_type_meta = (finalized_prob.get("meta") or {}).get("problem_type")
            if problem_type_meta:
                stage_meta["problem_type"] = problem_type_meta
            VeriGraph.meta.setdefault("author_stage", {}).update(stage_meta)
            if stage_outputs is not None:
                stage_outputs["author"] = {
                    "strategy": "unified",
                    "combo": combo,
                    "from_scratch": from_scratch,
                }
                if isinstance(combo, dict):
                    blueprint_info = combo.get("blueprint")
                    if blueprint_info:
                        stage_outputs["blueprint"] = blueprint_info
                if problem_type_meta:
                    stage_outputs["author"]["mode"] = problem_type_meta
                stage_outputs["author_stats"] = author_stats
            if from_scratch:
                _copy_graph_state(original_VeriGraph, VeriGraph)
            return finalized_prob
        if last_combo_error:
            raise last_combo_error
        raise RuntimeError("Author unified pipeline did not produce a problem")

def generate_solver_proof_narration(
    problem_json: Dict[str, Any],
    solver_output: Dict[str, Any],
    nl_description: str,
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    timeout_secs: Optional[int] = None,
) -> str:
    """
    Call the LLM to convert VeriScript actions + post-solve graph facts into an English natural-language proof.
    """
    endpoint, api_key, model, _, difficulty, style, cfg_max_tokens, cfg_temperature, cfg_timeout = _resolve_llm_config(
        endpoint, api_key, model, None, None, None, max_tokens, temperature, timeout_secs
    )
    # Fill missing values with resolved defaults so timeout is never None (requests would block indefinitely)
    max_tokens = max_tokens if max_tokens is not None else cfg_max_tokens
    temperature = temperature if temperature is not None else cfg_temperature
    timeout_secs = timeout_secs if timeout_secs is not None else cfg_timeout
    if not api_key:
        raise RuntimeError("generate_solver_proof_narration: missing API Key")

    payload = {
        "problem": problem_json or {},
        "solver_actions": solver_output or {},
        "nl_description": nl_description or "",
        "instructions": (
            "Based on the problem statement, reasoning steps, and post-solve graph facts, write a logically complete and well-organized English geometry proof. "
            "The proof must contain the following elements: "
            "(1) Cite and explain key given conditions, indicating their role in the figure; "
            "(2) For each auxiliary construction or geometric relation, explain its logical purpose and significance in the reasoning; "
            "(3) Develop the chain of reasoning in logical order, with each step stating its basis and result; "
            "(4) Finally clearly summarize how the prior reasoning yields the target conclusion. "
            "The writing style should be pedagogical and explanatory of the logic; use natural-language expression; do not output JSON, code, or symbolic structures. "
            "A numbered structure is recommended, e.g.: "
            "(1) Key givens and analysis; (2) Auxiliary constructions and their role; (3) Reasoning process; (4) Conclusion."
        ),
    }

    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                #"content": (
                    #"You are a geometry teacher. Please convert the given problem, the VeriScript actions obtained by the LLM solver, and the final graph facts "
                    #"into a natural-language proof, in English, with clear logic."


                #),
                "content": (
                    "You are a geometry-reasoning expert writing data samples for training a geometry LLM. "
                    "Your goal is to produce logically complete, pedagogically strong natural-language geometry proofs so the model can learn a clear reasoning chain. "
                    "Based on the input problem statement (problem), givens, reasoning steps (proof_outline), and post-solve graph facts (graph facts), write a complete English geometry proof. "
                    "Warning: the output must be a natural-language narrative; it must not contain JSON, code, or formulaic symbolic descriptions."

                    "### Writing rules:"
                    "1. Clear structure, hierarchically numbered."
                    "   Organize your reasoning with the following hierarchy:"
                    "   - (1) Key givens and figure analysis"
                    "   - (2) Auxiliary constructions and their purpose"
                    "   - (3) Stepwise reasoning chain"
                    "   - (4) Summary and target conclusion"

                    "2. The reasoning chain should be made explicit."
                    "   For each step, state:"
                    "   - which geometric property or theorem was used (e.g., \"angle-bisector property\", \"isosceles-triangle criterion\");"
                    "   - what intermediate conclusion was derived;"
                    "   - how that conclusion helps approach the final goal."

                    "3. Auxiliary constructions must have their motivation explained."
                    "   Explain \"why construct this element\", e.g.: \"Draw the altitude BD in order to prove AC ⟂ BD\"."

                    "4. Pedagogical style."
                    "   The model should learn geometric reasoning thinking from reading this proof, not merely imitate the textual form."
                    "   Try to use explicit logical connectives:"
                    "   because...therefore...; from...we know...; further...; thereby obtaining...; combining the previous results we can deduce..."

                    "5. Ensure goal alignment."
                    "   The last section must separately point out the target conclusion and explicitly state \"from which key facts it is obtained\"."

                    "---"
                    "Output format template (natural language; do not output the headings):"
                    "(1) Key givens and analysis:"
                    "  Given... (list and explain the logical meaning)"
                    "(2) Auxiliary constructions and their purpose:"
                    "  In the figure draw..., the purpose of this construction is to..."
                    "(3) Reasoning process:"
                    "  (i) From... we know..."
                    "  (ii) And because... therefore..."
                    "  (iii) From (i)(ii) we obtain..."
                    "(4) Conclusion:"
                    "  Combining the above, we obtain..., which is what was to be proved."

                    "---"
                    "Quality criteria:"
                    "- The reasoning chain must contain at least 4-6 steps;"
                    "- Each step should correspond to a geometric theorem or logical inference;"
                    "- Natural language, grammatical, logically closed;"
                    "- Avoid vague expressions (e.g., \"it can be derived\", \"obviously\");"
                    "- As much as possible, exhibit causal relations between geometric theorems and the constraints."

                    "---"
                    "Enhancement options:"
                    "1. When narrating each step, implicitly reflect the figure's changes and logical progression without referencing any symbolic tags."
                    "2. When using geometric concepts (e.g., \"perpendicular\", \"parallel\", \"isosceles\"), attach a brief definitional explanation on first occurrence to help the model form stable conceptual knowledge."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "text"},
    }

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    proxies = {"http": None, "https": None}
    url = endpoint.rstrip("/") + "/v1/chat/completions"

    resp = _post_llm_request(
        url,
        headers,
        body,
        timeout_secs,
        proxies,
        label="teacher_narration",
    )
    resp.raise_for_status()
    data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return (content or "").strip()


def solve_problem_with_llm(
    VeriGraph,
    problem_json: Dict[str, Any],
    endpoint=None, api_key=None, model=None,
    sub_ids=None, max_tokens=None, temperature=None, timeout_secs=None,
    max_retries: int = 2
) -> Dict[str, Any]:
    """
    Strict VeriScript generation: strong schema, strong validation, error-driven retry.
    Returns {"actions": [...], "final_answer": "..."}.
    """
    if problem_json is None:
        raise ValueError("Problem (problem_json) must not be empty; please successfully generate a problem first")
    """
    Strict VeriScript generation: strong schema, strong validation, error-driven retry.
    Returns {"actions": [...], "final_answer": "..."}.
    """
    import re, copy
    from typing import List

    # Fall back to CONFIG defaults
    endpoint     = endpoint     or CONFIG.endpoint
    api_key      = api_key      or CONFIG.api_key
    model        = model        or CONFIG.model
    sub_ids      = CONFIG.sub_ids if sub_ids is None else sub_ids
    max_tokens   = max_tokens if max_tokens is not None else CONFIG.max_tokens
    temperature  = temperature if temperature is not None else CONFIG.temperature  # solver: stay conservative
    timeout_secs = timeout_secs if timeout_secs is not None else CONFIG.timeout_secs

    if not api_key:
        raise RuntimeError("API key not provided. Set the VERIGEO_API_KEY environment variable.")

    # Disable any local HTTP/HTTPS proxy env that might intercept the API call.
    os.environ.pop("HTTP_PROXY", None); os.environ.pop("HTTPS_PROXY", None)

    # Extract usable entities + graph-level facts
    allowed_ids, clean_constraints, alias_map = _extract_allowed_and_constraints(VeriGraph, sub_ids=sub_ids)
    angle_catalog = _collect_angle_catalog(VeriGraph)

    # Pull problem statement + goal from problem_json
    if not problem_json:
        raise ValueError("problem_json must not be None")
    if not isinstance(problem_json, dict):
        raise ValueError(f"problem_json must be a dict type, not {type(problem_json)}")
    prob_text = str(problem_json.get("problem") or "").strip()
    goal_text = str(problem_json.get("goal") or "").strip()
    if not goal_text:
        raise ValueError("problem_json must contain goal.")
    if not prob_text:
        prob_text = goal_text
    # Build solver payload (includes constraint-type enumeration to help the model avoid mistakes)
    user_payload = {
        **_build_solver_payload(allowed_ids, clean_constraints, prob_text, goal_text, angle_catalog),
        "constraint_types": SOLVER_CONSTRAINT_TYPES
    }

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    proxies = {"http": None, "https": None}
    url = f"{endpoint.rstrip('/')}/v1/chat/completions"

    # Local strong validation (stricter schema: ID set, args length, etc.)
    def _validate_VeriScript(obj: Dict[str, Any], alias_map: Optional[Dict[str, str]] = None) -> List[str]:
        alias_map = alias_map or {}
        errs = []
        if not isinstance(obj, dict):
            return ["top-level is not an object"]
        actions = obj.get("actions")
        if not isinstance(actions, list):
            errs.append('\"actions\" must be a list')
        elif not actions:
            errs.append('\"actions\" list must not be empty')
        if "final_answer" not in obj or not isinstance(obj["final_answer"], str) or not obj["final_answer"].strip():
            errs.append('\"final_answer\" must be a non-empty string')
        if "proof" in obj:
            if not isinstance(obj["proof"], str) or not obj["proof"].strip():
                errs.append('\"proof\" if provided must be a non-empty string')
        allowed_dynamic = set(allowed_ids)
        triangle_symbol = "△"
        angle_symbol = "∠"

        def _normalize_special_token(token: str) -> str:
            tok = token.strip()
            raw = tok.upper()
            if tok.startswith(triangle_symbol):
                tok = "TRI_" + tok[1:].upper()
            elif raw.startswith("TRI_"):
                suffix = tok.split("_", 1)[1]
                tok = "TRI_" + suffix.upper()
            elif tok.startswith(angle_symbol):
                tok = angle_symbol + tok[1:].upper()
            elif raw.startswith("ANG_"):
                suffix = tok.split("_", 1)[1]
                tok = "ANG_" + suffix.upper()
            elif raw.startswith("SEG_") and len(raw) == 6:
                body = raw[4:]
                if len(body) == 2 and body.isalpha():
                    a, b = body[0], body[1]
                    canonical = f"SEG_{a}{b}" if a < b else f"SEG_{b}{a}"
                    if canonical in allowed_ids or canonical == raw:
                        tok = canonical
                    else:
                        tok = raw
            else:
                tok = raw
            tok = alias_map.get(tok, tok)
            return tok

        def _maybe_allow(token: str):
            if token.startswith((angle_symbol, "ANG_", "TRI_")):
                allowed_dynamic.add(token)
            if token in alias_map:
                allowed_dynamic.add(alias_map[token])

        for i, act in enumerate(actions or []):
            if not isinstance(act, dict):
                errs.append(f"actions[{i}] is not an object")
                continue
            op = act.get("op")
            typ = act.get("type")
            args = act.get("args")
            if op not in VeriScript_ALLOWED_OPS:
                errs.append(f"actions[{i}].op invalid: {op}")
                continue
            if not isinstance(args, list):
                errs.append(f"actions[{i}].args must be an array")
                continue

            norm_args: List[str] = []
            invalid = False
            for tok in args:
                if isinstance(tok, str):
                    norm_tok = _normalize_special_token(tok)
                    _maybe_allow(norm_tok)
                    norm_args.append(norm_tok)
                elif isinstance(tok, (int, float)):
                    norm_args.append(str(tok))
                else:
                    errs.append(f"actions[{i}].args only support strings or numbers")
                    invalid = True
                    break
            if invalid:
                continue

            if op == "AddEdge":
                if typ not in SOLVER_CONSTRAINT_TYPES:
                    errs.append(f"actions[{i}].type invalid: {typ}")
                if not norm_args:
                    errs.append(f"actions[{i}] AddEdge requires at least 1 argument")
                    continue
                if typ == "ExprConstraint":
                    if len(norm_args) < 3:
                        errs.append(f"actions[{i}] ExprConstraint requires at least 3 arguments")
                    continue
                if typ == "VectorLinear":
                    if len(norm_args) < 5 or (len(norm_args) - 2) % 3 != 0:
                        errs.append(f"actions[{i}] VectorLinear argument count incorrect")
                        continue
                    u_tok = alias_map.get(norm_args[0], norm_args[0])
                    v_tok = alias_map.get(norm_args[1], norm_args[1])
                    if u_tok not in allowed_dynamic:
                        errs.append(f"actions[{i}] VectorLinear target point {u_tok} not in allowed_ids")
                    if v_tok not in allowed_dynamic:
                        errs.append(f"actions[{i}] VectorLinear target point {v_tok} not in allowed_ids")
                    for idx in range(2, len(norm_args), 3):
                        try:
                            _author_parse_float(norm_args[idx], "coeff")
                        except Exception:
                            errs.append(f"actions[{i}] VectorLinear coefficient must be numeric")
                        p_tok = alias_map.get(norm_args[idx + 1], norm_args[idx + 1])
                        q_tok = alias_map.get(norm_args[idx + 2], norm_args[idx + 2])
                        if p_tok not in allowed_dynamic:
                            errs.append(f"actions[{i}] VectorLinear point {p_tok} not in allowed_ids")
                        if q_tok not in allowed_dynamic:
                            errs.append(f"actions[{i}] VectorLinear point {q_tok} not in allowed_ids")
                    continue
                for tok in norm_args:
                    resolved = alias_map.get(tok, tok)
                    if resolved not in allowed_dynamic:
                        errs.append(f"actions[{i}].args contains invalid ID: {tok}")
            elif op == "AddPoint":
                if typ not in VeriScript_POINT_TYPES:
                    errs.append(f"actions[{i}].type invalid: {typ}")
                if not norm_args:
                    errs.append(f"actions[{i}] AddPoint missing new point ID")
                    continue
                pid = norm_args[0]
                if pid in allowed_dynamic:
                    errs.append(f"actions[{i}] new point {pid} already exists")
                if typ == "OnSegment":
                    if len(norm_args) < 2:
                        errs.append(f"actions[{i}] AddPoint-OnSegment missing segment ID")
                    else:
                        seg = alias_map.get(norm_args[1], norm_args[1])
                        if seg not in allowed_dynamic:
                            errs.append(f"actions[{i}] segment {seg} not in allowed_ids")
                if typ in ("OnCircle", "PointOnCircle"):
                    if len(norm_args) < 2:
                        errs.append(f"actions[{i}] AddPoint-OnCircle missing circle ID")
                    else:
                        circle_id = alias_map.get(norm_args[1], norm_args[1])
                        if circle_id not in allowed_dynamic:
                            errs.append(f"actions[{i}] circle {circle_id} not in allowed_ids")
                allowed_dynamic.add(pid)
            elif op == "AddAuxLine":
                if typ not in VeriScript_AUX_LINE_TYPES:
                    errs.append(f"actions[{i}].type invalid: {typ}")
                if len(norm_args) < 2:
                    errs.append(f"actions[{i}] AddAuxLine must provide 2 points")
                    continue
                p = alias_map.get(norm_args[0], norm_args[0])
                q = alias_map.get(norm_args[1], norm_args[1])
                for tok in (p, q):
                    if tok not in allowed_dynamic:
                        errs.append(f"actions[{i}] point {tok} not in allowed_ids")
                try:
                    seg_id = _canonical_segment_id(p, q)
                    allowed_dynamic.add(seg_id)
                except ValueError:
                    errs.append(f"actions[{i}] auxiliary-line endpoints must be distinct")
            elif op == "VerifyPoint":
                if typ not in VeriScript_VERIFY_POINT_TYPES:
                    errs.append(f"actions[{i}].type invalid: {typ}")
                if len(norm_args) < 3:
                    errs.append(f"actions[{i}] VerifyPoint must provide point ID and coordinates")
                    continue
                pid = alias_map.get(norm_args[0], norm_args[0])
                if pid not in allowed_dynamic:
                    errs.append(f"actions[{i}] point {pid} not in allowed_ids")
                for idx in (1, 2):
                    try:
                        _author_parse_float(norm_args[idx], "coord")
                    except Exception:
                        errs.append(f"actions[{i}] VerifyPoint coordinates must be numeric")
                        break
            elif op == "VerifyFunction":
                if typ not in VeriScript_VERIFY_FUNCTION_TYPES:
                    errs.append(f"actions[{i}].type invalid: {typ}")
                if typ == "PointsOnFunction":
                    if len(norm_args) < 2:
                        errs.append(f"actions[{i}] VerifyFunction-PointsOnFunction must provide function and point")
                        continue
                    tokens = norm_args[1:]
                    if len(tokens) >= 2:
                        try:
                            _author_parse_float(tokens[-1], "tol")
                            if len(tokens) % 2 == 1:
                                tokens = tokens[:-1]
                        except Exception:
                            pass
                    idx = 0
                    while idx < len(tokens):
                        tok = tokens[idx]
                        resolved = alias_map.get(tok, tok)
                        if resolved in allowed_dynamic:
                            idx += 1
                            continue
                        if idx + 1 >= len(tokens):
                            errs.append(f"actions[{i}] VerifyFunction-PointsOnFunction requires coordinate pairs")
                            break
                        try:
                            _author_parse_float(tokens[idx], "x")
                            _author_parse_float(tokens[idx + 1], "y")
                        except Exception:
                            errs.append(f"actions[{i}] VerifyFunction-PointsOnFunction coordinates must be numeric")
                            break
                        idx += 2
                elif typ == "DerivativeAt":
                    if len(norm_args) < 3:
                        errs.append(f"actions[{i}] VerifyFunction-DerivativeAt must provide function, x, and derivative value")
                        continue
                    x_tok = alias_map.get(norm_args[1], norm_args[1])
                    if x_tok not in allowed_dynamic:
                        try:
                            _author_parse_float(x_tok, "x")
                        except Exception:
                            errs.append(f"actions[{i}] VerifyFunction-DerivativeAt x is invalid")
                    try:
                        _author_parse_float(norm_args[2], "derivative")
                    except Exception:
                        errs.append(f"actions[{i}] VerifyFunction-DerivativeAt derivative value is invalid")
                    if len(norm_args) >= 5:
                        try:
                            _parse_derivative_order(norm_args[3])
                            _author_parse_float(norm_args[4], "tol")
                        except Exception:
                            errs.append(f"actions[{i}] VerifyFunction-DerivativeAt order/tolerance invalid")
                    elif len(norm_args) == 4:
                        tok = str(norm_args[3]).strip().lower()
                        if tok.startswith("n="):
                            try:
                                _parse_derivative_order(tok)
                            except Exception:
                                errs.append(f"actions[{i}] VerifyFunction-DerivativeAt order invalid")
                        else:
                            try:
                                _author_parse_float(norm_args[3], "tol")
                            except Exception:
                                errs.append(f"actions[{i}] VerifyFunction-DerivativeAt tolerance invalid")
                elif typ == "IntegralOn":
                    if len(norm_args) < 4:
                        errs.append(f"actions[{i}] VerifyFunction-IntegralOn must provide function, interval, and integral value")
                        continue
                    def _is_inf(tok: str) -> bool:
                        t = str(tok).strip().lower()
                        return t in _INF_TOKENS_POS or t in _INF_TOKENS_NEG
                    for idx in (1, 2):
                        tok = alias_map.get(norm_args[idx], norm_args[idx])
                        if tok not in allowed_dynamic:
                            if not _is_inf(tok):
                                try:
                                    _author_parse_float(tok, "bound")
                                except Exception:
                                    errs.append(f"actions[{i}] VerifyFunction-IntegralOn interval endpoint invalid")
                    try:
                        _author_parse_float(norm_args[3], "integral")
                    except Exception:
                        errs.append(f"actions[{i}] VerifyFunction-IntegralOn integral value invalid")
                    rest = norm_args[4:]
                    if rest:
                        head = str(rest[0]).strip().lower()
                        if head in ("breaks", "breakpoints"):
                            rest = rest[1:]
                        else:
                            try:
                                _author_parse_float(rest[0], "tol")
                                rest = rest[1:]
                            except Exception:
                                pass
                            if rest and str(rest[0]).strip().lower() in ("breaks", "breakpoints"):
                                rest = rest[1:]
                        for tok in rest:
                            resolved = alias_map.get(tok, tok)
                            if resolved in allowed_dynamic:
                                continue
                            if not _is_inf(tok):
                                try:
                                    _author_parse_float(tok, "break")
                                except Exception:
                                    errs.append(f"actions[{i}] VerifyFunction-IntegralOn breakpoint invalid")
        return errs


    pass_rounds = max(0, int(CONFIG.solver_reflection_rounds_pass or 0))
    fail_rounds = max(1, int(CONFIG.solver_reflection_rounds_fail or 0))
    max_rounds_cap = max(pass_rounds, fail_rounds)
    # Bind attempt count to config: initial attempt + configured reflection rounds + reserved max_retries for parse failures
    max_attempts = 1 + pass_rounds + fail_rounds + max_retries
    history = [
        {"role": "system", "content": [
            {"type": "text", "text": _SYSTEM_PROMPT_SOLVER, "cache_control": {"type": "ephemeral"}},
        ]},
        {"role": "user", "content": [
            {"type": "text", "text": json.dumps(user_payload, ensure_ascii=False), "cache_control": {"type": "ephemeral"}},
        ]},
    ]
    use_schema = True
    reflection_loops = 0
    latest_logs: Optional[List[Dict[str, Any]]] = None
    latest_issues: List[str] = []

    def _format_solver_reflection_feedback(numeric_ok: bool, issues: List[str], logs: Optional[List[Dict[str, Any]]],
                                           round_idx: int, required: int) -> str:
        base_prompt = CONFIG.solver_reflection_prompt or ""
        if numeric_ok:
            header = "Numeric / intersection passed; this round only checks the logic chain and problem-statement consistency, but a complete JSON must still be returned."
        else:
            header = "VeriScript execution or numeric / intersection did not pass; please regenerate complete proof/actions/final_answer based on the logs."
        lines = [header]
        if base_prompt:
            lines.append(base_prompt)
        if issues:
            lines.append("Error summary:")
            lines.extend(f"- {item}" for item in issues[:10])
        if logs:
            try:
                lines.append("VeriScript log excerpt:\n" + json.dumps(logs[:8], ensure_ascii=False, indent=2))
            except Exception:
                pass
        lines.append(f"(Reflection progress: round {round_idx}/{max(required, round_idx)}) Please directly output the revised complete JSON.")
        return "\n".join(lines)

    attempt = 0
    attempt_records: List[Dict[str, Any]] = []
    numeric_failed_once = False
    logic_guard_active = False
    while attempt < max_attempts:
        body = {
            "model": model,
            "messages": history,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_schema", "json_schema": VeriScript_JSON_SCHEMA} if use_schema else {"type": "json_object"},
        }
        resp = _post_llm_request(url, headers, body, timeout_secs, proxies, label="solver")
        attempt += 1
        if resp.status_code >= 400:
            raise RuntimeError(f"Solver request failed: HTTP {resp.status_code} {resp.text[:200]}")

        raw = resp.text or ""
        try:
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content") or raw
        except Exception:
            content = raw

        if content is None:
            content = ""
        history.append({"role": "assistant", "content": content})

        try:
            obj = _loads_lenient(content)
        except Exception:
            obj = None

        errs = _validate_VeriScript(obj if isinstance(obj, dict) else {}, alias_map)
        if errs:
            latest_issues = errs
            attempt_records.append({
                "status": "invalid_schema",
                "errors": errs,
                "raw": content if isinstance(content, str) else str(content),
            })
            if logic_guard_active:
                raise RuntimeError("Logic-check failure: reflection-round output still did not pass schema validation")
            feedback = _format_solver_reflection_feedback(False, errs, latest_logs, reflection_loops + 1, fail_rounds)
            history.append({"role": "user", "content": feedback})
            if use_schema and attempt >= 2:
                use_schema = False
            continue

        if not isinstance(obj, dict):
            latest_issues = ["Reply could not be parsed as JSON."]
            feedback = _format_solver_reflection_feedback(False, latest_issues, latest_logs, reflection_loops + 1, fail_rounds)
            history.append({"role": "user", "content": feedback})
            if use_schema and attempt >= 2:
                use_schema = False
            continue

        try:
            actions = [VeriScriptAction(**a) for a in obj.get("actions", [])]
        except Exception as exc:
            latest_issues = [f"actions parsing failed: {exc}"]
            attempt_records.append({
                "status": "invalid_actions",
                "errors": latest_issues,
                "raw_obj": obj,
            })
            if logic_guard_active:
                raise RuntimeError(f"Logic-check failure: VeriScript parsing exception {exc}")
            feedback = _format_solver_reflection_feedback(False, latest_issues, latest_logs, reflection_loops + 1, fail_rounds)
            history.append({"role": "user", "content": feedback})
            if use_schema and attempt >= 2:
                use_schema = False
            continue

        sim_graph = copy.deepcopy(VeriGraph)
        latest_logs = _apply_VeriScript_on_VeriGraph(sim_graph, actions, verify_only=False)
        numeric_ok, issues = _analyze_action_logs(latest_logs)
        latest_issues = issues
        required_rounds = pass_rounds if numeric_ok else fail_rounds
        attempt_records.append({
            "status": "ok" if numeric_ok else "numeric_fail",
            "issues": issues,
            "obj": obj,
            "logs": latest_logs,
        })

        if not numeric_ok:
            if reflection_loops >= fail_rounds:
                err_msg = "; ".join((issues or latest_issues or ["Numeric / intersection not passed"])[:5])
                raise RuntimeError(f"VeriScript generation failed: numeric / intersection did not pass for {fail_rounds} consecutive rounds; last error: {err_msg}")
            numeric_failed_once = True
            if logic_guard_active:
                raise RuntimeError("Logic-check failure: numeric regression")
            print(f"[SOLVER] Numeric / intersection did not pass; triggering reflection round {reflection_loops + 1}/{required_rounds} regeneration")
            try:
                print(f"[SOLVER][DEBUG] Previous VeriScript (truncated): {json.dumps(obj, ensure_ascii=False)[:800]}")
            except Exception:
                pass
        else:
            if logic_guard_active:
                result = dict(obj)
                if latest_logs is not None:
                    result.setdefault("execution_logs", latest_logs)
                if attempt_records:
                    result["_attempt_records"] = attempt_records
                print("[SOLVER] Logic check passed; ending reflection")
                return result
            if numeric_failed_once:
                logic_guard_active = True
                print("[SOLVER] Numerics repaired; entering extra logic-check round")
            elif reflection_loops >= required_rounds:
                result = dict(obj)
                if latest_logs is not None:
                    result.setdefault("execution_logs", latest_logs)
                if attempt_records:
                    result["_attempt_records"] = attempt_records
                return result
            else:
                print(f"[SOLVER] Numeric validation passed; triggering round {reflection_loops + 1}/{required_rounds} logic self-check")

        reflection_loops += 1
        feedback = _format_solver_reflection_feedback(numeric_ok, issues or ["Please check the logic"], latest_logs, reflection_loops, required_rounds)
        history.append({"role": "user", "content": feedback})

        if use_schema and attempt >= 2 and not numeric_ok:
            use_schema = False

    raise RuntimeError(f"VeriScript generation failed: reached reflection limit and still failed validation; last error: {'; '.join(latest_issues[:5])}")

def _is_point_entity(VeriGraph, eid: str) -> bool:
    return eid in VeriGraph.entities and VeriGraph.entities[eid].type == "Point"

def _ensure_incidence(VeriGraph, point_id: str, seg_id: str):
    for c in VeriGraph.constraints.values():
        if c.type != "Incidence":
            continue
        if point_id in c.scope and seg_id in c.scope:
            return None
    return VeriGraph.add_constraint("Incidence", [point_id, seg_id], source="proof")

def _ensure_on_circle(VeriGraph, point_id: str, circle_id: str, source: str = "proof"):
    for c in VeriGraph.constraints.values():
        if c.type not in ("OnCircle", "PointOnCircle"):
            continue
        if point_id in c.scope and circle_id in c.scope:
            return None
    return VeriGraph.add_constraint("OnCircle", [point_id, circle_id], source=source)

def _ensure_segment_between_points(VeriGraph, p1: str, p2: str) -> str:
    if p1 == p2:
        raise ValueError("AngleBisector requires distinct points to construct a segment")
    if not _is_point_entity(VeriGraph, p1) or not _is_point_entity(VeriGraph, p2):
        raise ValueError(f"Points required by AngleBisector do not exist: {p1}, {p2}")
    candidates = [f"SEG_{p1}{p2}", f"SEG_{p2}{p1}"]
    for seg_id in candidates:
        ent = VeriGraph.entities.get(seg_id)
        if ent and ent.type == "Segment":
            _ensure_incidence(VeriGraph, p1, seg_id)
            _ensure_incidence(VeriGraph, p2, seg_id)
            return seg_id
    seg_id = _canonical_segment_id(p1, p2)
    VeriGraph.add_entity("Segment", id=seg_id)
    _ensure_incidence(VeriGraph, p1, seg_id)
    _ensure_incidence(VeriGraph, p2, seg_id)
    return seg_id

def _ensure_triangle_entity(VeriGraph, verts: Tuple[str, str, str]) -> str:
    a, b, c = verts
    for pid in (a, b, c):
        if not _is_point_entity(VeriGraph, pid):
            raise ValueError(f"Point required by Triangle does not exist: {pid}")
    tri_id = f"TRI_{a}{b}{c}"
    ent = VeriGraph.entities.get(tri_id)
    if ent is None:
        VeriGraph.add_entity("Triangle", id=tri_id, attrs=Attrs.triangle([a, b, c], provenance="proof"))
    elif ent.type != "Triangle":
        raise ValueError(f"{tri_id} already exists but has type {ent.type}")
    return tri_id


def _normalize_named_entity_token(VeriGraph, token: str, point_lookup: Optional[Dict[str, str]] = None) -> str:
    tok = (token or "").strip()
    if not tok:
        return tok
    if tok in VeriGraph.entities:
        return tok
    alias_map = _collect_alias_map(VeriGraph)
    tok = alias_map.get(tok, tok)
    if tok in VeriGraph.entities:
        return tok
    point_lookup = point_lookup or _build_point_lookup(VeriGraph)
    upper = tok.upper()
    if upper.startswith("SEG_"):
        body = upper[4:]
        pts = _split_token_into_points(point_lookup, body, 2)
        if pts:
            try:
                return _ensure_segment_between_points(VeriGraph, pts[0], pts[1])
            except Exception:
                return upper
        return upper
    if upper.startswith("TRI_"):
        body = upper[4:]
        pts = _split_token_into_points(point_lookup, body, 3)
        if pts:
            try:
                return _ensure_triangle_entity(VeriGraph, pts)
            except Exception:
                return upper
        return upper
    if upper.startswith("ANG_"):
        body = upper[4:]
        pts = _split_token_into_points(point_lookup, body, 3)
        if pts:
            try:
                return _ensure_angle_entity(VeriGraph, pts[0], pts[1], pts[2])
            except Exception:
                return upper
        return upper
    if tok.startswith("∠"):
        body = tok.replace("∠", "").upper()
        pts = _split_token_into_points(point_lookup, body, 3)
        if pts:
            try:
                return _ensure_angle_entity(VeriGraph, pts[0], pts[1], pts[2])
            except Exception:
                return tok
        return tok
    return tok


def _copy_graph_state(dst: "GeometryGraph", src: "GeometryGraph"):
    """Overwrite dst's entities/constraints/meta with src's for safe commit of transient edits."""
    dst.entities = copy.deepcopy(src.entities)
    dst.constraints = copy.deepcopy(src.constraints)
    dst.meta = copy.deepcopy(src.meta)

def _ensure_angle_entity(VeriGraph, a: str, b: str, c: str) -> str:
    for pid in (a, b, c):
        if not _is_point_entity(VeriGraph, pid):
            raise ValueError(f"Point required by AngleBisector does not exist: {pid}")
    angle_id = f"∠{a}{b}{c}"
    ent = VeriGraph.entities.get(angle_id)
    if ent is None:
        VeriGraph.add_entity("Angle", id=angle_id, attrs=Attrs.angle(angle_id, provenance="proof"))
    elif ent.type != "Angle":
        raise ValueError(f"{angle_id} already exists but has type {ent.type}")
    return angle_id

def _normalize_angle_bisector_scope(VeriGraph, scope: Tuple[str, ...]) -> Tuple[str, str]:
    if len(scope) == 2:
        seg_tok, ang_tok = scope
        alias_map = _collect_alias_map(VeriGraph)
        seg_tok = alias_map.get(seg_tok, seg_tok)
        ang_tok = alias_map.get(ang_tok, ang_tok)
        return (seg_tok, ang_tok)
    if len(scope) == 3:
        bis_seg, arm1, arm2 = scope

        def _ends(seg_id: str) -> Tuple[str, str]:
            try:
                return _segment_endpoints_for_VeriScript(VeriGraph, seg_id)
            except Exception as exc:
                raise ValueError(f"AngleBisector parsing failed: cannot obtain endpoints of {seg_id}") from exc

        arm1_pts = _ends(arm1)
        arm2_pts = _ends(arm2)
        common = set(arm1_pts).intersection(arm2_pts)
        if len(common) != 1:
            raise ValueError(
                f"AngleBisector parsing failed: {arm1} and {arm2} have no unique common vertex (current: {common})"
            )
        vertex = common.pop()
        left = arm1_pts[0] if arm1_pts[1] == vertex else arm1_pts[1]
        right = arm2_pts[0] if arm2_pts[1] == vertex else arm2_pts[1]

        bis_pts = _ends(bis_seg)
        if vertex not in bis_pts:
            raise ValueError(
                f"AngleBisector parsing failed: bisector {bis_seg} does not pass through common vertex {vertex}"
            )
        target = bis_pts[0] if bis_pts[1] == vertex else bis_pts[1]
        seg_id = _ensure_segment_between_points(VeriGraph, vertex, target)
        angle_id = _ensure_angle_entity(VeriGraph, left, vertex, right)
        return (seg_id, angle_id)
    if len(scope) == 4 and all(_is_point_entity(VeriGraph, token) for token in scope):
        left, vertex, right, bis_pt = scope
        seg_id = _ensure_segment_between_points(VeriGraph, vertex, bis_pt)
        angle_id = _ensure_angle_entity(VeriGraph, left, vertex, right)
        return (seg_id, angle_id)
    raise ValueError(f"AngleBisector requires (segment, angle) or (S, P, T, O) point format, got: {scope}")



def _auto_create_midpoint_point(VeriGraph, point_id: str, anchors: Tuple[str, ...]):
    ent = VeriGraph.entities.get(point_id)
    if ent:
        if ent.type != "Point":
            raise ValueError(f"Midpoint requires a point, but {point_id} is already a {ent.type}")
        return
    coords: List[Tuple[float, float]] = []
    for anchor in anchors:
        if not anchor or not _is_point_entity(VeriGraph, anchor):
            continue
        try:
            coords.append(VeriGraph.get_point_xy(anchor))
        except Exception:
            continue
        if len(coords) == 2:
            break
    if len(coords) == 2:
        mx = 0.5 * (coords[0][0] + coords[1][0])
        my = 0.5 * (coords[0][1] + coords[1][1])
    elif coords:
        mx, my = coords[0]
    else:
        span = max(5.0, max(len(VeriGraph.entities), 1) * 0.2)
        mx = random.uniform(-span, span)
        my = random.uniform(-span, span)
    VeriGraph.add_entity("Point", id=point_id, attrs=Attrs.point(mx, my, provenance="auto_midpoint"))

def _normalize_midpoint_scope(VeriGraph, args: List[str]) -> Tuple[str, str]:
    if len(args) == 2:
        m, seg_id = args
        seg_id = (seg_id or "").strip()
        if not seg_id:
            raise ValueError("Midpoint requires segment ID")
        alias_map = _collect_alias_map(VeriGraph)
        seg_id = alias_map.get(seg_id, seg_id)
        ent = VeriGraph.entities.get(seg_id)
        if ent and ent.type != "Segment":
            raise ValueError(f"Midpoint dependency {seg_id} is not a segment")
        if ent is None:
            point_lookup = _build_point_lookup(VeriGraph)
            upper = seg_id.upper()
            if upper.startswith("SEG_"):
                pts = _split_token_into_points(point_lookup, upper[4:], 2)
            else:
                pts = _split_token_into_points(point_lookup, upper, 2)
            if pts:
                seg_id = _ensure_segment_between_points(VeriGraph, pts[0], pts[1])
            else:
                raise ValueError(f"Midpoint depends on segment {seg_id}, but endpoints do not exist or cannot be parsed")
        anchors: Tuple[str, ...] = ()
        try:
            anchors = _segment_endpoints_for_VeriScript(VeriGraph, seg_id)
        except Exception:
            anchors = ()
        _auto_create_midpoint_point(VeriGraph, m, anchors)
        return (m, seg_id)
    if len(args) == 3:
        m, u, v = args
        seg_id = _ensure_segment_between_points(VeriGraph, u, v)
        _auto_create_midpoint_point(VeriGraph, m, (u, v))
        return (m, seg_id)
    raise ValueError("Midpoint requires [M, SEG] or [M, U, V]")

_SEGMENT_EDGE_TYPES = {"EqualLength", "Parallel", "Perpendicular"}

def _normalize_segment_edge_scope(
    VeriGraph,
    args: List[str],
    edge_type: str,
    expected_segments: int = 2,
) -> Tuple[str, ...]:
    """
    Normalize various input forms (SEG_XY, paired endpoints, etc.) into segment IDs.
    Allows ["SEG_AB","SEG_CD"] or ["A","B","C","D"] (which will become SEG_AB and SEG_CD).
    """
    tokens = [tok.strip() for tok in args if isinstance(tok, str) and tok.strip()]
    if not tokens:
        raise ValueError(f"{edge_type} requires {expected_segments} segments")
    point_lookup = _build_point_lookup(VeriGraph)
    alias_map = _collect_alias_map(VeriGraph)
    segments: List[str] = []
    idx = 0
    n = len(tokens)

    while idx < n:
        tok = tokens[idx]
        alias = alias_map.get(tok, tok)
        ent = VeriGraph.entities.get(alias)
        if ent and ent.type == "Segment":
            segments.append(alias)
            idx += 1
        else:
            upper = tok.upper()
            if upper.startswith("SEG_"):
                segments.append(_normalize_named_entity_token(VeriGraph, upper, point_lookup))
                idx += 1
            else:
                if idx + 1 >= n:
                    raise ValueError(f"{edge_type} segment arguments must be given as endpoint pairs, e.g. ['A','B','C','D'] -> SEG_AB, SEG_CD")
                p = tokens[idx]
                q = tokens[idx + 1]
                segments.append(_ensure_segment_between_points(VeriGraph, p, q))
                idx += 2
        if len(segments) == expected_segments and idx < n:
            raise ValueError(f"{edge_type} requires only {expected_segments} segments; received too many arguments")
    if len(segments) != expected_segments:
        raise ValueError(f"{edge_type} requires {expected_segments} segments; received {len(segments)}")
    return tuple(segments)


def _build_point_lookup(VeriGraph) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for eid, ent in VeriGraph.entities.items():
        if ent.type == "Point":
            lookup[eid] = eid
            lookup[eid.upper()] = eid
            lookup[eid.lower()] = eid
    return lookup


def _split_token_into_points(point_lookup: Dict[str, str], body: str, expected: int) -> Optional[Tuple[str, ...]]:
    body = body.upper()
    cache: Dict[Tuple[int, int], Optional[List[str]]] = {}

    def dfs(idx: int, left: int) -> Optional[List[str]]:
        if (idx, left) in cache:
            return cache[(idx, left)]
        if left == 0:
            return [] if idx == len(body) else None
        for end in range(idx + 1, len(body) + 1):
            chunk = body[idx:end]
            point = point_lookup.get(chunk)
            if not point:
                continue
            tail = dfs(end, left - 1)
            if tail is not None:
                cache[(idx, left)] = [point] + tail
                return cache[(idx, left)]
        cache[(idx, left)] = None
        return None

    result = dfs(0, expected)
    if result and len(result) == expected:
        return tuple(result)
    return None


class ExpressionEvaluator:
    """Controlled expression evaluator supporting geometric quantities (lengths/areas/angles/vectors) and common arithmetic"""
    _ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)
    _ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)

    def __init__(self, VeriGraph: "GeometryGraph"):
        self.VeriGraph = VeriGraph
        self.point_lookup: Dict[str, str] = {}

    def evaluate(self, expr: str) -> float:
        if not isinstance(expr, str):
            raise ValueError("Expression must be a string")
        cleaned = expr.strip()
        if not cleaned:
            raise ValueError("Expression must not be empty")
        cleaned = cleaned.replace("^", "**")
        self.point_lookup = _build_point_lookup(self.VeriGraph)
        try:
            node = ast.parse(cleaned, mode="eval").body
        except SyntaxError as exc:
            raise ValueError(f"Expression syntax error: {exc}") from exc
        value = self._eval_node(node)
        if isinstance(value, complex):
            return value
        vec_val = _as_vector2(value)
        if vec_val is not None:
            return vec_val
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Expression result is not real: {value}")

    def _eval_node(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        if type(node).__name__ == "Num":
            return node.n
        if isinstance(node, ast.Name):
            return self._resolve_identifier(node.id)
        if isinstance(node, ast.BinOp) and isinstance(node.op, self._ALLOWED_BINOPS):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left ** right
            if isinstance(node.op, ast.Mod):
                return left % right
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, self._ALLOWED_UNARYOPS):
            val = self._eval_node(node.operand)
            return val if isinstance(node.op, ast.UAdd) else -val
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Unsupported function call")
            fname = node.func.id
            args = [self._eval_node(arg) for arg in node.args]
            return self._call_function(fname, args)
        if isinstance(node, ast.Tuple):
            return tuple(self._eval_node(elt) for elt in node.elts)
        if isinstance(node, ast.List):
            return [self._eval_node(elt) for elt in node.elts]
        raise ValueError(f"Expression contains invalid node: {ast.dump(node, include_attributes=False)}")

    # ---- Functions and symbols ----
    def _call_function(self, name: str, args: List[Any]):
        table = {
            "Length": self._func_length,
            "Distance": self._func_length,
            "Area": self._func_area,
            "Angle": self._func_angle,
            "Sin": lambda x: math.sin(math.radians(self._require_scalar(x))),
            "Cos": lambda x: math.cos(math.radians(self._require_scalar(x))),
            "Tan": lambda x: math.tan(math.radians(self._require_scalar(x))),
            "Deg": lambda x: math.degrees(self._require_scalar(x)),
            "Rad": lambda x: math.radians(self._require_scalar(x)),
            "Abs": lambda x: abs(self._require_scalar(x)),
            "Sqrt": lambda x: math.sqrt(self._require_scalar(x)),
            "Pow": lambda a, b: self._require_scalar(a) ** self._require_scalar(b),
            "Min": lambda *vals: min(self._require_scalar(v) for v in vals),
            "Max": lambda *vals: max(self._require_scalar(v) for v in vals),
            "Vector": self._func_vector,
            "Dot": self._func_dot,
            "Cross": self._func_cross,
            "Complex": self._func_complex,
            "Arg": self._func_arg,
            "Normalize": self._func_normalize,
            "ComplexRatio": self._func_complex_ratio,
        }
        key = name.strip()
        func = table.get(key)
        if func is None:
            func = table.get(key.lower())
            if func is None:
                func = {k.lower(): v for k, v in table.items()}.get(key.lower())
        if func is None:
            raise ValueError(f"Expression contains unsupported function: {name}")
        try:
            return func(*args)
        except TypeError as exc:
            raise ValueError(f"{name} argument error: {exc}") from exc

    def _resolve_identifier(self, name: str):
        upper = name.upper()
        # Prefer existing entities (point/segment/angle/triangle) to avoid clash between single-letter point and math constants
        if upper in self.VeriGraph.entities:
            ent = self.VeriGraph.entities[upper]
            if ent.type in ("Point", "Segment", "Triangle", "Angle"):
                return upper
        pts2 = self._points_from_token(upper, 2)
        if pts2:
            return self._length_between_points(pts2[0], pts2[1])
        if upper.startswith("S_"):
            verts = self._points_from_token(upper[2:], 3)
            if verts:
                return self._triangle_area_points(verts)
            raise ValueError(f"Cannot parse area symbol {name}")
        if upper == "PI":
            return math.pi
        if upper == "E":
            return math.e
        return name

    # ---- Function implementations ----
    def _func_length(self, *args):
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, (int, float)):
                return float(arg)
            if isinstance(arg, str):
                return self._length_from_entity(arg)
            if isinstance(arg, (tuple, list)) and len(arg) == 2:
                return math.hypot(arg[0], arg[1])
            raise ValueError("Length(x) only accepts a segment/point or scalar")
        if len(args) == 2:
            p1 = self._require_point_id(args[0])
            p2 = self._require_point_id(args[1])
            return self._length_between_points(p1, p2)
        raise ValueError("Length only supports 1 or 2 arguments")

    def _func_area(self, *args):
        if len(args) == 1 and isinstance(args[0], str):
            token = args[0]
            verts = self._triangle_vertices(token)
            if verts:
                return self._triangle_area_points(verts)
        if len(args) == 3:
            verts = tuple(self._require_point_id(arg) for arg in args)
            return self._triangle_area_points(verts)
        raise ValueError("Area requires a triangle entity or three points")

    def _func_angle(self, *args):
        if len(args) == 1:
            verts = self._angle_vertices(str(args[0]))
            if not verts:
                raise ValueError(f"Cannot parse angle {args[0]}")
            return self._angle_from_points(verts)
        if len(args) == 3:
            verts = tuple(self._require_point_id(arg) for arg in args)
            return self._angle_from_points(verts)
        raise ValueError("Angle only supports 1 angle or 3 points")

    def _func_vector(self, start, end):
        sx, sy = self._point_xy(start)
        ex, ey = self._point_xy(end)
        return (ex - sx, ey - sy)

    def _func_dot(self, a, b):
        v1 = self._value_as_vector(a)
        v2 = self._value_as_vector(b)
        return v1[0] * v2[0] + v1[1] * v2[1]

    def _func_cross(self, a, b):
        v1 = self._value_as_vector(a)
        v2 = self._value_as_vector(b)
        return v1[0] * v2[1] - v1[1] * v2[0]

    def _func_complex(self, *args):
        if len(args) == 1:
            return self._value_as_complex(args[0])
        if len(args) == 2:
            start = self._require_point_id(args[0])
            end = self._require_point_id(args[1])
            vec = self._func_vector(start, end)
            return complex(vec[0], vec[1])
        raise ValueError("Complex requires 1 vector or 2 points")

    def _func_arg(self, value):
        comp = self._value_as_complex(value)
        import math
        return math.degrees(math.atan2(comp.imag, comp.real))

    def _func_normalize(self, value):
        vx, vy = self._value_as_vector(value)
        import math
        norm = math.hypot(vx, vy)
        if norm <= 1e-12:
            raise ValueError("Normalize: cannot normalize zero vector")
        return (vx / norm, vy / norm)

    def _func_complex_ratio(self, num, denom):
        numerator = self._value_as_complex(num)
        denominator = self._value_as_complex(denom)
        if abs(denominator) <= 1e-12:
            raise ValueError("ComplexRatio: denominator vector has zero length")
        return numerator / denominator

    # ---- Basic utilities ----
    def _require_scalar(self, value):
        if isinstance(value, (int, float)):
            return float(value)
        raise ValueError(f"Expected real number, received {value}")

    def _require_point_id(self, value):
        if isinstance(value, str) and value in self.VeriGraph.entities and self.VeriGraph.entities[value].type == "Point":
            return value
        key = str(value).upper()
        if key in self.point_lookup:
            return self.point_lookup[key]
        raise ValueError(f"Cannot parse point {value}")

    def _point_xy(self, value):
        pid = self._require_point_id(value)
        return self.VeriGraph.get_point_xy(pid)

    def _value_as_vector(self, value):
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return (float(value[0]), float(value[1]))
        if isinstance(value, complex):
            return (float(value.real), float(value.imag))
        if isinstance(value, str):
            pts = self._points_from_token(value.upper(), 2)
            if pts:
                return self._func_vector(pts[0], pts[1])
        raise ValueError(f"Cannot parse vector {value}")

    def _value_as_complex(self, value):
        if isinstance(value, complex):
            return value
        try:
            vx, vy = self._value_as_vector(value)
            return complex(vx, vy)
        except Exception as exc:
            raise ValueError(f"Cannot parse complex vector {value}") from exc

    def _points_from_token(self, body: str, expected: int) -> Optional[Tuple[str, ...]]:
        return _split_token_into_points(self.point_lookup, body, expected)

    def _length_from_entity(self, token: str) -> float:
        norm = _normalize_named_entity_token(self.VeriGraph, token, self.point_lookup)
        ent = self.VeriGraph.entities.get(norm)
        if ent and ent.type == "Segment":
            p, q = _segment_endpoints_for_VeriScript(self.VeriGraph, norm)
            return self._length_between_points(p, q)
        raise ValueError(f"Length cannot parse {token}")

    def _length_between_points(self, p1: str, p2: str) -> float:
        return _dist(self.VeriGraph.get_point_xy(p1), self.VeriGraph.get_point_xy(p2))

    def _triangle_vertices(self, token: str) -> Optional[Tuple[str, str, str]]:
        norm = _normalize_named_entity_token(self.VeriGraph, token, self.point_lookup)
        ent = self.VeriGraph.entities.get(norm)
        if ent and ent.type == "Triangle":
            verts = ent.attrs.data.get("verts")
            if isinstance(verts, list) and len(verts) == 3:
                return tuple(verts)
        body = token.upper().replace("TRI_", "")
        return self._points_from_token(body, 3)

    def _triangle_area_points(self, verts: Tuple[str, str, str]) -> float:
        coords = [self.VeriGraph.get_point_xy(v) for v in verts]
        return _triangle_area(*coords)

    def _angle_vertices(self, token: str) -> Optional[Tuple[str, str, str]]:
        return self.VeriGraph._angle_vertices_from_token(token, self.point_lookup)

    def _angle_from_points(self, verts: Tuple[str, str, str]) -> float:
        A = self.VeriGraph.get_point_xy(verts[0])
        B = self.VeriGraph.get_point_xy(verts[1])
        C = self.VeriGraph.get_point_xy(verts[2])
        return _angle_deg(A, B, C)


def _as_vector2(value: Any) -> Optional[Tuple[float, float]]:
    """Try to convert value to a 2D vector (x, y); return None otherwise."""
    if isinstance(value, (tuple, list)) and len(value) == 2:
        try:
            return (float(value[0]), float(value[1]))
        except Exception:
            return None
    return None


def evaluate_expr_relation(VeriGraph: "GeometryGraph", lhs: str, rhs: str, relation: str,
                           tolerance: Optional[float] = None) -> Tuple[bool, str, Dict[str, float]]:
    def _rewrite_arith_relation(lhs_expr: str, rel_expr: str, rhs_expr: str) -> Tuple[str, str, str]:
        rel_raw = (rel_expr or "").strip()
        if rel_raw in ("+", "-", "*", "/", "^", "**"):
            rhs_text = "" if rhs_expr is None else str(rhs_expr)
            for token in ("<=", ">=", "!=", "==", "="):
                if token in rhs_text:
                    left_part, right_part = rhs_text.split(token, 1)
                    return f"({lhs_expr}) {rel_raw} ({left_part.strip()})", token, right_part.strip()
            return f"({lhs_expr}) {rel_raw} ({rhs_text})", "=", "0"
        return lhs_expr, rel_expr, rhs_expr

    def _normalize_angle_tokens(expr: str) -> str:
        """Normalize ∠XYZ in the expression to ANG_XYZ, to make Angle() calls parseable."""
        if not isinstance(expr, str):
            return expr
        angle_pattern = re.compile(r"∠([A-Za-z]{3})")

        def repl(match: re.Match) -> str:
            body = match.group(1)
            if len(body) == 3 and body.isalpha():
                return f"ANG_{body.upper()}"
            return match.group(0)

        return angle_pattern.sub(repl, expr)

    lhs, relation, rhs = _rewrite_arith_relation(lhs, relation, rhs)
    lhs = _normalize_angle_tokens(lhs)
    rhs = _normalize_angle_tokens(rhs)
    evaluator = ExpressionEvaluator(VeriGraph)
    left_val = evaluator.evaluate(lhs)
    right_val = evaluator.evaluate(rhs)
    rel = (relation or "=").strip().lower()
    left_vec = _as_vector2(left_val)
    right_vec = _as_vector2(right_val)

    if left_vec is not None or right_vec is not None:
        if left_vec is None or right_vec is None:
            raise ValueError("Vector comparison requires two vectors")
        if rel not in ("=", "==", "eq", "≈", "~", "!=", "≠"):
            raise ValueError("Vector comparison only supports = or !=")
        diff_x = left_vec[0] - right_vec[0]
        diff_y = left_vec[1] - right_vec[1]
        diff_norm = math.hypot(diff_x, diff_y)
        left_norm = math.hypot(left_vec[0], left_vec[1])
        right_norm = math.hypot(right_vec[0], right_vec[1])
        scale = max(left_norm, right_norm, 1.0)
        tol = tolerance if tolerance is not None else max(1e-6, scale * 1e-6)

        def approx_eq_vec():
            return diff_norm <= tol

        if rel in ("=", "==", "eq", "≈", "~"):
            ok = approx_eq_vec()
        else:  # rel in ("!=", "≠")
            ok = not approx_eq_vec()
        msg = f"||Δv||={diff_norm:.2e} (tol={tol:.2e})"
        details = {
            "lhs": {"x": left_vec[0], "y": left_vec[1], "norm": left_norm},
            "rhs": {"x": right_vec[0], "y": right_vec[1], "norm": right_norm},
            "diff_norm": diff_norm,
            "tolerance": tol,
        }
        return ok, msg, details

    is_complex = isinstance(left_val, complex) or isinstance(right_val, complex)

    if is_complex:
        left_val = complex(left_val)
        right_val = complex(right_val)
        diff = left_val - right_val
        scale = max(abs(left_val), abs(right_val), 1.0)
        tol = tolerance if tolerance is not None else max(1e-6, scale * 1e-6)

        def approx_eq():
            return abs(diff) <= tol

        if rel in ("=", "==", "eq", "≈", "~"):
            ok = approx_eq()
        elif rel in ("!=", "≠"):
            ok = not approx_eq()
        else:
            raise ValueError("Complex comparison only supports = or !=")
        msg = f"{left_val} {relation or '='} {right_val} (|diff|={abs(diff):.2e}, tol={tol:.2e})"
        details = {
            "lhs": {"real": left_val.real, "imag": left_val.imag},
            "rhs": {"real": right_val.real, "imag": right_val.imag},
            "abs_diff": abs(diff),
            "tolerance": tol,
        }
        return ok, msg, details

    diff = left_val - right_val
    scale = max(abs(left_val), abs(right_val), 1.0)
    tol = tolerance if tolerance is not None else max(1e-6, scale * 1e-6)

    def approx_eq():
        return abs(diff) <= tol

    if rel in ("=", "==", "eq", "≈", "~"):
        ok = approx_eq()
    elif rel in ("!=", "≠"):
        ok = not approx_eq()
    elif rel in ("<=", "≤", "le"):
        ok = left_val <= right_val + tol
    elif rel in (">=", "≥", "ge"):
        ok = left_val >= right_val - tol
    elif rel in ("<", "lt"):
        ok = left_val < right_val + tol
    elif rel in (">", "gt"):
        ok = left_val > right_val - tol
    else:
        raise ValueError(f"Unknown comparison operator {relation}")
    msg = f"{left_val:.6g} {relation or '='} {right_val:.6g} (tol={tol:.2e})"
    return ok, msg, {"lhs": float(left_val), "rhs": float(right_val), "diff": float(diff), "tolerance": tol}

def _suggest_free_point_xy(VeriGraph) -> Tuple[float, float]:
    coords: List[Tuple[float, float]] = []
    pt_ids: List[str] = []
    for eid, ent in VeriGraph.entities.items():
        if ent.type != "Point":
            continue
        try:
            coords.append(VeriGraph.get_point_xy(eid))
            pt_ids.append(eid)
        except Exception:
            continue

    def _build_rng() -> random.Random:
        meta_seed = VeriGraph.meta.get("seed") or VeriGraph.meta.get("problem_seed") or VeriGraph.meta.get("VeriScript_seed") or ""
        ent_tokens = ",".join(sorted(VeriGraph.entities.keys()))
        cons_tokens = ",".join(sorted(VeriGraph.constraints.keys()))
        material = f"free|{meta_seed}|{len(pt_ids)}|{ent_tokens}|{cons_tokens}"
        digest = hashlib.blake2b(material.encode("utf-8"), digest_size=8).digest()
        return random.Random(int.from_bytes(digest, "big"))

    rng = _build_rng()

    if coords:
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        span = max(xmax - xmin, ymax - ymin, 1.0)
        base_pad = max(span * 0.35, 0.8)
        min_gap = max(span * 0.22, 0.6)
    else:
        xmin, xmax, ymin, ymax = 0.0, 5.0, 0.0, 5.0
        cx, cy = 2.5, 2.5
        span = 5.0
        base_pad = 2.0
        min_gap = 1.0

    x_range = (xmin - base_pad, xmax + base_pad)
    y_range = (ymin - base_pad, ymax + base_pad)
    min_gap_sq = min_gap * min_gap
    pad = base_pad

    def _far_enough(x: float, y: float, thresh_sq: float) -> bool:
        for px, py in coords:
            if (x - px) ** 2 + (y - py) < thresh_sq:
                return False
        return True

    for attempt in range(80):
        x = _quantized_sample(x_range, rng=rng)
        y = _quantized_sample(y_range, rng=rng)
        if _far_enough(x, y, min_gap_sq):
            return (x, y)
        if attempt in (20, 45, 65):
            pad *= 1.15
            min_gap_sq *= 0.8
            x_range = (xmin - pad, xmax + pad)
            y_range = (ymin - pad, ymax + pad)

    dx, dy = _PYTHAGOREAN_DIRECTIONS[rng.randrange(len(_PYTHAGOREAN_DIRECTIONS))]
    length = math.hypot(dx, dy) or 1.0
    step = math.sqrt(min_gap_sq) + pad + 0.2
    fallback_x = cx + dx / length * step
    fallback_y = cy + dy / length * step
    return (
        _quantized_sample((fallback_x - 0.25, fallback_x + 0.25), rng=rng),
        _quantized_sample((fallback_y - 0.25, fallback_y + 0.25), rng=rng),
    )


def _segment_endpoints_for_VeriScript(VeriGraph, seg_id: str) -> Tuple[str, str]:
    if isinstance(seg_id, str):
        alias_map = _collect_alias_map(VeriGraph)
        seg_id = alias_map.get(seg_id, seg_id)
    pts: List[str] = []
    for c in VeriGraph.constraints.values():
        if c.type != "Incidence" or seg_id not in c.scope:
            continue
        for token in c.scope:
            if token != seg_id and _is_point_entity(VeriGraph, token) and token not in pts:
                pts.append(token)
    if len(pts) < 2:
        raise ValueError(f"Segment {seg_id} lacks endpoint information; cannot place a new point on it")
    # If seg_id is of form SEG_XY, prefer returning in name order to keep the partition direction controllable
    upper = seg_id.upper()
    if upper.startswith("SEG_") and len(upper) >= 6:
        body = upper[4:]
        if len(body) >= 2:
            a, b = body[0], body[1]
            # Use only when the two endpoints derived from the name are distinct and both in the known endpoint set;
            # otherwise fall back to the actual endpoints collected from Incidence, to avoid SEG_HH1 being wrongly split into H and 1.
            if a != b and a in pts and b in pts:
                return a, b
    if len(pts) == 2:
        return pts[0], pts[1]
    coords = []
    for p in pts:
        try:
            coords.append(VeriGraph.get_point_xy(p))
        except Exception:
            coords.append((0.0, 0.0))
    idx_min = min(range(len(pts)), key=lambda i: coords[i][0] + coords[i][1])
    idx_max = max(range(len(pts)), key=lambda i: coords[i][0] + coords[i][1])
    if idx_min == idx_max:
        return pts[0], pts[1]
    return pts[idx_min], pts[idx_max]


def _apply_VeriScript_on_VeriGraph(VeriGraph, actions: List[VeriScriptAction], *, verify_only: bool = False) -> List[Dict[str, Any]]:
    """Apply VeriScript actions step by step to the graph: essentially "add edges / add constraints"."""
    logs = []

    def _is_parallel_declared(seg_a: str, seg_b: str) -> bool:
        target = {seg_a, seg_b}
        for cons in VeriGraph.constraints.values():
            if cons.type == "Parallel" and set(cons.scope) == target:
                return True
        return False

    def _verify_constraint_snapshot(constraint_ids: List[str]):
        if not constraint_ids:
            return {}
        report = VeriGraph.verify_constraints()
        snap = {}
        for cid in constraint_ids:
            snap[cid] = report.get(cid, {"ok": None, "msg": "not-verifiable"})
        return snap

    for i, a in enumerate(actions, start=1):
        ok, msg = True, "ok"
        geom_ok = None
        verification = None
        tracked_constraints: List[str] = []
        derived_records: List[Dict[str, Any]] = []
        try:
            if a.op == "AddEdge":
                if a.type == "ExprConstraint":
                    if len(a.args) < 3:
                        raise ValueError("ExprConstraint requires at least lhs, relation, rhs")
                    lhs, relation, rhs = a.args[:3]
                    params = {"lhs": lhs, "relation": relation, "rhs": rhs}
                    if len(a.args) >= 4:
                        try:
                            params["tolerance"] = _author_parse_float(a.args[3], "tol")
                        except Exception:
                            raise ValueError("ExprConstraint fourth argument must be a numeric tolerance")
                    cid = VeriGraph.add_constraint("ExprConstraint", [], params=params, source="proof")
                    tracked_constraints.append(cid)
                    msg = "expr"
                elif a.type == "VectorLinear":
                    if len(a.args) < 5 or (len(a.args) - 2) % 3 != 0:
                        raise ValueError("VectorLinear must be of form [U,V,k1,P1,Q1,...] with at least one linear-combination term")
                    norm_scope: List[str] = []
                    norm_scope.append(_normalize_named_entity_token(VeriGraph, a.args[0]))
                    norm_scope.append(_normalize_named_entity_token(VeriGraph, a.args[1]))
                    for idx in range(2, len(a.args), 3):
                        k = a.args[idx]
                        try:
                            _author_parse_float(k, "coeff")
                        except Exception as exc:
                            raise ValueError(f"VectorLinear coefficient must be numeric: {k}") from exc
                        norm_scope.append(k)
                        norm_scope.append(_normalize_named_entity_token(VeriGraph, a.args[idx + 1]))
                        norm_scope.append(_normalize_named_entity_token(VeriGraph, a.args[idx + 2]))
                    cid = VeriGraph.add_constraint("VectorLinear", list(norm_scope), source="proof")
                    tracked_constraints.append(cid)

                    # Supplementary parallel inference: if the linear combination has only one term, or all terms are known to be parallel, default the target vector || base vector
                    target_seg = _ensure_segment_between_points(VeriGraph, norm_scope[0], norm_scope[1])
                    dir_segs: List[str] = []
                    for idx in range(3, len(norm_scope), 3):  # take Pi, Qi successively
                        dir_segs.append(_ensure_segment_between_points(VeriGraph, norm_scope[idx], norm_scope[idx + 1]))
                    base_seg = dir_segs[0] if dir_segs else None
                    if base_seg:
                        all_parallel = len(dir_segs) == 1 or all(_is_parallel_declared(base_seg, s) for s in dir_segs[1:])
                        if all_parallel and not _is_parallel_declared(target_seg, base_seg):
                            try:
                                cid_par = VeriGraph.add_constraint("Parallel", [target_seg, base_seg], source="derived")
                                tracked_constraints.append(cid_par)
                                derived_records.append({"id": cid_par, "type": "Parallel", "scope": [target_seg, base_seg], "source": "derived"})
                            except Exception:
                                pass
                else:
                    scope = tuple(a.args)
                    if a.type == "AngleBisector":
                        scope = _normalize_angle_bisector_scope(VeriGraph, scope)
                    elif a.type == "Midpoint":
                        scope = _normalize_midpoint_scope(VeriGraph, list(a.args))
                    elif a.type in _SEGMENT_EDGE_TYPES:
                        scope = _normalize_segment_edge_scope(VeriGraph, list(a.args), a.type)
                    norm_scope = [_normalize_named_entity_token(VeriGraph, tok) for tok in scope]
                    cid = VeriGraph.add_constraint(a.type, list(norm_scope), source="proof")
                    tracked_constraints.append(cid)
                    if a.type == "AngleBisector" and scope != tuple(a.args):
                        msg = f"ok (normalized to {scope[0]}, {scope[1]})"
            elif a.op == "AddPoint":
                if not a.args:
                    raise ValueError("AddPoint missing new point ID")
                pid = a.args[0]
                if pid in VeriGraph.entities:
                    raise ValueError(f"Point {pid} already exists")
                if a.type == "Free":
                    x, y = _suggest_free_point_xy(VeriGraph)
                    VeriGraph.add_entity("Point", id=pid, attrs=Attrs.point(x, y, provenance="proof"))
                    try:
                        VeriGraph.entities[pid].attrs.data["origin"] = "free"
                    except Exception:
                        pass
                elif a.type == "OnSegment":
                    if len(a.args) < 2:
                        raise ValueError("AddPoint-OnSegment must provide segment ID")
                    seg_id = a.args[1]
                    seg_ent = VeriGraph.entities.get(seg_id)
                    if not seg_ent or seg_ent.type != "Segment":
                        raise ValueError(f"{seg_id} is not a valid segment")
                    u, v = _segment_endpoints_for_VeriScript(VeriGraph, seg_id)
                    ux, uy = VeriGraph.get_point_xy(u)
                    vx, vy = VeriGraph.get_point_xy(v)
                    ratio_val: Optional[float] = None
                    if len(a.args) >= 3:
                        try:
                            ratio_val = _author_parse_float(a.args[2], "ratio")
                        except Exception:
                            ratio_val = None
                    if ratio_val is not None:
                        ratio_val = max(0.0, min(1.0, ratio_val))
                    # Use midpoint as initial value; if no ratio is given, keep it as a "sliding point" for later constraints to determine
                    seed_ratio = 0.5 if ratio_val is None else ratio_val
                    x = ux + (vx - ux) * seed_ratio
                    y = uy + (vy - uy) * seed_ratio
                    VeriGraph.add_entity("Point", id=pid, attrs=Attrs.point(x, y, provenance="proof"))
                    cid_inc = _ensure_incidence(VeriGraph, pid, seg_id)
                    if cid_inc:
                        tracked_constraints.append(cid_inc)
                    try:
                        params = {"ratio": ratio_val} if ratio_val is not None else {}
                        cid_pos = VeriGraph.add_constraint("PointOnSegment", [pid, seg_id], params=params, source="proof")
                        tracked_constraints.append(cid_pos)
                    except Exception:
                        pass
                elif a.type in ("OnCircle", "PointOnCircle"):
                    if len(a.args) < 2:
                        raise ValueError("AddPoint-OnCircle must provide circle ID")
                    circle_id = _normalize_named_entity_token(VeriGraph, a.args[1])
                    circle_ent = VeriGraph.entities.get(circle_id)
                    if not circle_ent or circle_ent.type != "Circle":
                        raise ValueError(f"{circle_id} is not a valid circle")
                    params = VeriGraph._circle_params(circle_id)
                    if not params:
                        raise ValueError(f"{circle_id} circle parameters missing")
                    cx, cy, r, _ = params
                    if r <= 1e-9:
                        r = 1.0
                    x = cx + r
                    y = cy
                    VeriGraph.add_entity("Point", id=pid, attrs=Attrs.point(x, y, provenance="proof"))
                    cid = _ensure_on_circle(VeriGraph, pid, circle_id, source="proof")
                    if cid:
                        tracked_constraints.append(cid)
                else:
                    raise ValueError(f"Unknown AddPoint type: {a.type}")
                msg = f"point {pid}"
            elif a.op == "AddAuxLine":
                if len(a.args) < 2:
                    raise ValueError("AddAuxLine requires two endpoints")
                p, q = a.args[:2]
                seg_id = _ensure_segment_between_points(VeriGraph, p, q)
                msg = f"segment {seg_id}"
            elif a.op == "AddFunction":
                if len(a.args) < 2:
                    raise ValueError("AddFunction requires function ID and expression")
                func_id = a.args[0]
                expr_text = str(a.args[1]).strip()
                if not _normalize_function_expr(expr_text):
                    raise ValueError("AddFunction expression is empty")
                if _compile_function_expr(expr_text) is None:
                    raise ValueError(f"AddFunction expression cannot be parsed: {expr_text}")
                xmin_raw = a.args[2] if len(a.args) >= 3 else "-5"
                xmax_raw = a.args[3] if len(a.args) >= 4 else "5"
                samples_raw = a.args[4] if len(a.args) >= 5 else "240"
                xmin_val = _author_parse_float(xmin_raw, "x_min")
                xmax_val = _author_parse_float(xmax_raw, "x_max")
                if xmax_val < xmin_val:
                    xmin_val, xmax_val = xmax_val, xmin_val
                if abs(xmax_val - xmin_val) < 1e-9:
                    xmax_val = xmin_val + 1.0
                try:
                    samples_val = int(str(samples_raw).strip())
                except Exception:
                    samples_val = 240
                samples_val = max(20, min(samples_val, 2000))
                ent = VeriGraph.entities.get(func_id)
                if ent and ent.type != "Function":
                    raise ValueError(f"{func_id} already exists but is not a function")
                attrs = Attrs({
                    "expr": expr_text,
                    "xmin": float(xmin_val),
                    "xmax": float(xmax_val),
                    "samples": int(samples_val),
                    "provenance": "proof",
                })
                if ent:
                    VeriGraph.entities[func_id].attrs = attrs
                else:
                    VeriGraph.add_entity("Function", id=func_id, attrs=attrs)
                msg = f"function {func_id}"
            elif a.op == "VerifyPoint":
                if len(a.args) < 3:
                    raise ValueError("VerifyPoint requires point ID and coordinates (x,y)")
                pid = a.args[0]
                if pid not in VeriGraph.entities or VeriGraph.entities[pid].type != "Point":
                    raise ValueError(f"VerifyPoint point {pid} does not exist")
                x_in = _author_parse_float(a.args[1], "x")
                y_in = _author_parse_float(a.args[2], "y")
                tol = _author_parse_float(a.args[3], "tol", default=1e-4) if len(a.args) >= 4 else 1e-4
                x_act, y_act = VeriGraph.get_point_xy(pid)
                dx = x_act - x_in
                dy = y_act - y_in
                geom_ok = abs(dx) <= tol and abs(dy) <= tol
                ok = bool(geom_ok)
                verification = {"dx": dx, "dy": dy, "tol": tol}
                msg = f"verify {pid}: dx={dx:.3g}, dy={dy:.3g}, tol={tol:g}"
            elif a.op == "VerifyFunction":
                if len(a.args) < 2:
                    raise ValueError("VerifyFunction requires sufficient arguments")
                verify_type = a.type
                if verify_type not in VeriScript_VERIFY_FUNCTION_TYPES:
                    raise ValueError(f"VerifyFunction type not supported: {verify_type}")
                if verify_type == "PointsOnFunction":
                    func_token = a.args[0]
                    fn, expr = _resolve_function_spec(VeriGraph, func_token)
                    if fn is None:
                        raise ValueError(f"VerifyFunction cannot parse function: {func_token}")
                    tol, items = _parse_points_on_function_args(VeriGraph, a.args[1:])
                    failures = []
                    details = {}
                    for item_type, payload in items:
                        if item_type == "point":
                            pid = payload
                            x_val, y_val = VeriGraph.get_point_xy(pid)
                            label = pid
                        else:
                            x_val, y_val = payload
                            label = f"({x_val:.4g},{y_val:.4g})"
                        fx = _safe_eval_function(fn, x_val)
                        if fx is None:
                            failures.append(label)
                            details[label] = {"ok": False, "msg": "function undefined"}
                            continue
                        dy = y_val - fx
                        ok_local = abs(dy) <= tol
                        if not ok_local:
                            failures.append(label)
                        details[label] = {"ok": ok_local, "dy": dy, "tol": tol}
                    geom_ok = not failures
                    ok = bool(geom_ok)
                    verification = {"points": details, "expr": expr, "tol": tol}
                    msg = f"verify points on {func_token}: ok={geom_ok}"
                elif verify_type == "DerivativeAt":
                    func_token = a.args[0]
                    fn, expr = _resolve_function_spec(VeriGraph, func_token)
                    if fn is None:
                        raise ValueError(f"VerifyFunction cannot parse function: {func_token}")
                    x_val = _parse_point_or_value(VeriGraph, a.args[1], "x", "x")
                    target = _author_parse_float(a.args[2], "derivative")
                    order = 1
                    tol = 1e-4
                    if len(a.args) >= 5:
                        order = _parse_derivative_order(a.args[3])
                        tol = _author_parse_float(a.args[4], "tol", default=1e-4)
                    elif len(a.args) == 4:
                        token = a.args[3]
                        if str(token).strip().lower().startswith("n="):
                            order = _parse_derivative_order(token)
                        else:
                            tol = _author_parse_float(token, "tol", default=1e-4)
                    deriv = _approx_nth_derivative(fn, x_val, order)
                    if deriv is None or not math.isfinite(deriv):
                        raise ValueError(f"VerifyFunction derivative cannot be computed: x={x_val}, n={order}")
                    diff = deriv - target
                    geom_ok = abs(diff) <= tol
                    ok = bool(geom_ok)
                    verification = {"expr": expr, "x": x_val, "order": order, "derivative": deriv, "target": target, "tol": tol}
                    msg = f"verify deriv n={order} at x={x_val}: diff={diff:.3g}, tol={tol:g}"
                else:  # IntegralOn
                    func_token = a.args[0]
                    fn, expr = _resolve_function_spec(VeriGraph, func_token)
                    if fn is None:
                        raise ValueError(f"VerifyFunction cannot parse function: {func_token}")
                    a_val = _parse_bound_token(VeriGraph, a.args[1], "x")
                    b_val = _parse_bound_token(VeriGraph, a.args[2], "x")
                    target = _author_parse_float(a.args[3], "integral")
                    tol = 1e-4
                    break_tokens: List[str] = []
                    rest = a.args[4:]
                    if rest:
                        head = str(rest[0]).strip().lower()
                        if head in ("breaks", "breakpoints"):
                            break_tokens = rest[1:]
                        else:
                            try:
                                tol = _author_parse_float(rest[0], "tol")
                                rest = rest[1:]
                            except Exception:
                                pass
                            if rest:
                                head2 = str(rest[0]).strip().lower()
                                if head2 in ("breaks", "breakpoints"):
                                    break_tokens = rest[1:]
                                else:
                                    break_tokens = rest
                    breaks: List[float] = []
                    for tok in break_tokens:
                        val = _parse_bound_token(VeriGraph, tok, "x")
                        if not math.isfinite(val):
                            raise ValueError("VerifyFunction integral breakpoints must be finite")
                        breaks.append(val)
                    integ = _approx_integral(fn, a_val, b_val, breaks=breaks)
                    if integ is None or not math.isfinite(integ):
                        raise ValueError(f"VerifyFunction integral cannot be computed: [{a_val}, {b_val}]")
                    diff = integ - target
                    geom_ok = abs(diff) <= tol
                    ok = bool(geom_ok)
                    verification = {"expr": expr, "a": a_val, "b": b_val, "integral": integ, "target": target, "tol": tol, "breaks": breaks}
                    msg = f"verify integral [{a_val},{b_val}]: diff={diff:.3g}, tol={tol:g}"
            else:
                raise ValueError(f"Unsupported op: {a.op}")

            if tracked_constraints:
                verification = _verify_constraint_snapshot(tracked_constraints)
                if verification:
                    vals = [info for info in verification.values() if isinstance(info, dict)]
                    bools = [info.get("ok") for info in vals if info.get("ok") is not None]
                    if bools:
                        geom_ok = all(bools)
                        if geom_ok is False and ok:
                            msg += " (numeric verification did not pass)"

        except Exception as e:
            ok, msg = False, f"{a.op} failed: {e}"
        finally:
            if verify_only and tracked_constraints:
                for cid in tracked_constraints:
                    try:
                        VeriGraph.constraints.pop(cid, None)
                    except Exception:
                        pass
        logs.append({
            "step": i,
            "op": a.op,
            "type": a.type,
            "args": a.args,
            "ok": ok,
            "msg": msg,
            "geom_ok": geom_ok,
            "verification": verification
        })
    # Author ??/????????????????
    post_step = len(actions) + 1
    if verify_only:
        try:
            VeriGraph.VeriSolve_incidence(ignore_protection=True)
            logs.append({
                "step": post_step,
                "op": "VeriSolve",
                "type": "Author",
                "args": [],
                "ok": True,
                "msg": "VeriSolve_incidence (verify_only mode) complete",
                "geom_ok": None,
                "verification": None,
            })
        except Exception as exc:
            logs.append({
                "step": post_step,
                "op": "VeriSolve",
                "type": "Author",
                "args": [],
                "ok": False,
                "msg": f"VeriSolve_incidence (verify_only) failed: {exc}",
                "geom_ok": None,
                "verification": None,
            })
        post_step += 1
        try:
            snap_stats = VeriGraph.snap_free_rays_first_hit()
            logs.append({
                "step": post_step,
                "op": "SnapFreeRays",
                "type": "Author",
                "args": [],
                "ok": True,
                "msg": f"Free-ray snap complete: snapped={snap_stats.get('snapped', 0)}, hidden={snap_stats.get('hidden', 0)}, total={snap_stats.get('total', 0)}",
                "geom_ok": None,
                "verification": None,
            })
        except Exception as exc:
            logs.append({
                "step": post_step,
                "op": "SnapFreeRays",
                "type": "Author",
                "args": [],
                "ok": False,
                "msg": f"Free-ray snap failed: {exc}",
                "geom_ok": None,
                "verification": None,
            })
    else:
        try:
            VeriGraph.VeriSolve_incidence(ignore_protection=True)
            logs.append({
                "step": post_step,
                "op": "VeriSolve",
                "type": "Author",
                "args": [],
                "ok": True,
                "msg": "?????????/??/???????????",
                "geom_ok": None,
                "verification": None,
            })
        except Exception as exc:
            logs.append({
                "step": post_step,
                "op": "VeriSolve",
                "type": "Author",
                "args": [],
                "ok": False,
                "msg": f"VeriSolve_incidence ??: {exc}",
                "geom_ok": None,
                "verification": None,
            })
        post_step += 1

        if CONFIG.author_enable_enforce:
            try:
                VeriGraph.polish_soft_constraints(ignore_protection=True)
                logs.append({
                    "step": post_step,
                    "op": "SoftPolish",
                    "type": "Author",
                    "args": [],
                    "ok": True,
                    "msg": "????? enforce?????",
                    "geom_ok": None,
                    "verification": None,
                })
            except Exception as exc:
                logs.append({
                    "step": post_step,
                    "op": "SoftPolish",
                    "type": "Author",
                    "args": [],
                    "ok": False,
                    "msg": f"soft polish ?? : {exc}",
                    "geom_ok": None,
                    "verification": None,
                })
            post_step += 1

        try:
            snap_stats = VeriGraph.snap_free_rays_first_hit()
            logs.append({
                "step": post_step,
                "op": "SnapFreeRays",
                "type": "Author",
                "args": [],
                "ok": True,
                "msg": f"Free-ray snap complete: snapped={snap_stats.get('snapped', 0)}, hidden={snap_stats.get('hidden', 0)}, total={snap_stats.get('total', 0)}",
                "geom_ok": None,
                "verification": None,
            })
        except Exception as exc:
            logs.append({
                "step": post_step,
                "op": "SnapFreeRays",
                "type": "Author",
                "args": [],
                "ok": False,
                "msg": f"Free-ray snap failed: {exc}",
                "geom_ok": None,
                "verification": None,
            })

    return logs
# ================================ Utilities ================================
def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:6]}"

def _dist(p, q) -> float:
    return math.hypot(p[0]-q[0], p[1]-q[1])

def _triangle_area(A, B, C) -> float:
    return abs(0.5 * ((B[0]-A[0])*(C[1]-A[1]) - (B[1]-A[1])*(C[0]-A[0])))

def _angle_deg(A, B, C) -> float:
    # ∠ABC
    v1 = (A[0]-B[0], A[1]-B[1])
    v2 = (C[0]-B[0], C[1]-B[1])
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    n1 = math.hypot(*v1); n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0: return float('nan')
    cosv = max(-1.0, min(1.0, dot/(n1*n2)))
    return math.degrees(math.acos(cosv))

def _normalize(vx, vy, eps=1e-12):
    n = math.hypot(vx, vy)
    if n < eps: return (0.0, 0.0)
    return (vx / n, vy / n)

def _bisector_intersection_with_BC(A, B, C, eps=1e-9):
    """
    Return E: intersection of A's interior angle bisector with line BC.
    Method: v = unit(AB) + unit(AC) as the interior bisector direction; find intersection of A + t v with B + s (C-B).
    If degenerate, fall back to the midpoint of BC.
    """
    ax, ay = A; bx, by = B; cx, cy = C
    ab = _normalize(bx - ax, by - ay)
    ac = _normalize(cx - ax, cy - ay)
    vx, vy = (ab[0] + ac[0], ab[1] + ac[1])
    # When near a straight angle, v may approach zero; degeneracy guard
    if abs(vx) < eps and abs(vy) < eps:
        vx, vy = (ab[0] - ac[0], ab[1] - ac[1])
        if abs(vx) < eps and abs(vy) < eps:
            return ((bx + cx)/2.0, (by + cy)/2.0)

    wx, wy = (cx - bx, cy - by)  # BC direction
    det = vx * (-wy) - (-wx) * vy  # v x w
    if abs(det) < eps:
        return ((bx + cx)/2.0, (by + cy)/2.0)
    t = ((bx - ax) * (-wy) - (-wx) * (by - ay)) / det
    ex, ey = (ax + t * vx, ay + t * vy)
    return (ex, ey)

def _circumcenter_from_points(A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float],
                              eps: float = 1e-9) -> Optional[Tuple[float, float]]:
    """
    Compute the circumcenter of three points A/B/C; return None if the three points are nearly collinear.
    """
    ax, ay = A; bx, by = B; cx, cy = C
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < eps:
        return None

    def _sq_sum(x, y):
        return x * x + y * y

    ux = ((_sq_sum(ax, ay) * (by - cy)) +
          (_sq_sum(bx, by) * (cy - ay)) +
          (_sq_sum(cx, cy) * (ay - by))) / d
    uy = ((_sq_sum(ax, ay) * (cx - bx)) +
          (_sq_sum(bx, by) * (ax - cx)) +
          (_sq_sum(cx, cy) * (bx - ax))) / d
    return (ux, uy)

def _circle_line_intersections(cx: float, cy: float, r: float,
                               p1: Tuple[float, float], p2: Tuple[float, float],
                               eps: float = 1e-9) -> List[Tuple[float, float]]:
    """???????? p1-p2 ????0/1/2 ???????????"""
    x1, y1 = p1; x2, y2 = p2
    dx = x2 - x1; dy = y2 - y1
    a = dx * dx + dy * dy
    if a < eps:
        return []
    fx = x1 - cx; fy = y1 - cy
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r
    disc = b * b - 4.0 * a * c
    if disc < -eps:
        return []
    disc = max(disc, 0.0)
    sqrt_disc = math.sqrt(disc)
    t1 = (-b + sqrt_disc) / (2.0 * a)
    intersections = [(x1 + t1 * dx, y1 + t1 * dy)]
    if sqrt_disc > eps:
        t2 = (-b - sqrt_disc) / (2.0 * a)
        intersections.append((x1 + t2 * dx, y1 + t2 * dy))
    return intersections

def _line_param(p1: Tuple[float, float], p2: Tuple[float, float],
                q: Tuple[float, float], eps: float = 1e-9) -> Optional[float]:
    """Return the parameter t (0<=t<=1) of q on segment p1-p2; return None if invalid"""
    x1, y1 = p1; x2, y2 = p2
    dx = x2 - x1; dy = y2 - y1
    denom = dx * dx + dy * dy
    if denom < eps:
        return None
    t = ((q[0] - x1) * dx + (q[1] - y1) * dy) / denom
    return t

def _verify_angle_bisector(A, B, C, E, tol_deg=1e-3) -> bool:
    """Check whether ∠BAE and ∠EAC are approximately equal (small numeric tolerance)"""
    return abs(_angle_deg(B, A, E) - _angle_deg(E, A, C)) <= tol_deg


class AnalyticSolver:
    """Analytic geometry solver: one-shot equations, no nudging."""

    def __init__(self, VeriGraph: "GeometryGraph", tol: float = 1e-6, ignore_protection: bool = False):
        import sympy as sp

        self.VeriGraph = VeriGraph
        self.tol = tol
        self.ignore_protection = ignore_protection
        self.sym = sp
        self.point_lookup = _build_point_lookup(self.VeriGraph)
        self.known: Dict[str, Tuple[float, float]] = {}
        self.symbols: Dict[str, Tuple[sp.Symbol, sp.Symbol]] = {}
        self.param_symbols: Dict[Tuple[str, str], sp.Symbol] = {}
        self.equations: List[Any] = []
        self.closed_form_results: Dict[str, Tuple[float, float]] = {}
        self.pending_closed_forms: List[Tuple[str, Tuple[float, float]]] = []
        self.point_usage: Dict[str, int] = {}
        self._collect_known_points()
        self._compute_point_usage()
        self._apply_closed_forms()
        self._build_symbols()
        self._materialize_pending_closed_forms()
        self._build_equations()

    # === public entry ===
    def solve(self):
        import concurrent.futures as cf

        def _attempt(solution_map: Optional[Dict[Any, Any]]) -> Optional[Dict[str, Tuple[float, float]]]:
            if solution_map is None:
                return None
            results = self._build_results_from_solution(solution_map)
            if results is None:
                return None
            if not self._check_residual(solution_map):
                return None
            return results

        timeout_sec = getattr(CONFIG, "analytic_timeout_seconds", 60.0)
        timed_out = False
        analytic_solution: Optional[Dict[Any, Any]] = None
        if timeout_sec is not None and timeout_sec > 0:
            executor = ThreadPoolExecutor(max_workers=1)
            fut = executor.submit(self._solve_equations)
            try:
                analytic_solution = fut.result(timeout=timeout_sec)
            except cf.TimeoutError:
                timed_out = True
                print(f"[ANALYTIC][TIMEOUT] Symbolic solve exceeded {timeout_sec}s; abandoning this problem")
            finally:
                executor.shutdown(wait=not timed_out)
        else:
            analytic_solution = self._solve_equations()

        if timed_out:
            return False

        # First try analytic solving; if it fails or yields degenerate solutions, fall back to numeric nsolve
        results = _attempt(analytic_solution)
        if results is None:
            results = _attempt(self._solve_equations_numeric())
        if results is None:
            return False

        self._commit(results)
        self._refresh_circles()
        self._fix_circle_line_second_intersection()
        return True

    def _build_results_from_solution(self, solution_map: Dict[Any, Any]) -> Optional[Dict[str, Tuple[float, float]]]:
        results: Dict[str, Tuple[float, float]] = {}
        results.update(self.closed_form_results)
        for pid, (sx, sy) in self.symbols.items():
            vx = solution_map.get(sx)
            vy = solution_map.get(sy)
            if vx is None or vy is None:
                return None
            try:
                results[pid] = (float(vx), float(vy))
            except Exception:
                return None
        return results

    def _list_unknown_symbols(self) -> List[Any]:
        seen = set()
        syms: List[Any] = []
        for sx, sy in self.symbols.values():
            for s in (sx, sy):
                if s not in seen:
                    syms.append(s); seen.add(s)
        for s in self.param_symbols.values():
            if s not in seen:
                syms.append(s); seen.add(s)
        return syms

    def _solve_equations_numeric(self, trials: int = 4) -> Optional[Dict[Any, Any]]:
        """Numeric fallback using nsolve to escape degenerate roots."""
        import random
        unknown_syms = self._list_unknown_symbols()
        if not unknown_syms:
            return {}
        exprs = []
        for eq in self.equations:
            try:
                diff = eq.lhs - eq.rhs if isinstance(eq, self.sym.Equality) else eq
            except Exception:
                diff = eq
            exprs.append(diff)
        if len(exprs) < len(unknown_syms):
            return None
        tol = max(self.tol, 1e-6)
        subset = exprs[:len(unknown_syms)]
        for _ in range(max(1, trials)):
            guess = [random.uniform(-1.5, 1.5) for _ in unknown_syms]
            try:
                sol_vec = self.sym.nsolve(subset, list(unknown_syms), guess, tol=tol, maxsteps=50)
            except Exception:
                continue
            try:
                vals = [float(v) for v in sol_vec]
            except Exception:
                continue
            return {sym: val for sym, val in zip(unknown_syms, vals)}
        return None

    def _is_degenerate_results(self, results: Dict[str, Tuple[float, float]]) -> bool:
        """Return True if points collapse or constraint segments degenerate to zero length."""
        eps = max(self.tol, 1e-6)
        coord_map: Dict[str, Tuple[float, float]] = {}
        coord_map.update(self.known)
        coord_map.update(self.closed_form_results)
        coord_map.update(results)

        ids = list(coord_map.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                dx = coord_map[ids[i]][0] - coord_map[ids[j]][0]
                dy = coord_map[ids[i]][1] - coord_map[ids[j]][1]
                if dx * dx + dy * dy < eps * eps:
                    return True

        def _len_sq(a: str, b: str) -> Optional[float]:
            pa = coord_map.get(a); pb = coord_map.get(b)
            if pa is None or pb is None:
                return None
            dx = pa[0] - pb[0]; dy = pa[1] - pb[1]
            return dx * dx + dy * dy

        for cons in self.VeriGraph.constraints.values():
            if cons.type not in ("Perpendicular", "Parallel", "EqualLength"):
                continue
            ends1 = self._segment_endpoints_safe(cons.scope[0])
            ends2 = self._segment_endpoints_safe(cons.scope[1]) if len(cons.scope) > 1 else None
            for ends in (ends1, ends2):
                if not ends or len(ends) < 2:
                    continue
                l2 = _len_sq(ends[0], ends[1])
                if l2 is not None and l2 < eps * eps:
                    return True
        return False

    # === initialization ===
    def _collect_known_points(self):
        for pid, ent in self.VeriGraph.entities.items():
            if ent.type != "Point":
                continue
            coord = ent.attrs.data.get("coord")
            if not isinstance(coord, dict):
                continue
            if (not self.ignore_protection) and self.VeriGraph._is_protected_point(pid):
                self.known[pid] = (float(coord.get("x", 0.0)), float(coord.get("y", 0.0)))
                continue
            prov = (ent.attrs.data.get("provenance") or "").lower()
            if prov in ("given", "problem", "derived", "aux"):
                self.known[pid] = (float(coord.get("x", 0.0)), float(coord.get("y", 0.0)))

    def _build_symbols(self):
        import sympy as sp

        self.param_symbols = {}
        for pid, ent in self.VeriGraph.entities.items():
            if ent.type != "Point":
                continue
            if pid in self.known or pid in self.closed_form_results:
                continue
            sx, sy = sp.symbols(f"x_{pid} y_{pid}", real=True)
            self.symbols[pid] = (sx, sy)
        for cons in self.VeriGraph.constraints.values():
            if cons.type != "PointOnSegment":
                continue
            if not cons.scope or len(cons.scope) < 2:
                continue
            pid, target = cons.scope[0], cons.scope[1]
            ratio_raw = None
            if isinstance(cons.params, dict):
                ratio_raw = cons.params.get("ratio")
            try:
                ratio_val = float(Fraction(str(ratio_raw))) if ratio_raw is not None else None
            except Exception:
                ratio_val = None
            if ratio_val is not None:
                continue  # fixed ratio, no free parameter needed
            key = (pid, target)
            if key in self.param_symbols:
                continue
            if pid in self.VeriGraph.entities and self.VeriGraph.entities[pid].type == "Point":
                name = self._sanitize_symbol_name(pid, target)
                self.param_symbols[key] = sp.symbols(name, real=True)

    # === helper: coordinate expression ===
    def _coord(self, pid: str):
        if pid in self.closed_form_results:
            return self.closed_form_results[pid]
        if pid in self.known:
            return self.known[pid]
        if pid in self.symbols:
            return self.symbols[pid]
        ent = self.VeriGraph.entities.get(pid)
        if ent and ent.type == "Point":
            coord = ent.attrs.data.get("coord")
            if coord is not None:
                return (float(coord.get("x", 0.0)), float(coord.get("y", 0.0)))
        return None

    # === helper: symbolic coordinates ===
    def _xy(self, pid: str):
        if pid in self.closed_form_results:
            x, y = self.closed_form_results[pid]
            return (self.sym.Float(x), self.sym.Float(y))
        if pid in self.known:
            x, y = self.known[pid]
            return (self.sym.Float(x), self.sym.Float(y))
        if pid in self.symbols:
            return self.symbols[pid]
        return None

    def _const_xy(self, pid: str) -> Optional[Tuple[float, float]]:
        if pid in self.closed_form_results:
            return self.closed_form_results[pid]
        if pid in self.known:
            return self.known[pid]
        return None

    # === closed form targets (????) ===
    def _is_numeric_point(self, pt: Optional[Tuple[Any, Any]]) -> bool:
        return isinstance(pt, tuple) and len(pt) == 2 and all(isinstance(v, (int, float)) for v in pt)

    def _compute_point_usage(self):
        """Roughly count point appearances in constraints to decide if locking coordinates is safe."""
        usage: Dict[str, int] = defaultdict(int)
        for cons in self.VeriGraph.constraints.values():
            pts: set = set()
            for tok in cons.scope:
                ent = self.VeriGraph.entities.get(tok)
                if ent and ent.type == "Point":
                    pts.add(tok)
                elif ent and ent.type == "Segment":
                    try:
                        ends = self._segment_endpoints_safe(tok)
                        for p in ends or []:
                            pts.add(p)
                    except Exception:
                        pass
                elif ent and ent.type == "Angle":
                    name = ent.attrs.data.get("name")
                    if isinstance(name, str) and len(name) >= 4 and name[0] == "∠":
                        pts.update([name[1], name[2], name[3]])
            for p in pts:
                usage[p] += 1
        self.point_usage = usage

    def _record_closed_form(self, pid: str, xy: Tuple[float, float]):
        """Directly assign coordinates to closed-form result; skip if point already has coordinates."""
        if pid in self.known or pid in self.closed_form_results:
            return
        self.closed_form_results[pid] = (float(xy[0]), float(xy[1]))

    def _apply_closed_forms(self):
        for cons in self.VeriGraph.constraints.values():
            if cons.type == "Midpoint" and len(cons.scope) >= 2:
                mid = cons.scope[0]
                if mid in self.closed_form_results or mid in self.known:
                    continue
                if len(cons.scope) >= 3:
                    ends = (cons.scope[1], cons.scope[2])
                else:
                    ends = self._segment_endpoints_safe(cons.scope[1])
                if not ends:
                    continue
                u, v = ends
                U = self._const_xy(u); V = self._const_xy(v)
                if U and V:
                    self._record_closed_form(mid, ((U[0] + V[0]) / 2.0, (U[1] + V[1]) / 2.0))

            if cons.type == "PointOnSegment" and len(cons.scope) >= 2:
                pid, seg_id = cons.scope[0], cons.scope[1]
                if pid in self.closed_form_results or pid in self.known:
                    continue
                ratio_raw = None
                if isinstance(cons.params, dict):
                    ratio_raw = cons.params.get("ratio")
                try:
                    ratio = float(Fraction(str(ratio_raw))) if ratio_raw is not None else None
                except Exception:
                    ratio = None
                if ratio is None:
                    continue
                ends = self._segment_endpoints_safe(seg_id)
                if not ends:
                    continue
                a, b = ends
                A = self._const_xy(a); B = self._const_xy(b)
                if A and B:
                    x = A[0] + ratio * (B[0] - A[0])
                    y = A[1] + ratio * (B[1] - A[1])
                    self._record_closed_form(pid, (float(x), float(y)))
        # Special configuration completion: when triangle ABC is known, find E = circumcircle ∩ angle bisector, the incenter I, and altitude-foot points N/P
        def _known(pid):
            return self._const_xy(pid)

        def _triangle_known():
            return _known("A") and _known("B") and _known("C")

        if _triangle_known():
            A_xy = _known("A"); B_xy = _known("B"); C_xy = _known("C")
            # Fill E: on A's angle bisector, intersecting circumcircle at E (other than A); unconditionally use triangle ABC
            if "E" not in self.known and "E" not in self.closed_form_results and self.point_usage.get("E", 0) == 0:
                center = _circumcenter_from_points(A_xy, B_xy, C_xy)
                if center:
                    cx, cy = center
                    r = math.hypot(cx - A_xy[0], cy - A_xy[1])
                    bax, bay = B_xy[0] - A_xy[0], B_xy[1] - A_xy[1]
                    cax, cay = C_xy[0] - A_xy[0], C_xy[1] - A_xy[1]
                    def _unit(x,y):
                        n=math.hypot(x,y)
                        return (x/n, y/n) if n>1e-9 else None
                    u1=_unit(bax,bay); u2=_unit(cax,cay)
                    if u1 and u2:
                        dirx, diry = u1[0]+u2[0], u1[1]+u2[1]
                        nlen=math.hypot(dirx,diry)
                        if nlen>1e-9:
                            dirx/=nlen; diry/=nlen
                            Ax,Ay=A_xy; dx,dy=dirx,diry
                            fx,fy=Ax-cx, Ay-cy
                            a=dx*dx+dy*dy; b=2*(fx*dx+fy*dy); c=fx*fx+fy*fy - r*r
                            disc=b*b-4*a*c
                            if disc>=-1e-9:
                                disc=max(disc,0.0); sd=math.sqrt(disc)
                                t1=(-b+sd)/(2*a); t2=(-b-sd)/(2*a)
                                t_candidates=[t for t in (t1,t2) if abs(t)>1e-6]
                                if t_candidates:
                                    t=max(t_candidates, key=lambda t: abs(t))
                                    ex=Ax+ t*dx; ey=Ay+ t*dy
                                    self._record_closed_form("E",(float(ex),float(ey)))
            # Fill I: triangle incenter
            if "I" not in self.known and "I" not in self.closed_form_results and self.point_usage.get("I", 0) == 0:
                ax,ay=A_xy; bx,by=B_xy; cx,cy=C_xy
                len_a=math.hypot(bx-cx, by-cy)
                len_b=math.hypot(ax-cx, ay-cy)
                len_c=math.hypot(ax-bx, ay-by)
                px=(len_a*ax + len_b*bx + len_c*cx)/(len_a+len_b+len_c)
                py=(len_a*ay + len_b*by + len_c*cy)/(len_a+len_b+len_c)
                self._record_closed_form("I",(float(px),float(py)))
            # Fill foot points N, P: defined as the right-angle points on the circles with diameters EB/EC, with fixed direction to avoid coincidence
            def _thales_point(p1, p2, target):
                if target in self.closed_form_results or target in self.known:
                    return
                if self.point_usage.get(target, 0) > 0:
                    return
                if not (_known(p1) and _known(p2)):
                    return
                A=_known(p1); B=_known(p2)
                mx,my=(A[0]+B[0])/2.0,(A[1]+B[1])/2.0
                vx,vy=B[0]-A[0],B[1]-A[1]
                norm=math.hypot(vx,vy)
                if norm<1e-9:
                    return
                r=norm/2.0
                nx,ny= -vy/norm, vx/norm
                # Choose the direction that makes the cross product with A positive, to avoid coincidence
                tx,ty=mx + r*nx, my + r*ny
                if (vx*(-r*ny) - vy*(r*nx)) < 0:  # cross sign
                    tx,ty=mx - r*nx, my - r*ny
                self._record_closed_form(target,(float(tx),float(ty)))
            _thales_point("E","B","N")
            _thales_point("E","C","P")
        self._closed_form_circle_line_second_intersection()
        self._closed_form_parallel_single_unknown()

    def _materialize_pending_closed_forms(self):
        """Convert unlocked closed-form results to equality constraints rather than direct coordinate writes."""
        for pid, (x, y) in self.pending_closed_forms:
            P = self._sym_point(pid)
            if not P:
                continue
            px, py = P
            self.equations.append(px - self.sym.Float(x))
            self.equations.append(py - self.sym.Float(y))

    def _closed_form_circle_line_second_intersection(self):
        from collections import defaultdict

        eps = max(self.tol, 1e-9)
        collinear_map: Dict[str, List[Tuple["Constraint", Tuple[str, ...]]]] = defaultdict(list)
        for cons in self.VeriGraph.constraints.values():
            if cons.type != "Collinear":
                continue
            pts = tuple(pid for pid in cons.scope if (pid in self.VeriGraph.entities and self.VeriGraph.entities[pid].type == "Point"))
            if len(pts) < 3:
                continue
            for pid in pts:
                collinear_map[pid].append((cons, pts))

        for cons in self.VeriGraph.constraints.values():
            if cons.type != "Concyclic":
                continue
            pts = [pid for pid in cons.scope if pid in self.VeriGraph.entities and self.VeriGraph.entities[pid].type == "Point"]
            if len(pts) != 4:
                continue
            unknown = [p for p in pts if p not in self.known and p not in self.closed_form_results]
            if len(unknown) != 1:
                continue
            target = unknown[0]
            base = [p for p in pts if p != target]
            coords = [self._const_xy(pid) for pid in base]
            if any(c is None for c in coords):
                continue
            A_xy, B_xy, C_xy = coords  # type: ignore
            center = _circumcenter_from_points(A_xy, B_xy, C_xy, eps=eps)
            if center is None:
                continue
            cx, cy = center
            r = math.hypot(cx - A_xy[0], cy - A_xy[1])
            if not math.isfinite(r) or r < eps:
                continue
            anchor_id = None
            anchor_xy = None
            line_point_id = None
            line_xy = None
            base_coord_map = {pid: coord for pid, coord in zip(base, coords)}
            for _col_cons, pts_tuple in collinear_map.get(target, []):
                pts_set = set(pts_tuple)
                base_hits = [pid for pid in base_coord_map if pid in pts_set]
                if not base_hits:
                    continue
                for anchor in base_hits:
                    for other in pts_tuple:
                        if other == target or other == anchor:
                            continue
                        other_xy = self._const_xy(other)
                        if other_xy is None:
                            continue
                        anchor_id = anchor
                        anchor_xy = base_coord_map[anchor]
                        line_point_id = other
                        line_xy = other_xy
                        break
                    if anchor_id:
                        break
                if anchor_id:
                    break
            if not (anchor_xy and line_xy):
                continue
            inters = _circle_line_intersections(cx, cy, r, anchor_xy, line_xy, eps=eps)
            if not inters:
                continue
            best = None
            best_dist = 0.0
            for pt in inters:
                dist = (pt[0] - anchor_xy[0]) ** 2 + (pt[1] - anchor_xy[1]) ** 2
                if dist < eps * eps:
                    continue
                if best is None or dist > best_dist:
                    best = pt
                    best_dist = dist
            if best is None:
                continue
            self._record_closed_form(target, (float(best[0]), float(best[1])))

    def _closed_form_parallel_single_unknown(self):
        eps = max(self.tol, 1e-9)
        for cons in self.VeriGraph.constraints.values():
            if cons.type != "Parallel":
                continue
            if len(cons.scope) < 2:
                continue
            seg_tokens = cons.scope[:2]
            seg_data: List[List[Tuple[str, Optional[Tuple[float, float]]]]] = []
            valid = True
            for token in seg_tokens:
                ends = self._segment_endpoints_safe(token)
                if not ends:
                    valid = False
                    break
                info = []
                for pid in ends:
                    coord = self._const_xy(pid)
                    info.append((pid, coord))
                seg_data.append(info)
            if not valid or len(seg_data) != 2:
                continue
            ref_segment = None
            target_segment = None
            for info in seg_data:
                known = [item for item in info if item[1] is not None]
                unknown = [item for item in info if item[1] is None and item[0] not in self.closed_form_results]
                if len(known) == 2 and ref_segment is None:
                    ref_segment = known
                elif len(known) == 1 and len(unknown) == 1 and target_segment is None:
                    target_segment = (known[0], unknown[0])
            if not ref_segment or not target_segment:
                continue
            (a_id, A_xy), (b_id, B_xy) = ref_segment
            (anchor_id, anchor_xy), (unknown_id, _) = target_segment
            if anchor_xy is None or unknown_id in self.closed_form_results:
                continue

            # If the unknown point is involved in multiple constraints (appears directly in scope or via endpoints), skip closed-form and leave it to the global equation solver
            def _appears_in_constraint(cons):
                if unknown_id in cons.scope:
                    return True
                for tok in cons.scope:
                    ends = self._segment_endpoints_safe(tok)
                    if ends and unknown_id in ends:
                        return True
                return False

            involved = [c for c in self.VeriGraph.constraints.values() if _appears_in_constraint(c)]
            if len(involved) > 1:
                continue

            dx = B_xy[0] - A_xy[0]
            dy = B_xy[1] - A_xy[1]
            if abs(dx) < eps and abs(dy) < eps:
                continue
            new_x = anchor_xy[0] + dx
            new_y = anchor_xy[1] + dy
            self._record_closed_form(unknown_id, (float(new_x), float(new_y)))

    # === equation construction ===
    def _build_equations(self):
        handlers = {
            "Collinear": self._eq_collinear,
            "Parallel": self._eq_parallel,
            "Perpendicular": self._eq_perpendicular,
            "EqualLength": self._eq_equal_length,
            "EqualAngle": self._eq_equal_angle,
            "AngleBisector": self._eq_angle_bisector,
            "VectorDotZero": self._eq_vector_dot_zero,
            "Midpoint": self._eq_midpoint,
            "PointOnLine": self._eq_point_on_line,
            "PointOnSegment": self._eq_point_on_segment,
            "OnCircle": self._eq_on_circle,
            "PointOnCircle": self._eq_on_circle,
            "Concyclic": self._eq_concyclic,
            "Incidence": self._eq_incidence,
        }
        for cons in self.VeriGraph.constraints.values():
            handler = handlers.get(cons.type)
            if handler:
                handler(cons)
        self._add_gauge_constraints()

    # === solving ===
    def _solve_equations(self) -> Optional[Dict[Any, Any]]:
        import sympy as sp

        unknown_syms: List[Any] = []
        seen_syms = set()
        for sx, sy in self.symbols.values():
            for sym in (sx, sy):
                if sym not in seen_syms:
                    unknown_syms.append(sym)
                    seen_syms.add(sym)
        for sym in self.param_symbols.values():
            if sym not in seen_syms:
                unknown_syms.append(sym)
                seen_syms.add(sym)
        if not unknown_syms:
            return {}

        eqs = [eq if isinstance(eq, sp.Equality) else sp.Eq(eq, 0) for eq in self.equations]

        def _as_real_float(expr):
            try:
                evaluated = self.sym.N(expr)
            except Exception:
                return None
            if hasattr(evaluated, "as_real_imag"):
                real_part, imag_part = evaluated.as_real_imag()
                try:
                    imag_val = float(imag_part)
                except Exception:
                    return None
                if abs(imag_val) > self.tol:
                    return None
                try:
                    return float(real_part)
                except Exception:
                    return None
            try:
                return float(evaluated)
            except Exception:
                return None

        def _equation_syms(eq_item):
            diff = eq_item.lhs - eq_item.rhs if isinstance(eq_item, sp.Equality) else eq_item
            return list(diff.free_symbols)

        cleaned: List[Any] = []
        for eq in eqs:
            diff = eq.lhs - eq.rhs if isinstance(eq, sp.Equality) else eq
            syms = diff.free_symbols
            if not syms:
                num = _as_real_float(diff)
                if num is None:
                    continue
                if abs(num) > self.tol:
                    return None
                continue
            cleaned.append(eq)
        if not cleaned:
            return None

        def _is_trivial_zero(eq_item):
            diff = eq_item.lhs - eq_item.rhs if isinstance(eq_item, sp.Equality) else eq_item
            try:
                simplified = self.sym.simplify(diff)
            except Exception:
                simplified = diff
            if simplified == 0:
                return True
            if getattr(simplified, "free_symbols", None):
                return False
            num = _as_real_float(simplified)
            return num is not None and abs(num) <= self.tol

        eqs = [eq for eq in cleaned if not _is_trivial_zero(eq)]
        if not eqs:
            return None

        def _split_components(eq_list: List[Any]):
            parent: Dict[Any, Any] = {}

            def find(x):
                parent.setdefault(x, x)
                if parent[x] != x:
                    parent[x] = find(parent[x])
                return parent[x]

            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

            for eq in eq_list:
                syms = _equation_syms(eq)
                if not syms:
                    continue
                base = syms[0]
                for s in syms[1:]:
                    union(base, s)

            comp_syms: Dict[Any, set] = defaultdict(set)
            comp_eqs: Dict[Any, List[Any]] = defaultdict(list)
            for eq in eq_list:
                syms = _equation_syms(eq)
                if not syms:
                    continue
                root = find(syms[0])
                comp_eqs[root].append(eq)
                for s in syms:
                    comp_syms[root].add(s)
            return [(list(comp_syms[k]), comp_eqs[k]) for k in comp_eqs]

        def _estimate_component_rank(eq_list: List[Any], sym_list: List[Any], trials: int = 4) -> int:
            if not eq_list or not sym_list:
                return min(len(eq_list), len(sym_list))
            best = 0
            for _ in range(max(1, trials)):
                subs: Dict[Any, Any] = {}
                for s in sym_list:
                    try:
                        subs[s] = self.sym.Float(random.uniform(-2.0, 2.0))
                    except Exception:
                        subs[s] = 1.0
                rows: List[List[float]] = []
                for eq_item in eq_list:
                    diff = eq_item.lhs - eq_item.rhs if isinstance(eq_item, sp.Equality) else eq_item
                    grad: List[float] = []
                    for sym in sym_list:
                        try:
                            val = self.sym.diff(diff, sym).subs(subs)
                            grad.append(float(val))
                        except Exception:
                            grad.append(0.0)
                    rows.append(grad)
                try:
                    r = int(np.linalg.matrix_rank(np.array(rows, dtype=float)))
                except Exception:
                    r = 0
                best = max(best, r)
                if best >= len(sym_list):
                    break
            best = min(best, len(sym_list), len(eq_list))
            if best == 0 and eq_list:
                best = min(len(sym_list), len(eq_list))
            return best

        def _select_independent_equations(eq_list: List[Any], sym_list: List[Any], target_rank: int) -> List[Any]:
            if target_rank <= 0 or not eq_list:
                return []
            target = min(target_rank, len(eq_list), max(1, len(sym_list)))
            best_sel: List[Any] = []
            for _ in range(max(3, target + 1)):
                subs: Dict[Any, Any] = {}
                for s in sym_list:
                    try:
                        subs[s] = self.sym.Float(random.uniform(-2.0, 2.0))
                    except Exception:
                        subs[s] = 1.0
                rows: List[List[float]] = []
                for eq_item in eq_list:
                    diff = eq_item.lhs - eq_item.rhs if isinstance(eq_item, sp.Equality) else eq_item
                    grad: List[float] = []
                    for sym in sym_list:
                        try:
                            val = self.sym.diff(diff, sym).subs(subs)
                            grad.append(float(val))
                        except Exception:
                            grad.append(0.0)
                    rows.append(grad)

                selected: List[Any] = []
                mat: List[List[float]] = []
                for idx, row in enumerate(rows):
                    new_mat = mat + [row]
                    try:
                        new_rank = int(np.linalg.matrix_rank(np.array(new_mat, dtype=float)))
                        old_rank = int(np.linalg.matrix_rank(np.array(mat, dtype=float))) if mat else 0
                    except Exception:
                        new_rank = len(mat)
                        old_rank = len(mat)
                    if new_rank > old_rank:
                        selected.append(eq_list[idx])
                        mat = new_mat
                    if len(mat) >= target:
                        break
                if len(selected) > len(best_sel):
                    best_sel = selected
                if len(best_sel) >= target:
                    break
            return best_sel if best_sel else eq_list[:target]

        def _residual_ok(eq_list: List[Any], cand: Dict[Any, Any]) -> bool:
            thresh = max(self.tol, 1e-6)
            for eq in eq_list:
                diff = eq.lhs - eq.rhs if isinstance(eq, sp.Equality) else eq
                try:
                    val = diff.subs(cand)
                    num = _as_real_float(val)
                except Exception:
                    return False
                if num is None or abs(num) > thresh:
                    return False
            return True

        def _solve_linear_component(sym_list: List[Any], eq_list: List[Any]) -> Optional[Dict[Any, Any]]:
            exprs = []
            for eq in eq_list:
                diff = eq.lhs - eq.rhs if isinstance(eq, sp.Equality) else eq
                try:
                    poly = self.sym.Poly(diff, sym_list)
                    if poly.total_degree() > 1:
                        return None
                except Exception:
                    return None
                exprs.append(diff)
            try:
                M, b = self.sym.linear_eq_to_matrix(exprs, sym_list)
                sol_set = self.sym.linsolve((M, b), sym_list)
            except Exception:
                return None
            if not sol_set:
                return None
            sol_tuple = next(iter(sol_set))
            if any(getattr(v, "free_symbols", None) for v in sol_tuple):
                return None
            return {s: v for s, v in zip(sym_list, sol_tuple)}

        def _solve_component(sym_list: List[Any], eq_list: List[Any]) -> Optional[Dict[Any, Any]]:
            # insurance fuse
            max_u = getattr(CONFIG, "analytic_max_unknowns", None)
            max_e = getattr(CONFIG, "analytic_max_equations", None)
            if max_u is not None and len(sym_list) > max_u:
                return None
            if max_e is not None and len(eq_list) > max_e:
                return None

            # If under-constrained, deterministically fix some variables to constants and solve the remaining ones, to avoid hard-coded fills producing nonzero residuals
            if len(eq_list) < len(sym_list):
                fill_const = getattr(CONFIG, "analytic_fill_constant", 0.0)
                gap = len(sym_list) - len(eq_list)
                # Pick variables to fix in ascending order of frequency (favor under-constrained points)
                freq = {s: 0 for s in sym_list}
                for eq in eq_list:
                    diff = eq.lhs - eq.rhs if isinstance(eq, self.sym.Equality) else eq
                    for s in diff.free_symbols:
                        if s in freq:
                            freq[s] += 1
                # Try to use the common point of perpendicular constraints to construct a right-angle point on a "diameter circle" (deterministic, non-coincident)
                point_ids = set()
                for s in sym_list:
                    name = str(s)
                    if "_" in name:
                        point_ids.add(name.split("_", 1)[1])
                assign_perp: Dict[Any, Any] = {}
                for cons in self.VeriGraph.constraints.values():
                    if cons.type != "Perpendicular":
                        continue
                    if len(cons.scope) < 2:
                        continue
                    ends1 = self._segment_endpoints_safe(cons.scope[0])
                    ends2 = self._segment_endpoints_safe(cons.scope[1])
                    if not (ends1 and ends2):
                        continue
                    common = set(ends1).intersection(ends2)
                    if len(common) != 1:
                        continue
                    u = common.pop()
                    if u not in point_ids:
                        continue
                    others = [p for p in ends1 + ends2 if p != u]
                    if len(others) < 2:
                        continue
                    pA = self._const_xy(others[0])
                    pB = self._const_xy(others[1])
                    if not (pA and pB):
                        continue
                    mx, my = (pA[0] + pB[0]) / 2.0, (pA[1] + pB[1]) / 2.0
                    vx, vy = pB[0] - pA[0], pB[1] - pA[1]
                    norm = (vx * vx + vy * vy) ** 0.5
                    if norm < max(self.tol, 1e-9):
                        continue
                    r = norm / 2.0
                    nx, ny = -vy / norm, vx / norm  # unit normal
                    ux, uy = mx + r * nx, my + r * ny  # choose fixed direction to avoid coincidence
                    sx = next((s for s in sym_list if str(s) == f"x_{u}"), None)
                    sy = next((s for s in sym_list if str(s) == f"y_{u}"), None)
                    if sx:
                        assign_perp[sx] = ux
                    if sy:
                        assign_perp[sy] = uy
                if assign_perp:
                    reduced_eqs = []
                    for eq in eq_list:
                        diff = eq.lhs - eq.rhs if isinstance(eq, self.sym.Equality) else eq
                        try:
                            diff = diff.subs(assign_perp)
                        except Exception:
                            pass
                        reduced_eqs.append(diff)
                    reduced_syms = [s for s in sym_list if s not in assign_perp]
                    partial = _solve_component(reduced_syms, reduced_eqs) if reduced_syms else {}
                    if partial is not None:
                        cand_solution = dict(assign_perp)
                        cand_solution.update(partial)
                        if _residual_ok(eq_list, cand_solution):
                            return cand_solution
                # Try to extract a deterministic solution from constants in the equations: for points appearing least often and containing obvious constants, place them directly at the constant coordinates
                point_syms: Dict[str, Dict[str, Any]] = {}
                for s in sym_list:
                    name = str(s)
                    if "_" in name:
                        axis, pid = name.split("_", 1)
                        point_syms.setdefault(pid, {})[axis] = s
                assign_const: Dict[Any, Any] = {}
                for pid, axes in point_syms.items():
                    sx = axes.get("x"); sy = axes.get("y")
                    if not (sx and sy):
                        continue
                    fcnt = freq.get(sx, 0) + freq.get(sy, 0)
                    if fcnt > 2:
                        continue  # only do constant extraction for points with few appearances
                    consts = set()
                    for eq in eq_list:
                        diff = eq.lhs - eq.rhs if isinstance(eq, self.sym.Equality) else eq
                        if sx in diff.free_symbols or sy in diff.free_symbols:
                            for c in diff.atoms(self.sym.Number):
                                if c.is_real:
                                    try:
                                        consts.add(float(c))
                                    except Exception:
                                        pass
                    if len(consts) >= 2:
                        consts_sorted = sorted(consts)
                        assign_const[sx] = consts_sorted[0]
                        assign_const[sy] = consts_sorted[1]
                if assign_const:
                    reduced_eqs = []
                    for eq in eq_list:
                        diff = eq.lhs - eq.rhs if isinstance(eq, self.sym.Equality) else eq
                        try:
                            diff = diff.subs(assign_const)
                        except Exception:
                            pass
                        reduced_eqs.append(diff)
                    reduced_syms = [s for s in sym_list if s not in assign_const]
                    partial = _solve_component(reduced_syms, reduced_eqs) if reduced_syms else {}
                    if partial is not None:
                        cand_solution = dict(assign_const)
                        cand_solution.update(partial)
                        if _residual_ok(eq_list, cand_solution):
                            return cand_solution
                param_vars = sorted(sym_list, key=lambda s: (freq.get(s, 0), str(s)))[:gap]
                candidate_vals = [fill_const, fill_const + 1.0, fill_const - 1.0, fill_const + 2.0]

                from itertools import product

                def _with_perp_completion(solution: Dict[Any, Any]) -> Dict[Any, Any]:
                    """If a perpendicular constraint exists with an unknown common vertex and both other endpoints already have coordinates, place that point at the other end of the diameter circle to avoid coincidence."""
                    augmented = dict(solution)
                    point_lookup = _build_point_lookup(self.VeriGraph)
                    def _sym_xy(pid: str):
                        sx = next((s for s in sym_list if str(s) == f"x_{pid}"), None)
                        sy = next((s for s in sym_list if str(s) == f"y_{pid}"), None)
                        if sx in augmented and sy in augmented:
                            return (augmented[sx], augmented[sy])
                        try:
                            return self.VeriGraph.get_point_xy(pid)
                        except Exception:
                            return None

                    for cons in self.VeriGraph.constraints.values():
                        if cons.type != "Perpendicular" or len(cons.scope) < 2:
                            continue
                        ends1 = self._segment_endpoints_safe(cons.scope[0])
                        ends2 = self._segment_endpoints_safe(cons.scope[1])
                        if not (ends1 and ends2):
                            continue
                        common = set(ends1).intersection(ends2)
                        if len(common) != 1:
                            continue
                        u = common.pop()
                        sx = next((s for s in sym_list if str(s) == f"x_{u}"), None)
                        sy = next((s for s in sym_list if str(s) == f"y_{u}"), None)
                        if sx not in sym_list and sy not in sym_list:
                            continue
                        if sx in augmented and sy in augmented:
                            continue
                        others = [p for p in ends1 + ends2 if p != u]
                        if len(others) < 2:
                            continue
                        pA = _sym_xy(others[0])
                        pB = _sym_xy(others[1])
                        if not (pA and pB):
                            continue
                        mx, my = (pA[0] + pB[0]) / 2.0, (pA[1] + pB[1]) / 2.0
                        vx, vy = pB[0] - pA[0], pB[1] - pA[1]
                        norm = (vx * vx + vy * vy) ** 0.5
                        if norm < max(self.tol, 1e-9):
                            continue
                        r = norm / 2.0
                        nx, ny = -vy / norm, vx / norm
                        ux, uy = mx + r * nx, my + r * ny  # deterministic direction
                        if sx in sym_list:
                            augmented[sx] = ux
                        if sy in sym_list:
                            augmented[sy] = uy
                    return augmented

                for combo in product(candidate_vals, repeat=len(param_vars)):
                    assign = {sym: val for sym, val in zip(param_vars, combo)}
                    sub_eqs: List[Any] = []
                    for eq in eq_list:
                        diff = eq.lhs - eq.rhs if isinstance(eq, self.sym.Equality) else eq
                        try:
                            diff = diff.subs(assign)
                        except Exception:
                            pass
                        sub_eqs.append(diff)
                    sub_syms = [s for s in sym_list if s not in assign]
                    # Recursively solve remaining variables
                    partial = _solve_component(sub_syms, sub_eqs) if sub_syms else {}
                    if partial is None:
                        continue
                    cand_solution = dict(assign)
                    cand_solution.update(partial)
                    cand_solution = _with_perp_completion(cand_solution)
                    # Substitute the recursive result back into the original equations to verify residuals
                    if _residual_ok(eq_list, cand_solution):
                        return cand_solution
                # No feasible solution found; subsequent code will continue trying linear/symbolic solving (allowing free parameters), then decide whether to give up

            linear_solution = _solve_linear_component(sym_list, eq_list)
            if linear_solution is not None:
                return linear_solution
            try:
                sol = self.sym.solve(eq_list, sym_list, dict=True, simplify=False)
            except Exception:
                return None
            fill_const = getattr(CONFIG, "analytic_fill_constant", 0.0)
            # Assign distinct fill values to free symbols of the same component to avoid all overlapping
            fill_pool: List[float] = []
            step = 0.5 if abs(fill_const) < 1e-9 else fill_const
            for i in range(max(8, len(sym_list) + 4)):
                fill_pool.append(fill_const + (i + 1) * step)

            def _fill_for_symbol(sym_obj: Any) -> float:
                try:
                    name = str(sym_obj)
                    h = int(hashlib.sha256(name.encode("utf-8")).hexdigest(), 16)
                    idx = h % len(fill_pool)
                    return float(fill_pool[idx])
                except Exception:
                    return float(fill_const)

            for cand in sol:
                # If the solution contains free parameters, fix them with fill_const and verify residuals
                sub_map = {}
                for val in cand.values():
                    try:
                        for fs in getattr(val, "free_symbols", []) or []:
                            sub_map[fs] = _fill_for_symbol(fs)
                    except Exception:
                        pass
                filled: Dict[Any, Any] = {}
                for sym in sym_list:
                    val = cand.get(sym, _fill_for_symbol(sym))
                    try:
                        if sub_map:
                            val = val.subs(sub_map)
                    except Exception:
                        pass
                    num = _as_real_float(val)
                    if num is None:
                        filled = None  # type: ignore
                        break
                    filled[sym] = num
                if filled is None:
                    continue
                if _residual_ok(eq_list, filled):
                    return filled
            if getattr(CONFIG, "analytic_fill_free_params", False):
                # First try a partial solve: fix the other variables and solve one variable
                for anchor in sym_list:
                    try:
                        partial_solutions = self.sym.solve(eq_list, [anchor], dict=True, simplify=False)
                    except Exception:
                        partial_solutions = []
                    for cand in partial_solutions:
                        filled = {sym: _fill_for_symbol(sym) for sym in sym_list}
                        val = cand.get(anchor, _fill_for_symbol(anchor))
                        try:
                            for fs in getattr(val, "free_symbols", []) or []:
                                val = val.subs(fs, _fill_for_symbol(fs))
                        except Exception:
                            pass
                        num = _as_real_float(val)
                        if num is None:
                            continue
                        filled[anchor] = num
                        if _residual_ok(eq_list, filled):
                            return filled
                # Then fall back to filling all values
                filled = {sym: _fill_for_symbol(sym) for sym in sym_list}
                if _residual_ok(eq_list, filled):
                    return filled
            return None

        solution_map: Dict[Any, Any] = {}
        fill_const = getattr(CONFIG, "analytic_fill_constant", 0.0)
        for syms_block, eq_block in _split_components(eqs):
            if not syms_block:
                continue
            target_rank = _estimate_component_rank(eq_block, syms_block)
            filtered_eqs = _select_independent_equations(eq_block, syms_block, target_rank)
            solved = _solve_component(syms_block, filtered_eqs)
            if solved is None and filtered_eqs != eq_block:
                solved = _solve_component(syms_block, eq_block)
            if solved is None:
                return None
            solution_map.update(solved)
        for sym in unknown_syms:
            if sym not in solution_map:
                solution_map[sym] = fill_const
        return solution_map

    def _check_residual(self, solution_map: Dict[Any, Any]) -> bool:
        subs = dict(solution_map)
        thresh = max(self.tol, 1e-6)
        for eq in self.equations:
            if isinstance(eq, self.sym.Equality):
                val = (eq.lhs - eq.rhs).subs(subs)
            else:
                val = eq.subs(subs)
            try:
                val_num = self.sym.N(val)
            except Exception:
                return False
            if hasattr(val_num, "free_symbols") and val_num.free_symbols:
                return False
            try:
                if abs(float(val_num)) > thresh:
                    return False
            except Exception:
                return False
        return True

    # === commit (single write per point) ===
    def _commit(self, results: Dict[str, Tuple[float, float]]):
        locked = set(self.VeriGraph.meta.get("hard_locked_points") or [])
        for pid, xy in results.items():
            if pid in self.known:
                continue
            if not self.ignore_protection and self.VeriGraph._is_protected_point(pid):
                continue
            if pid not in self.VeriGraph.entities or self.VeriGraph.entities[pid].type != "Point":
                continue
            ent = self.VeriGraph.entities[pid]
            ent.attrs.data.setdefault("coord", {"x": 0.0, "y": 0.0})
            ent.attrs.data["coord"]["x"] = float(xy[0])
            ent.attrs.data["coord"]["y"] = float(xy[1])
            locked.add(pid)
            self._refresh_incident_entities(pid)
        self.VeriGraph.meta["hard_locked_points"] = sorted(locked)

    # === utility helpers ===
    def _refresh_circles(self):
        """Sync circle radii based on current center/through points."""
        for cid, ent in self.VeriGraph.entities.items():
            if ent.type != "Circle":
                continue
            center_id = ent.attrs.data.get("center_id") or ent.attrs.data.get("center")
            through_id = ent.attrs.data.get("through_id")
            if not (center_id and through_id):
                continue
            cxy = self.VeriGraph.get_point_xy(center_id)
            txy = self.VeriGraph.get_point_xy(through_id)
            if not (cxy and txy):
                continue
            dx = txy[0] - cxy[0]
            dy = txy[1] - cxy[1]
            ent.attrs.data["radius"] = (dx * dx + dy * dy) ** 0.5

    def _fix_circle_line_second_intersection(self):
        """Adjust points stuck at anchor when constrained by circle + collinear."""
        from collections import defaultdict

        eps = max(self.tol, 1e-6)

        def _point_xy(pid: str) -> Optional[Tuple[float, float]]:
            try:
                return self.VeriGraph.get_point_xy(pid)
            except Exception:
                return None

        collinear_map: Dict[str, List[Tuple["Constraint", Tuple[str, ...]]]] = defaultdict(list)
        for cons in self.VeriGraph.constraints.values():
            if cons.type != "Collinear":
                continue
            pts = tuple(
                pid
                for pid in cons.scope
                if pid in self.VeriGraph.entities and self.VeriGraph.entities[pid].type == "Point"
            )
            if len(pts) < 3:
                continue
            for pid in pts:
                collinear_map[pid].append((cons, pts))

        for cons in self.VeriGraph.constraints.values():
            if cons.type != "Concyclic":
                continue
            pts = [
                pid
                for pid in cons.scope
                if pid in self.VeriGraph.entities and self.VeriGraph.entities[pid].type == "Point"
            ]
            if len(pts) != 4:
                continue
            coord_map = {}
            skip = False
            for pid in pts:
                if pid in self.known:
                    # keep known (given/problem) fixed; they don't need correction
                    pass
                xy = _point_xy(pid)
                if xy is None:
                    skip = True
                    break
                coord_map[pid] = xy
            if skip:
                continue
            for target in pts:
                if target in self.known:
                    continue
                base = [pid for pid in pts if pid != target]
                base_coords = [coord_map[b] for b in base]
                if len(base_coords) != 3:
                    continue
                center = _circumcenter_from_points(
                    base_coords[0], base_coords[1], base_coords[2], eps=eps
                )
                if center is None:
                    continue
                cx, cy = center
                r = math.hypot(cx - base_coords[0][0], cy - base_coords[0][1])
                if not math.isfinite(r) or r < eps:
                    continue
                target_xy = coord_map[target]
                adjusted = False
                for _col_cons, pts_tuple in collinear_map.get(target, []):
                    pts_set = set(pts_tuple)
                    anchors = [pid for pid in base if pid in pts_set]
                    if not anchors:
                        continue
                    for anchor in anchors:
                        anchor_xy = coord_map[anchor]
                        if (anchor_xy[0] - target_xy[0]) ** 2 + (anchor_xy[1] - target_xy[1]) ** 2 > eps ** 2:
                            continue
                        for other in pts_tuple:
                            if other in (target, anchor):
                                continue
                            other_xy = _point_xy(other)
                            if other_xy is None:
                                continue
                            inters = _circle_line_intersections(
                                cx, cy, r, anchor_xy, other_xy, eps=eps
                            )
                            if not inters:
                                continue
                            best = None
                            best_dist = 0.0
                            for pt in inters:
                                dist = (pt[0] - anchor_xy[0]) ** 2 + (pt[1] - anchor_xy[1]) ** 2
                                if dist < eps ** 2:
                                    continue
                                if best is None or dist > best_dist:
                                    best = pt
                                    best_dist = dist
                            if best is None:
                                continue
                            ent = self.VeriGraph.entities.get(target)
                            if not ent or ent.type != "Point":
                                continue
                            ent.attrs.data.setdefault("coord", {})
                            ent.attrs.data["coord"]["x"] = float(best[0])
                            ent.attrs.data["coord"]["y"] = float(best[1])
                            coord_map[target] = (float(best[0]), float(best[1]))
                            self._refresh_incident_entities(target)
                            adjusted = True
                            break
                        if adjusted:
                            break
                    if adjusted:
                        break

    def _pid(self, label: str) -> Optional[str]:
        return self.point_lookup.get(label.upper()) or (label if label in self.VeriGraph.entities else None)

    def _sanitize_symbol_name(self, pid: str, target: str) -> str:
        import re
        raw = f"t_{pid}_{target}"
        return re.sub(r"[^A-Za-z0-9_]", "_", raw)

    def _get_or_create_param_symbol(self, key: Tuple[Any, ...], raw_name: str) -> Any:
        import re
        if key not in self.param_symbols:
            name = re.sub(r"[^A-Za-z0-9_]", "_", raw_name)
            self.param_symbols[key] = self.sym.symbols(name, real=True)
        return self.param_symbols[key]

    def _sym_point(self, pid: str) -> Optional[Tuple[Any, Any]]:
        return self._xy(pid)

    def _angle_cos_from_points(self, a_id: str, b_id: str, c_id: str) -> Optional[Any]:
        """Return signed cosine of ∠ABC (0–π) using symbolic coordinates."""
        A = self._sym_point(a_id); B = self._sym_point(b_id); C = self._sym_point(c_id)
        if not (A and B and C):
            return None
        Ax, Ay = A; Bx, By = B; Cx, Cy = C
        ux, uy = Ax - Bx, Ay - By
        vx, vy = Cx - Bx, Cy - By
        dot = ux * vx + uy * vy
        len_u2 = ux * ux + uy * uy
        len_v2 = vx * vx + vy * vy
        try:
            len_u = self.sym.sqrt(len_u2)
            len_v = self.sym.sqrt(len_v2)
            return dot / (len_u * len_v)
        except Exception:
            return None

    def _unit_vec(self, vec: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        vx, vy = vec
        n = math.hypot(vx, vy)
        if n < self.tol:
            return None
        return (vx / n, vy / n)

    def _project_point_to_line(self, pt: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        ab = (b[0] - a[0], b[1] - a[1])
        denom = ab[0] * ab[0] + ab[1] * ab[1]
        if denom < self.tol * self.tol:
            return None
        t = ((pt[0] - a[0]) * ab[0] + (pt[1] - a[1]) * ab[1]) / denom
        return (a[0] + ab[0] * t, a[1] + ab[1] * t)

    def _line_intersection(self, p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float], p4: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < self.tol:
            return None
        det1 = x1 * y2 - y1 * x2
        det2 = x3 * y4 - y3 * x4
        px = (det1 * (x3 - x4) - (x1 - x2) * det2) / denom
        py = (det1 * (y3 - y4) - (y1 - y2) * det2) / denom
        if not (math.isfinite(px) and math.isfinite(py)):
            return None
        return (px, py)

    def _segment_endpoints_safe(self, seg_id: str) -> Optional[Tuple[str, str]]:
        ent = self.VeriGraph.entities.get(seg_id)
        pts = None
        if ent and ent.type in ("Segment", "Line", "Ray", "Vector"):
            cand = ent.attrs.data.get("points") or ent.attrs.data.get("verts")
            if isinstance(cand, list) and len(cand) >= 2:
                pts = (cand[0], cand[1])
        if pts is None:
            try:
                pts = _segment_endpoints_for_VeriScript(self.VeriGraph, seg_id)
            except Exception:
                pts = None
        if pts is None and isinstance(seg_id, str) and seg_id.upper().startswith("SEG_"):
            decoded = _split_token_into_points(self.point_lookup, seg_id.upper()[4:], 2)
            if decoded:
                pts = decoded
        return pts

    def _circle_params_numeric(self, circle_id: str) -> Tuple[Optional[Tuple[float, float]], Optional[float]]:
        params = self.VeriGraph._circle_params(circle_id)
        if not params:
            return (None, None)
        cx, cy, r, _ = params
        if not (math.isfinite(cx) and math.isfinite(cy) and math.isfinite(r)):
            return (None, None)
        if r < self.tol:
            return (None, None)
        return ((float(cx), float(cy)), float(r))

    def _circle_params_symbolic(self, circle_id: str) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
        ent = self.VeriGraph.entities.get(circle_id)
        if not ent or ent.type != "Circle":
            return (None, None, None)
        center_id = ent.attrs.data.get("center")
        if not center_id:
            return (None, None, None)
        center_xy = self._sym_point(center_id)
        if not center_xy:
            return (None, None, None)
        cx, cy = center_xy
        r2: Optional[Any] = None
        radius_raw = ent.attrs.data.get("radius")
        if radius_raw is not None:
            try:
                radius_val = float(Fraction(str(radius_raw)))
                r2 = self.sym.Float(radius_val * radius_val)
            except Exception:
                r2 = None
        through_pt = ent.attrs.data.get("point_on_circle")
        if r2 is None and through_pt:
            pt_xy = self._sym_point(through_pt)
            if pt_xy:
                px, py = pt_xy
                r2 = (px - cx) ** 2 + (py - cy) ** 2
        if r2 is None:
            r2 = self._get_or_create_param_symbol((circle_id, "r2"), f"R2_{circle_id}")
        return (cx, cy, r2)

    def _eq_collinear(self, cons: "Constraint"):
        pts = [p for p in cons.scope if p in self.VeriGraph.entities]
        if len(pts) < 3:
            return
        for a, b, c in combinations(pts, 3):
            A = self._xy(a); B = self._xy(b); C = self._xy(c)
            if not (A and B and C):
                continue
            ax, ay = A; bx, by = B; cx, cy = C
            self.equations.append((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))

    def _eq_parallel(self, cons: "Constraint"):
        if len(cons.scope) < 2:
            return
        p1 = self._segment_endpoints_safe(cons.scope[0])
        p2 = self._segment_endpoints_safe(cons.scope[1])
        if not (p1 and p2):
            return
        a, b = p1; c, d = p2
        A = self._xy(a); B = self._xy(b); C = self._xy(c); D = self._xy(d)
        if not (A and B and C and D):
            return
        ax, ay = A; bx, by = B; cx, cy = C; dx, dy = D
        self.equations.append((bx - ax) * (dy - cy) - (by - ay) * (dx - cx))

    def _eq_perpendicular(self, cons: "Constraint"):
        if len(cons.scope) < 2:
            return
        p1 = self._segment_endpoints_safe(cons.scope[0])
        p2 = self._segment_endpoints_safe(cons.scope[1])
        if not (p1 and p2):
            return
        a, b = p1; c, d = p2
        A = self._xy(a); B = self._xy(b); C = self._xy(c); D = self._xy(d)
        if not (A and B and C and D):
            return
        ax, ay = A; bx, by = B; cx, cy = C; dx, dy = D
        self.equations.append((bx - ax) * (dx - cx) + (by - ay) * (dy - cy))

    def _eq_equal_length(self, cons: "Constraint"):
        if len(cons.scope) < 2:
            return
        p1 = self._segment_endpoints_safe(cons.scope[0])
        p2 = self._segment_endpoints_safe(cons.scope[1])
        if not (p1 and p2):
            return
        a, b = p1; c, d = p2
        A = self._xy(a); B = self._xy(b); C = self._xy(c); D = self._xy(d)
        if not (A and B and C and D):
            return
        ax, ay = A; bx, by = B; cx, cy = C; dx, dy = D
        len_ab2 = (bx - ax) ** 2 + (by - ay) ** 2
        len_cd2 = (dx - cx) ** 2 + (dy - cy) ** 2
        self.equations.append(len_ab2 - len_cd2)

    def _eq_equal_angle_poly(self, angle1_verts: Tuple[str, str, str], angle2_verts: Tuple[str, str, str]) -> bool:
        """Radical-free polynomial form for equal angles (orientation-agnostic; used as fallback).

        Condition: ((u·v)^2) * |u2|^2|v2|^2 = ((u2·v2)^2) * |u|^2|v|^2
        where u/v are the two arm vectors of angle 1 and u2/v2 are the two arm vectors of angle 2.
        """
        A1, B1, C1 = angle1_verts
        A2, B2, C2 = angle2_verts
        P1 = self._sym_point(A1); V1 = self._sym_point(B1); Q1 = self._sym_point(C1)
        P2 = self._sym_point(A2); V2 = self._sym_point(B2); Q2 = self._sym_point(C2)
        if not (P1 and V1 and Q1 and P2 and V2 and Q2):
            return False
        p1x, p1y = P1; v1x, v1y = V1; q1x, q1y = Q1
        p2x, p2y = P2; v2x, v2y = V2; q2x, q2y = Q2
        u1x, u1y = p1x - v1x, p1y - v1y
        v1bx, v1by = q1x - v1x, q1y - v1y
        u2x, u2y = p2x - v2x, p2y - v2y
        v2bx, v2by = q2x - v2x, q2y - v2y
        dot1 = u1x * v1bx + u1y * v1by
        dot2 = u2x * v2bx + u2y * v2by
        len1a = u1x * u1x + u1y * u1y
        len1b = v1bx * v1bx + v1by * v1by
        len2a = u2x * u2x + u2y * u2y
        len2b = v2bx * v2bx + v2by * v2by
        self.equations.append(dot1 * dot1 * len2a * len2b - dot2 * dot2 * len1a * len1b)
        return True

    def _eq_equal_angle_triplets(self, angle1_verts: Tuple[str, str, str], angle2_verts: Tuple[str, str, str]) -> bool:
        # First use signed cos form (distinguishes interior/exterior angles), then fall back to polynomial form.
        cos1 = self._angle_cos_from_points(*angle1_verts)
        cos2 = self._angle_cos_from_points(*angle2_verts)
        if cos1 is not None and cos2 is not None:
            self.equations.append(cos1 - cos2)
            return True
        return self._eq_equal_angle_poly(angle1_verts, angle2_verts)

    
    def _eq_equal_angle(self, cons: "Constraint"):
        if len(cons.scope) < 2:
            return
        sc = cons.scope
        verts1 = self._angle_vertices_from_token(sc[0])
        verts2 = self._angle_vertices_from_token(sc[1])
        # fallback: direct 6-point scope
        if not verts1 and len(sc) >= 3:
            verts1 = tuple(sc[:3])
        if not verts2 and len(sc) >= 6:
            verts2 = tuple(sc[3:6])
        if not (verts1 and verts2):
            return
        self._eq_equal_angle_triplets(tuple(verts1), tuple(verts2))

    def _eq_angle_bisector(self, cons: "Constraint"):
        if len(cons.scope) < 2:
            return
        seg_id = None
        angle_tok = None
        for tok in cons.scope:
            if seg_id is None and self._segment_endpoints_safe(tok):
                seg_id = tok
            elif angle_tok is None:
                angle_tok = tok
        if not seg_id or not angle_tok:
            return
        ends = self._segment_endpoints_safe(seg_id)
        verts = self._angle_vertices_from_token(angle_tok)
        if not (ends and verts):
            return
        vertex = verts[1]
        if vertex == ends[0]:
            other = ends[1]
        elif vertex == ends[1]:
            other = ends[0]
        else:
            return
        # Prefer signed cos form to preserve interior-angle direction; fall back to polynomial on failure.
        cos1 = self._angle_cos_from_points(verts[0], vertex, other)
        cos2 = self._angle_cos_from_points(other, vertex, verts[2])
        if cos1 is not None and cos2 is not None:
            self.equations.append(cos1 - cos2)
            return
        self._eq_equal_angle_triplets((verts[0], vertex, other), (other, vertex, verts[2]))

    def _eq_vector_dot_zero(self, cons: "Constraint"):
        if len(cons.scope) < 4:
            return
        a, b, c, d = cons.scope[:4]
        A = self._sym_point(a); B = self._sym_point(b); C = self._sym_point(c); D = self._sym_point(d)
        if not (A and B and C and D):
            return
        ax, ay = A; bx, by = B; cx, cy = C; dx, dy = D
        self.equations.append((bx - ax) * (dx - cx) + (by - ay) * (dy - cy))

    def _eq_midpoint(self, cons: "Constraint"):
        if len(cons.scope) < 2:
            return
        m = cons.scope[0]
        if len(cons.scope) >= 3:
            ends = (cons.scope[1], cons.scope[2])
        else:
            ends = self._segment_endpoints_safe(cons.scope[1])
        if not ends:
            return
        a, b = ends
        M = self._xy(m); A = self._xy(a); B = self._xy(b)
        if not (M and A and B):
            return
        mx, my = M; ax, ay = A; bx, by = B
        self.equations.append(2 * mx - ax - bx)
        self.equations.append(2 * my - ay - by)

    def _eq_point_on_line(self, cons: "Constraint"):
        if len(cons.scope) < 2:
            return
        pid, line_id = cons.scope[0], cons.scope[1]
        pts = self._segment_endpoints_safe(line_id)
        if not pts:
            return
        a, b = pts
        P = self._sym_point(pid); A = self._sym_point(a); B = self._sym_point(b)
        if not (P and A and B):
            return
        px, py = P; ax, ay = A; bx, by = B
        self.equations.append((px - ax) * (by - ay) - (py - ay) * (bx - ax))

    def _eq_point_on_segment(self, cons: "Constraint"):
        if len(cons.scope) < 2:
            return
        pid, seg_id = cons.scope[0], cons.scope[1]
        pts = self._segment_endpoints_safe(seg_id)
        if not pts:
            return
        a, b = pts
        P = self._sym_point(pid); A = self._sym_point(a); B = self._sym_point(b)
        if not (P and A and B):
            return
        px, py = P; ax, ay = A; bx, by = B
        ratio_val = None
        if isinstance(cons.params, dict):
            try:
                ratio_val = float(Fraction(str(cons.params.get("ratio"))))
            except Exception:
                ratio_val = None
        key = (pid, seg_id)
        t_sym = None if ratio_val is not None else self.param_symbols.get(key)
        if t_sym is None and ratio_val is None:
            t_sym = self._get_or_create_param_symbol(key, self._sanitize_symbol_name(pid, seg_id))
        if ratio_val is not None:
            self.equations.append(px - (ax + ratio_val * (bx - ax)))
            self.equations.append(py - (ay + ratio_val * (by - ay)))
        elif t_sym is not None:
            self.equations.append(px - (ax + t_sym * (bx - ax)))
            self.equations.append(py - (ay + t_sym * (by - ay)))
        else:
            self.equations.append((bx - ax) * (py - ay) - (by - ay) * (px - ax))

    def _eq_incidence(self, cons: "Constraint"):
        if len(cons.scope) < 2:
            return
        point_id = None
        target = None
        for tok in cons.scope:
            ent = self.VeriGraph.entities.get(tok)
            if ent and ent.type == "Point" and point_id is None:
                point_id = tok
            elif target is None:
                target = tok
        if point_id is None or target is None:
            return
        target_ent = self.VeriGraph.entities.get(target)
        if target_ent and target_ent.type == "Circle":
            P = self._sym_point(point_id)
            cx, cy, r2 = self._circle_params_symbolic(target)
            if P is None or cx is None or cy is None or r2 is None:
                return
            px, py = P
            self.equations.append((px - cx) ** 2 + (py - cy) ** 2 - r2)
            return
        pts = self._segment_endpoints_safe(target)
        if not pts:
            return
        a, b = pts
        P = self._sym_point(point_id); A = self._sym_point(a); B = self._sym_point(b)
        if not (P and A and B):
            return
        px, py = P; ax, ay = A; bx, by = B
        self.equations.append((px - ax) * (by - ay) - (py - ay) * (bx - ax))

    def _eq_on_circle(self, cons: "Constraint"):
        if len(cons.scope) < 2:
            return
        pid, circle_id = cons.scope[0], cons.scope[1]
        P = self._sym_point(pid)
        cx, cy, r2 = self._circle_params_symbolic(circle_id)
        if P is None or cx is None or cy is None or r2 is None:
            return
        px, py = P
        self.equations.append((px - cx) ** 2 + (py - cy) ** 2 - r2)

    def _eq_concyclic(self, cons: "Constraint"):
        scope = cons.scope or []
        if len(scope) == 2 and scope[1] in self.VeriGraph.entities and self.VeriGraph.entities[scope[1]].type == "Circle":
            return self._eq_on_circle(cons)
        pts = [p for p in scope if p in self.VeriGraph.entities and self.VeriGraph.entities[p].type == "Point"]
        if len(pts) < 4:
            return
        def _row(pid: str):
            xy = self._sym_point(pid)
            if not xy:
                return None
            xi, yi = xy
            return [xi * xi + yi * yi, xi, yi, 1]
        base_rows = []
        for pid in pts[:3]:
            row = _row(pid)
            if row is None:
                return
            base_rows.append(row)
        for pid in pts[3:]:
            row4 = _row(pid)
            if row4 is None:
                return
            mat = self.sym.Matrix(base_rows + [row4])
            self.equations.append(mat.det())

    def _add_gauge_constraints(self):
        known_ids = list(self.known.keys()) + list(self.closed_form_results.keys())
        if len(known_ids) >= 2:
            return
        free_points = [pid for pid in self.symbols.keys()]
        if len(free_points) < 1:
            return
        if len(known_ids) == 1:
            anchor = known_ids[0]
            q = free_points[0] if free_points[0] != anchor else (free_points[1] if len(free_points) > 1 else None)
            if q is None:
                return
            ax, ay = self._sym_point(anchor)
            qx, qy = self.symbols[q]
            self.equations.extend([qy - ay, qx - ax - 1])
            return
        if len(free_points) >= 2:
            p0, p1 = free_points[0], free_points[1]
            x0, y0 = self.symbols[p0]
            x1, y1 = self.symbols[p1]
            self.equations.extend([x0, y0, y1, x1 - 1])

    def _circumcircle_params(self, A: Optional[Tuple[float, float]], B: Optional[Tuple[float, float]], C: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float, float]]:
        if not (A and B and C):
            return None
        center = _circumcenter_from_points(A, B, C)
        if not center:
            return None
        r = math.hypot(center[0] - A[0], center[1] - A[1])
        if not math.isfinite(r) or r < self.tol:
            return None
        return (center[0], center[1], r)

    def _angle_vertices_from_token(self, token: str) -> Optional[Tuple[str, str, str]]:
        return self.VeriGraph._angle_vertices_from_token(token, self.point_lookup)

    def _refresh_incident_entities(self, pid: str):
        for ent in self.VeriGraph.entities.values():
            if ent.type not in ("Segment", "Line", "Ray", "Vector"):
                continue
            pts = ent.attrs.data.get("points") or ent.attrs.data.get("verts")
            if not (isinstance(pts, list) and len(pts) >= 2 and pid in pts):
                continue
            try:
                p0 = self.VeriGraph.get_point_xy(pts[0])
                p1 = self.VeriGraph.get_point_xy(pts[1])
            except Exception:
                continue
            ent.attrs.data.setdefault("points", list(pts))
            ent.attrs.data["cached_coords"] = [{"x": float(p0[0]), "y": float(p0[1])}, {"x": float(p1[0]), "y": float(p1[1])}]
            dx = float(p1[0] - p0[0]); dy = float(p1[1] - p0[1])
            ent.attrs.data["dir"] = {"dx": dx, "dy": dy}
            length = math.hypot(dx, dy)
            ent.attrs.data["length"] = length
            if length > 1e-12:
                ent.attrs.data["normal"] = {"dx": -dy / length, "dy": dx / length}
            else:
                ent.attrs.data["normal"] = {"dx": 0.0, "dy": 0.0}
            if ent.type in ("Line", "Ray", "Vector"):
                ent.attrs.data["direction"] = {"dx": dx, "dy": dy}

# ================================ Data Structures ================================
@dataclass
class Attrs:
    data: Dict[str, Any] = field(default_factory=dict)


    @staticmethod
    def quad(verts: List[str], kind: str = "Quad", provenance: str = "given"):
        """
        General quadrilateral (and subclasses): kind ∈ {"Quad","Parallelogram","Rectangle","Rhombus","Trapezoid","Square","IsoscelesTrapezoid","Cyclic","Kite"}
        """
        return Attrs({"verts": list(verts), "kind": kind, "provenance": provenance})

    @staticmethod
    def polygon(verts: List[str], provenance: str = "given"):
        """Use this for more general polygons if needed (this patch does not enforce its use)"""
        return Attrs({"verts": list(verts), "provenance": provenance})

    @staticmethod
    def point(x: float, y: float, provenance: str = "given"):
        return Attrs({"coord": {"x": float(x), "y": float(y)}, "provenance": provenance})

    @staticmethod
    def circle(center: str, radius: float, provenance: str = "given"):
        return Attrs({"center": center, "radius": float(radius), "provenance": provenance})

    @staticmethod
    def triangle(verts: List[str], provenance: str = "given"):
        return Attrs({"verts": list(verts), "provenance": provenance})

    @staticmethod
    def angle(vertices: str, provenance: str = "given"):
        return Attrs({"name": vertices, "provenance": provenance})

@dataclass
class Entity:
    id: str
    type: str  # Point / Segment / Line / Circle / Triangle / Angle
    attrs: Attrs = field(default_factory=Attrs)

@dataclass
class Constraint:
    id: str
    type: str
    scope: List[str]
    params: Dict[str, Any] = field(default_factory=dict)
    source: str = "problem"      # problem / proof / derived / check / aux
    validity: str = "assumed"

class GeometryGraph:
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.constraints: Dict[str, Constraint] = {}
        self.meta: Dict[str, Any] = {"tolerance": 1e-6, "triangle_edges": [], "hard_locked_points": []}
        self.semantic_priority: Dict[str, int] = {}
        # Ensure construction_order exists
        self.meta.setdefault("construction_order", [])
    # ---- helper: ??????????????----
    def _is_protected_point(self, pid: str, ignore_constraint: Optional[str] = None) -> bool:
        if pid not in self.entities or self.entities[pid].type != "Point":
            return False
        ent = self.entities[pid]
        prov = ent.attrs.data.get("provenance", "")
        locked_pts = set(self.meta.get("hard_locked_points") or [])
        if pid in locked_pts:
            return True
        if prov in ("given", "problem"):
            return True
        for c in self.constraints.values():
            if ignore_constraint and c.id == ignore_constraint:
                continue
            if c.type in ("OnCircle", "Concyclic") and pid in c.scope:
                return True
        for eid, e in self.entities.items():
            if e.type == "Circle" and e.attrs.data.get("center") == pid:
                return True
        if "TRI_ABC" in self.entities:
            verts = self.entities["TRI_ABC"].attrs.data.get("verts", [])
            if pid in verts:
                return True
        locked = set(self.meta.get("triangle_edges", []))
        if locked:
            for c in self.constraints.values():
                if c.type != "Incidence" or pid not in c.scope:
                    continue
                other = [s for s in c.scope if s != pid]
                if other and other[0] in locked:
                    return True
        return False

    def _angle_vertices_from_token(self, token: str, point_lookup: Optional[Dict[str, str]] = None) -> Optional[Tuple[str, str, str]]:
        lookup = point_lookup or _build_point_lookup(self)
        body = (token or "").strip()
        if not body:
            return None
        ent = self.entities.get(body)
        if ent is None:
            alias_map = _collect_alias_map(self)
            alias = alias_map.get(body)
            if alias and alias in self.entities:
                body = alias
                ent = self.entities.get(body)
        name = None
        if ent and ent.type == "Angle":
            name = ent.attrs.data.get("name") or body
        target = (name or body).strip().upper()
        for ch in ("∠", " ", "?"):
            target = target.replace(ch, "")
        if target.startswith("ANG_"):
            target = target[4:]
        pts = _split_token_into_points(lookup, target, 3)
        return tuple(pts) if pts else None

    def _can_adjust_circle_center(self, center_id: str) -> bool:
        ent = self.entities.get(center_id)
        if not ent or ent.type != "Point":
            return False
        prov = ent.attrs.data.get("provenance", "")
        if prov in ("given", "problem"):
            return False
        if "TRI_ABC" in self.entities:
            verts = self.entities["TRI_ABC"].attrs.data.get("verts", [])
            if isinstance(verts, list) and center_id in verts:
                return False
        locked = set(self.meta.get("triangle_edges", []))
        if locked:
            for c in self.constraints.values():
                if c.type != "Incidence" or center_id not in c.scope:
                    continue
                other = [s for s in c.scope if s != center_id]
                if other and other[0] in locked:
                    return False
        return True

    def _collect_protected_points_on_circle(self, circle_id: str,
                                            ignore_constraint: Optional[str] = None) -> List[str]:
        pts: List[str] = []
        seen: set[str] = set()
        for c in self.constraints.values():
            if ignore_constraint and c.id == ignore_constraint:
                continue
            if c.type not in ("OnCircle", "Concyclic"):
                continue
            if len(c.scope) != 2:
                continue
            p, cir = c.scope
            if cir != circle_id or p in seen:
                continue
            if p not in self.entities or self.entities[p].type != "Point":
                continue
            if self._is_protected_point(p, ignore_constraint=ignore_constraint):
                pts.append(p)
                seen.add(p)
        return pts

    def _relocate_circle_center(self, circle_id: str, center_id: str, protected_points: List[str]) -> bool:
        coords: List[Tuple[str, Tuple[float, float]]] = []
        seen: set[str] = set()
        for pid in protected_points:
            if pid == center_id or pid in seen:
                continue
            if pid not in self.entities or self.entities[pid].type != "Point":
                continue
            try:
                coords.append((pid, self.get_point_xy(pid)))
            except Exception:
                continue
            seen.add(pid)
        if len(coords) < 2:
            return False
        if center_id not in self.entities or self.entities[center_id].type != "Point":
            return False
        try:
            cx_old, cy_old = self.get_point_xy(center_id)
        except Exception:
            cx_old = cy_old = 0.0

        target_center: Optional[Tuple[float, float]] = None
        target_radius: Optional[float] = None

        for (pid1, p1), (pid2, p2), (pid3, p3) in combinations(coords, 3):
            center = _circumcenter_from_points(p1, p2, p3)
            if center is None:
                continue
            radius = math.hypot(center[0] - p1[0], center[1] - p1[1])
            if radius < 1e-6 or not math.isfinite(radius):
                continue
            target_center = center
            target_radius = radius
            break

        if target_center is None:
            (pid1, p1), (pid2, p2) = coords[0], coords[1]
            chord = (p2[0] - p1[0], p2[1] - p1[1])
            chord_len = math.hypot(chord[0], chord[1])
            if chord_len < 1e-9:
                return False
            mid = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
            perp = (-chord[1], chord[0])
            perp_len = math.hypot(perp[0], perp[1])
            if perp_len < 1e-9:
                return False
            nx, ny = perp[0] / perp_len, perp[1] / perp_len
            vec = (cx_old - mid[0], cy_old - mid[1])
            t = vec[0] * nx + vec[1] * ny
            target_center = (mid[0] + t * nx, mid[1] + t * ny)
            target_radius = math.hypot(target_center[0] - p1[0], target_center[1] - p1[1])

        if target_radius is None or not math.isfinite(target_radius) or target_radius < 1e-6:
            return False

        center_ent = self.entities[center_id]
        center_ent.attrs.data.setdefault("coord", {})
        center_ent.attrs.data["coord"]["x"] = float(target_center[0])
        center_ent.attrs.data["coord"]["y"] = float(target_center[1])
        self.entities[circle_id].attrs.data["radius"] = float(target_radius)
        return True

    def _point_bound_circles(self, pid: str, ignore_constraint: Optional[str] = None) -> List[str]:
        circles: List[str] = []
        for c in self.constraints.values():
            if ignore_constraint and c.id == ignore_constraint:
                continue
            if c.type not in ("OnCircle", "Concyclic") or len(c.scope) != 2:
                continue
            p, cir = c.scope
            if p == pid:
                circles.append(cir)
        return circles

    def _point_line_guides(self, pid: str, ignore_constraint: Optional[str] = None) -> List[Dict[str, Any]]:
        guides: List[Dict[str, Any]] = []
        seen_pairs: set[Tuple[str, str, str]] = set()

        def _append_from_points(a: str, b: str, kind: str, seg_id: Optional[str] = None):
            if a == b:
                return
            key = tuple(sorted((a, b)) + [kind + (seg_id or "")])
            if key in seen_pairs:
                return
            try:
                pa = self.get_point_xy(a); pb = self.get_point_xy(b)
            except Exception:
                return
            guides.append({"kind": kind, "pts": (pa, pb), "seg_id": seg_id})
            seen_pairs.add(key)

        for c in self.constraints.values():
            if ignore_constraint and c.id == ignore_constraint:
                continue
            if pid not in c.scope:
                continue
            if c.type == "PointOnSegment":
                seg_id = next((s for s in c.scope if s != pid), None)
                if not seg_id:
                    continue
                try:
                    u, v = _segment_endpoints_for_VeriScript(self, seg_id)
                    _append_from_points(u, v, "segment", seg_id)
                except Exception:
                    continue
            elif c.type == "Incidence":
                seg_ids = [s for s in c.scope if s != pid and s in self.entities and self.entities[s].type == "Segment"]
                for seg_id in seg_ids:
                    try:
                        u, v = _segment_endpoints_for_VeriScript(self, seg_id)
                        _append_from_points(u, v, "segment", seg_id)
                    except Exception:
                        continue
            elif c.type == "Collinear" and len(c.scope) >= 3:
                others = [s for s in c.scope if s != pid and s in self.entities and self.entities[s].type == "Point"]
                if len(others) < 2:
                    continue
                _append_from_points(others[0], others[1], "line")
        return guides

    def _move_point_to_circle_line(self, point_id: str, circle_id: str,
                                   guides: List[Dict[str, Any]]) -> bool:
        params = self._circle_params(circle_id)
        if not params or point_id not in self.entities:
            return False
        cx, cy, r, _ = params
        p_ent = self.entities[point_id]
        px = p_ent.attrs.data["coord"]["x"]
        py = p_ent.attrs.data["coord"]["y"]

        for guide in guides:
            pts = guide.get("pts")
            if not pts:
                continue
            intersections = _circle_line_intersections(cx, cy, r, pts[0], pts[1])
            if not intersections:
                continue
            candidates: List[Tuple[float, float]] = []
            if guide.get("kind") == "segment":
                for pt in intersections:
                    t = _line_param(pts[0], pts[1], pt)
                    if t is None:
                        continue
                    if -1e-6 <= t <= 1.0 + 1e-6:
                        candidates.append(pt)
            else:
                candidates = intersections
            if not candidates:
                continue
            target = min(candidates, key=lambda q: (q[0]-px)**2 + (q[1]-py)**2)
            p_ent.attrs.data["coord"]["x"] = float(target[0])
            p_ent.attrs.data["coord"]["y"] = float(target[1])
            return True
        return False

    def _enforce_circle_line_for_collinear(self, points: List[str], ignore_constraint: Optional[str] = None):
        valid = [p for p in points if p in self.entities and self.entities[p].type == "Point"]
        if len(valid) < 3:
            return
        for pid in valid:
            others = [q for q in valid if q != pid]
            if len(others) < 2:
                continue
            try:
                p1 = self.get_point_xy(others[0])
                p2 = self.get_point_xy(others[1])
            except Exception:
                continue
            guide = {"kind": "line", "pts": (p1, p2)}
            circles = self._point_bound_circles(pid, ignore_constraint=ignore_constraint)
            for cir in circles:
                self._move_point_to_circle_line(pid, cir, [guide])
    def _circle_params(self, circle_id: str):
        if circle_id not in self.entities or self.entities[circle_id].type != "Circle":
            return None
        attrs = self.entities[circle_id].attrs.data
        center_id = attrs.get("center")
        if not center_id or center_id not in self.entities:
            return None
        if self.entities[center_id].type != "Point":
            return None
        try:
            cx, cy = self.get_point_xy(center_id)
        except Exception:
            return None
        try:
            radius = float(attrs.get("radius"))
        except (TypeError, ValueError):
            return None
        return cx, cy, radius, center_id

    # ---------- CRUD ----------
    def add_entity(self, type: str, id: Optional[str] = None, attrs: Optional[Attrs] = None) -> str:
        eid = id or _uid(type[:2].upper())
        self.entities[eid] = Entity(eid, type, attrs or Attrs())
        if type == "Point":
            # Initialize semantic priority to avoid lookup defaults later
            self.semantic_priority[eid] = self.semantic_priority.get(eid, 0)
        return eid

    def add_constraint(self, type: str, scope: List[str], params: Optional[Dict[str, Any]] = None,
                       source: str = "problem", validity: str = "assumed", id: Optional[str] = None,
                       apply_immediate: bool = True) -> str:
        skip_numeric_indices = set()
        if type == "VectorLinear":
            for idx in range(2, len(scope), 3):
                skip_numeric_indices.add(idx)
        for idx, s in enumerate(scope):
            if idx in skip_numeric_indices:
                continue
            if s not in self.entities:
                raise ValueError(f"Unknown entity in scope: {s}")
        cid = id or _uid(type[:2].upper())
        self.constraints[cid] = Constraint(cid, type, scope, params or {}, source, validity)
        self.meta.setdefault("construction_order", []).append(cid)
        # === Automatic geometry correction (circle-incidence consistency) ===
        # Rules:
        # 1) If the point is protected (given/skeleton point), prefer "adjust the circle": change the radius so the point lies on the circle;
        # 2) If the point is not protected, move the point to the circle; when moving, try to keep the original direction (using the original polar angle as the main reference).
        if apply_immediate and type in ("Concyclic", "OnCircle") and len(scope) == 2:
            point_id, circle_id = scope
            if point_id in self.entities and circle_id in self.entities:
                p_ent = self.entities[point_id]
                if p_ent.type == "Point":
                    params = self._circle_params(circle_id)
                    if params is not None:
                        import math, random
                        cx, cy, r, center_id = params
                        px = p_ent.attrs.data["coord"]["x"]
                        py = p_ent.attrs.data["coord"]["y"]
                        dx, dy = px - cx, py - cy
                        dist = math.hypot(dx, dy)

                        moved_on_line = False
                        line_guides = self._point_line_guides(point_id, ignore_constraint=cid)
                        if line_guides and not self._is_protected_point(point_id, ignore_constraint=cid):
                            moved_on_line = self._move_point_to_circle_line(point_id, circle_id, line_guides)

                        if not moved_on_line and self._is_protected_point(point_id, ignore_constraint=cid):
                            protected_pts = self._collect_protected_points_on_circle(circle_id, ignore_constraint=cid)
                            if point_id not in protected_pts:
                                protected_pts.append(point_id)
                            center_adjusted = False
                            if len(protected_pts) >= 2 and self._can_adjust_circle_center(center_id):
                                center_adjusted = self._relocate_circle_center(circle_id, center_id, protected_pts)
                                if center_adjusted:
                                    params = self._circle_params(circle_id) or params
                                    cx, cy, r, center_id = params
                            if not center_adjusted:
                                new_r = dist if dist > 1e-9 else max(r, 1.0)
                                self.entities[circle_id].attrs.data["radius"] = float(new_r)
                        elif not moved_on_line:
                            # —— Move point to circle (preserving direction as much as possible) ——
                            if dist < 1e-9:
                                dir_vec = random.choice(_PYTHAGOREAN_DIRECTIONS)
                                dir_len = math.hypot(dir_vec[0], dir_vec[1]) or 1.0
                                nx = cx + r * dir_vec[0] / dir_len
                                ny = cy + r * dir_vec[1] / dir_len
                            else:
                                theta = math.atan2(dy, dx)
                                nx = cx + r * math.cos(theta)
                                ny = cy + r * math.sin(theta)
                            p_ent.attrs.data["coord"]["x"] = float(nx)
                            p_ent.attrs.data["coord"]["y"] = float(ny)

        if apply_immediate and type == "Collinear" and len(scope) >= 3:
            self._enforce_circle_line_for_collinear(scope, ignore_constraint=cid)

        return cid

    def get_point_xy(self, pid: str) -> Tuple[float, float]:
        e = self.entities[pid]
        x = e.attrs.data["coord"]["x"]; y = e.attrs.data["coord"]["y"]
        return float(x), float(y)
    def to_json(self) -> str:
        return json.dumps({
            "entities": [asdict(e) for e in self.entities.values()],
            "constraints": [asdict(c) for c in self.constraints.values()],
            "meta": self.meta
        }, ensure_ascii=False, indent=2)

    def normalize_polygons(self):
        poly_types = {"Quad", "Parallelogram", "Rectangle", "Rhombus", "Trapezoid", "Polygon", "Quadrilateral", "Square", "IsoscelesTrapezoid", "Kite", "Cyclic"}
        preserve_kind = {"Trapezoid", "Parallelogram", "Rectangle", "Square", "Rhombus", "IsoscelesTrapezoid", "Kite", "Cyclic"}
        for eid, ent in self.entities.items():
            if ent.type not in poly_types:
                continue
            kind = str(ent.attrs.data.get("kind", "") or "")
            if ent.attrs.data.get("preserve_order") or (ent.type == "Quadrilateral" and kind in preserve_kind):
                # Preserve input order to avoid A-B-C-D being reordered to A-B-D-C
                continue
            verts = ent.attrs.data.get("verts")
            if not (isinstance(verts, list) and len(verts) >= 3):
                continue
            coords = []
            valid = True
            for vid in verts:
                if vid not in self.entities or self.entities[vid].type != "Point":
                    valid = False
                    break
                coords.append((vid, self.get_point_xy(vid)))
            if not valid:
                continue
            cx = sum(p[1][0] for p in coords) / len(coords)
            cy = sum(p[1][1] for p in coords) / len(coords)
            ordered = sorted(
                coords,
                key=lambda item: math.atan2(item[1][1] - cy, item[1][0] - cx)
            )
            ent.attrs.data["verts"] = [vid for vid, _ in ordered]

    def snap_free_rays_first_hit(self, tol: float = 1e-6, hide_on_fail: bool = True, include_circles: bool = True) -> Dict[str, int]:
        """
        Snap free points along the "anchor → free point" ray to the nearest intersection; if there is no intersection, optionally hide the point to avoid flying out of the canvas.
        Only acts on non-protected points whose origin=free or which historically had no origin, and which participate only in Incidence constraints.
        """
        eff_tol = max(tol, float(self.meta.get("tolerance", 1e-6)))
        hidden_points = set(self.meta.get("hidden_points") or [])
        stats = {"total": 0, "snapped": 0, "hidden": 0}

        # First clear stale hidden marks from free points to allow re-snapping this round
        for pid, ent in self.entities.items():
            if ent.type != "Point":
                continue
            if ent.attrs.data.get("origin") not in ("free", "author_free", "proof_free", "llm_free"):
                continue
            ent.attrs.data.pop("hidden", None)

        def _is_loose_free(pid: str) -> bool:
            ent = self.entities.get(pid)
            if not ent or ent.type != "Point":
                return False
            origin = ent.attrs.data.get("origin")
            prov = ent.attrs.data.get("provenance", "")
            origin_free = origin in ("free", "author_free", "proof_free", "llm_free")
            if origin in ("free", "author_free", "proof_free", "llm_free"):
                pass
            elif origin is None and prov not in ("given", "problem"):
                pass  # backward compatible with old Free points
            else:
                return False
            if (not origin_free) and self._is_protected_point(pid):
                return False
            for cons in list(self.constraints.values()):
                if pid not in cons.scope:
                    continue
                if cons.type in ("PointOnSegment", "PointOnLine", "OnCircle", "PointOnCircle", "Concyclic"):
                    return False
                if cons.type != "Incidence":
                    return False
            return True

        def _choose_anchor(pid: str) -> Optional[Tuple[str, Optional[str]]]:
            best: Optional[Tuple[int, str, Optional[str]]] = None
            for cons in self.constraints.values():
                if cons.type != "Incidence" or pid not in cons.scope:
                    continue
                for tok in cons.scope:
                    if tok == pid:
                        continue
                    ent = self.entities.get(tok)
                    if not ent or ent.type != "Segment":
                        continue
                    try:
                        u, v = _segment_endpoints_for_VeriScript(self, tok)
                    except Exception:
                        continue
                    if pid not in (u, v):
                        continue
                    anchor = v if u == pid else u
                    if anchor not in self.entities or self.entities[anchor].type != "Point":
                        continue
                    score = 0
                    if self._is_protected_point(anchor):
                        score += 10
                    origin_flag = self.entities[anchor].attrs.data.get("origin")
                    if origin_flag == "free":
                        score -= 1
                    if best is None or score > best[0] or (score == best[0] and anchor < best[1]):
                        best = (score, anchor, tok)
            if best:
                return best[1], best[2]
            return None

        def _ray_line_intersection(origin: Tuple[float, float], direction: Tuple[float, float],
                                   a: Tuple[float, float], b: Tuple[float, float],
                                   bounded: bool) -> Optional[Tuple[float, Tuple[float, float]]]:
            ox, oy = origin
            dx, dy = direction
            sx, sy = b[0] - a[0], b[1] - a[1]
            denom = dx * sy - dy * sx
            if abs(denom) < eff_tol:
                return None
            t = ((a[0] - ox) * sy - (a[1] - oy) * sx) / denom
            if t < eff_tol:
                return None
            if bounded:
                u = ((a[0] - ox) * dy - (a[1] - oy) * dx) / denom
                if u < -eff_tol or u > 1.0 + eff_tol:
                    return None
            hit = (ox + dx * t, oy + dy * t)
            if not (math.isfinite(hit[0]) and math.isfinite(hit[1])):
                return None
            return (t, hit)

        def _ray_circle_params(origin: Tuple[float, float], direction: Tuple[float, float],
                               center: Tuple[float, float], radius: float) -> List[float]:
            ox, oy = origin[0] - center[0], origin[1] - center[1]
            dx, dy = direction
            a = dx * dx + dy * dy
            if a < eff_tol * eff_tol:
                return []
            b = 2.0 * (ox * dx + oy * dy)
            c = ox * ox + oy * oy - radius * radius
            disc = b * b - 4.0 * a * c
            if disc < -eff_tol:
                return []
            disc = max(disc, 0.0)
            sqrt_disc = math.sqrt(disc)
            t1 = (-b - sqrt_disc) / (2.0 * a)
            t2 = (-b + sqrt_disc) / (2.0 * a)
            return [t for t in (t1, t2) if t >= eff_tol]

        def _xy(pid: str) -> Optional[Tuple[float, float]]:
            try:
                return self.get_point_xy(pid)
            except Exception:
                return None

        circle_list: List[Tuple[float, float, float, str]] = []
        if include_circles:
            for eid, ent in self.entities.items():
                if ent.type != "Circle":
                    continue
                params = self._circle_params(eid)
                if params:
                    cx, cy, r, _ = params
                    circle_list.append((cx, cy, r, eid))
            for cons in self.constraints.values():
                if cons.type != "Concyclic":
                    continue
                pts = [p for p in cons.scope if p in self.entities and self.entities[p].type == "Point"]
                if len(pts) < 3:
                    continue
                A = _xy(pts[0]); B = _xy(pts[1]); C = _xy(pts[2])
                if not (A and B and C):
                    continue
                ax, ay = A; bx, by = B; cx, cy = C
                d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
                if abs(d) < eff_tol:
                    continue
                def _sq(x, y): return x * x + y * y
                ux = (_sq(ax, ay) * (by - cy) + _sq(bx, by) * (cy - ay) + _sq(cx, cy) * (ay - by)) / d
                uy = (_sq(ax, ay) * (cx - bx) + _sq(bx, by) * (ax - cx) + _sq(cx, cy) * (bx - ax)) / d
                r = math.hypot(ux - ax, uy - ay)
                if math.isfinite(ux) and math.isfinite(uy) and math.isfinite(r) and r > eff_tol:
                    circle_list.append((ux, uy, r, f"Concyclic_{cons.id}"))

        free_candidates = [pid for pid in self.entities if _is_loose_free(pid)]
        stats["total"] = len(free_candidates)
        for pid in free_candidates:
            anchor_info = _choose_anchor(pid)
            if not anchor_info:
                continue
            anchor_id, _ = anchor_info
            try:
                anchor_xy = self.get_point_xy(anchor_id)
                free_xy = self.get_point_xy(pid)
            except Exception:
                continue
            dir_vec = (free_xy[0] - anchor_xy[0], free_xy[1] - anchor_xy[1])
            dir_norm = math.hypot(dir_vec[0], dir_vec[1])
            if dir_norm < eff_tol:
                continue
            hits_lines: List[Tuple[float, Tuple[float, float], str]] = []
            for eid, ent in self.entities.items():
                if ent.type in ("Segment", "Line", "Ray", "Vector"):
                    try:
                        u, v = _segment_endpoints_for_VeriScript(self, eid)
                    except Exception:
                        continue
                    if pid in (u, v):
                        continue
                    try:
                        a_xy = self.get_point_xy(u)
                        b_xy = self.get_point_xy(v)
                    except Exception:
                        continue
                    bounded = ent.type == "Segment"
                    hit = _ray_line_intersection(anchor_xy, dir_vec, a_xy, b_xy, bounded=bounded)
                    if hit:
                        hits_lines.append((hit[0], hit[1], eid))
            hits = hits_lines
            if not hits and include_circles:
                for (cx, cy, r, cid) in circle_list:
                    ts = _ray_circle_params(anchor_xy, dir_vec, (cx, cy), r)
                    for t in ts:
                        hits.append((t, (anchor_xy[0] + dir_vec[0] * t, anchor_xy[1] + dir_vec[1] * t), cid))
            if not hits:
                continue
            hits.sort(key=lambda item: item[0])
            best_t, best_xy, hit_id = hits[0]
            ent = self.entities[pid]
            ent.attrs.data.setdefault("coord", {})
            ent.attrs.data["coord"]["x"] = float(best_xy[0])
            ent.attrs.data["coord"]["y"] = float(best_xy[1])
            ent.attrs.data.pop("hidden", None)
            ent.attrs.data["ray_snap"] = {"anchor": anchor_id, "hit": hit_id, "t": float(best_t)}
            stats["snapped"] += 1

        # Refresh the hidden list based on entity marks, ensuring re-snapped points are removed from hidden
        hidden_points = {
            pid for pid, ent in self.entities.items()
            if ent.type == "Point" and ent.attrs.data.get("hidden")
        }
        self.meta["hidden_points"] = sorted(hidden_points) if hidden_points else []
        return stats

    # ---------- Render ----------
    def render(self, figsize=(6, 6), save_path: Optional[str] = None, show=True, minimal: bool = True):
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        ax.set_facecolor("#ffffff")
        fig.patch.set_facecolor("#ffffff")

        hidden_points = {
            eid for eid, e in self.entities.items()
            if e.type == "Point" and e.attrs.data.get("hidden")
        }
        points = {eid: e for eid, e in self.entities.items() if e.type == "Point" and eid not in hidden_points}
        xs: List[float] = []
        ys: List[float] = []
        point_xy: Dict[str, Tuple[float, float]] = {}
        for pid, p in points.items():
            x = float(p.attrs.data["coord"]["x"]); y = float(p.attrs.data["coord"]["y"])
            point_xy[pid] = (x, y)
            xs.append(x); ys.append(y)
        point_bbox_x: List[float] = list(xs)
        point_bbox_y: List[float] = list(ys)
        bbox_x: List[float] = list(xs)
        bbox_y: List[float] = list(ys)
        show_labels = not minimal
        line_color = "#000000"
        function_segments: List[Tuple[List[float], List[float], str]] = []
        has_function = False
        for fid, ent in self.entities.items():
            if ent.type != "Function":
                continue
            has_function = True
            expr_text = ent.attrs.data.get("expr") or ent.attrs.data.get("expression")
            if not expr_text:
                continue
            fn = _compile_function_expr(expr_text)
            if fn is None:
                continue
            try:
                xmin = float(ent.attrs.data.get("xmin", -5.0))
                xmax = float(ent.attrs.data.get("xmax", 5.0))
                samples = int(ent.attrs.data.get("samples", 240))
            except Exception:
                xmin, xmax, samples = -5.0, 5.0, 240
            if xmax < xmin:
                xmin, xmax = xmax, xmin
            if abs(xmax - xmin) < 1e-9:
                xmax = xmin + 1.0
            samples = max(20, min(samples, 2000))
            xs_arr = np.linspace(xmin, xmax, samples)
            seg_xs: List[float] = []
            seg_ys: List[float] = []
            for x in xs_arr:
                y = fn(float(x))
                if y is None or not math.isfinite(y):
                    if seg_xs:
                        function_segments.append((seg_xs, seg_ys, fid))
                        seg_xs, seg_ys = [], []
                    continue
                seg_xs.append(float(x))
                seg_ys.append(float(y))
                bbox_x.append(float(x))
                bbox_y.append(float(y))
            if seg_xs:
                function_segments.append((seg_xs, seg_ys, fid))
        if has_function:
            if points:
                bbox_x = list(point_bbox_x)
                bbox_y = list(point_bbox_y)
                bbox_x.append(0.0)
                bbox_y.append(0.0)
            else:
                bbox_x.append(0.0)
                bbox_y.append(0.0)

        # collect segment incidences and adjacency for label offsetting
        incid: Dict[str, List[str]] = {}
        adj: Dict[str, List[str]] = defaultdict(list)
        for c in self.constraints.values():
            if c.type == "Incidence":
                p = [s for s in c.scope if s in points]
                s = [s for s in c.scope if s not in points]
                for sid in s:
                    incid.setdefault(sid, []).extend(p)
                if len(p) >= 2:
                    for i in range(len(p)):
                        for j in range(i + 1, len(p)):
                            adj[p[i]].append(p[j]); adj[p[j]].append(p[i])

        span_x = (max(bbox_x) - min(bbox_x)) if bbox_x else 1.0
        span_y = (max(bbox_y) - min(bbox_y)) if bbox_y else 1.0
        base_offset = max(span_x, span_y) * 0.03
        base_offset = max(0.03, min(base_offset, 0.4))

        # Unify offset scale to avoid label drifting too far in small coordinate ranges
        def _label_offset(_: str) -> Tuple[float, float]:
            return (base_offset, base_offset)

        for pid, (x, y) in point_xy.items():
            ax.scatter(x, y, s=18, edgecolors="none", facecolors="black", zorder=3)
            dx, dy = _label_offset(pid)
            ax.text(
                x+dx, y+dy, pid, fontsize=11, color="#1f77b4",
                ha="center", va="center", zorder=5,
                path_effects=[patheffects.withStroke(linewidth=3, foreground="white")]
            )

        # draw segments
        tri_edges = set(self.meta.get("triangle_edges", []))
        for sid, plist in incid.items():
            if sid not in self.entities or self.entities[sid].type != "Segment": continue
            if len(plist) < 2: continue
            xy = np.array([[points[p].attrs.data["coord"]["x"], points[p].attrs.data["coord"]["y"]] for p in plist])
            i1 = int(np.argmin(xy.sum(axis=1))); i2 = int(np.argmax(xy.sum(axis=1)))
            if i1 == i2 and len(plist) >= 2:
                i2 = 1 - i1  # handle degeneracy when sums are equal
            x1,y1 = xy[i1]; x2,y2 = xy[i2]
            color = line_color
            lw = 2.0 if sid in tri_edges else 2.4
            ax.plot([x1,x2],[y1,y2], '-', lw=lw, color=color, alpha=0.95, zorder=2,
                    solid_capstyle="round", solid_joinstyle="round")
            if show_labels:
                ax.text((x1+x2)/2,(y1+y2)/2, sid, fontsize=8, color="gray")

        for xs_seg, ys_seg, _ in function_segments:
            if len(xs_seg) < 2:
                continue
            ax.plot(xs_seg, ys_seg, '-', lw=1.8, color=line_color, alpha=0.9, zorder=1.4)

        # visualize pure collinear relations
        seen_collinear: set = set()
        for cons in self.constraints.values():
            if cons.type != "Collinear":
                continue
            pts = [pid for pid in cons.scope if pid in points]
            if len(pts) < 3:
                continue
            key = tuple(sorted(pts))
            if key in seen_collinear:
                continue
            coords = []
            for pid in pts:
                try:
                    coords.append((pid, self.get_point_xy(pid)))
                except Exception:
                    coords = []
                    break
            if len(coords) < 3:
                continue
            base_dir = None
            base_origin = None
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    p1 = coords[i][1]
                    p2 = coords[j][1]
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    norm = math.hypot(dx, dy)
                    if norm > 1e-6:
                        base_dir = (dx / norm, dy / norm)
                        base_origin = p1
                        break
                if base_dir:
                    break
            if base_dir is None or base_origin is None:
                continue
            def _proj(coord):
                vx = coord[0] - base_origin[0]
                vy = coord[1] - base_origin[1]
                return vx * base_dir[0] + vy * base_dir[1]
            coords.sort(key=lambda item: _proj(item[1]))
            xs = [c[1][0] for c in coords]
            ys = [c[1][1] for c in coords]
            ax.plot(xs, ys, '-', lw=1.0, color=line_color, alpha=0.7, zorder=1,
                    solid_capstyle="round", solid_joinstyle="round")
            seen_collinear.add(key)

        # circles
        for eid, e in self.entities.items():
            if e.type == "Circle":
                center_id = e.attrs.data.get("center_id") or e.attrs.data.get("center")
                through_id = e.attrs.data.get("through_id")
                center_xy = self.get_point_xy(center_id) if center_id else None
                radius = None
                if center_xy and through_id:
                    txy = self.get_point_xy(through_id)
                    if txy:
                        dx = txy[0] - center_xy[0]
                        dy = txy[1] - center_xy[1]
                        radius = (dx * dx + dy * dy) ** 0.5
                if radius is None:
                    radius = float(e.attrs.data.get("radius", 1.0))
                if not center_xy:
                    continue
                bbox_x.extend([center_xy[0] - radius, center_xy[0] + radius])
                bbox_y.extend([center_xy[1] - radius, center_xy[1] + radius])
                circ = MPLCircle(center_xy, radius, fill=False, lw=1.6, color=line_color, alpha=0.9)
                ax.add_patch(circ)
                if show_labels:
                    ax.text(center_xy[0]+radius*0.1, center_xy[1]+radius*0.1, eid, fontsize=9, color="#1f77b4")

        # circles implied by Concyclic constraints (no explicit circle entity)
        seen_concyclic = set()
        for cons in self.constraints.values():
            if cons.type != "Concyclic":
                continue
            pts = [pid for pid in cons.scope if pid in points]
            if len(pts) < 3:
                continue
            key = tuple(sorted(pts))
            if key in seen_concyclic:
                continue
            try:
                A = self.get_point_xy(pts[0]); B = self.get_point_xy(pts[1]); C = self.get_point_xy(pts[2])
            except Exception:
                continue
            center = _circumcenter_from_points(A, B, C)
            if center is None:
                continue
            cx, cy = center
            r = math.hypot(cx - A[0], cy - A[1])
            if not math.isfinite(r) or r < self.meta.get("tolerance", 1e-6):
                continue
            bbox_x.extend([cx - r, cx + r])
            bbox_y.extend([cy - r, cy + r])
            circ = MPLCircle((cx, cy), r, fill=False, lw=1.2, color=line_color, alpha=0.65, linestyle="-")
            ax.add_patch(circ)
            if show_labels:
                ax.text(cx + r * 0.08, cy + r * 0.08, cons.id, fontsize=8, color="#1f77b4")
            seen_concyclic.add(key)

        # triangle outline
        for eid, e in self.entities.items():
            if e.type == "Triangle":
                verts = e.attrs.data["verts"]
                if all(v in points for v in verts):
                    xs = [self.get_point_xy(v)[0] for v in verts] + [self.get_point_xy(verts[0])[0]]
                    ys = [self.get_point_xy(v)[1] for v in verts] + [self.get_point_xy(verts[0])[1]]
                    ax.plot(xs, ys, '-', lw=1.8, color=line_color, alpha=0.7,
                            solid_capstyle="round", solid_joinstyle="round")
                    if show_labels:
                        ax.text(np.mean(xs[:-1]), np.mean(ys[:-1]), eid, fontsize=9, color="#777777")

        # angles by name "∠XYZ"
        for eid, e in self.entities.items():
            if e.type != "Angle": continue
            name: str = e.attrs.data.get("name", "")
            if len(name)==4 and name[0]=="∠":
                X,Y,Z = name[1], name[2], name[3]
                if X in points and Y in points and Z in points:
                    A = self.get_point_xy(Y); B = self.get_point_xy(X); C = self.get_point_xy(Z)
                    if show_labels:
                        ax.text(A[0]+0.35, A[1]+0.35, name, fontsize=9, color="#000000")
        ax.set_aspect('equal', adjustable='box')
        if minimal or has_function:
            if bbox_x and bbox_y:
                span_x = max(bbox_x) - min(bbox_x)
                span_y = max(bbox_y) - min(bbox_y)
                pad = max(span_x, span_y) * 0.28 or 0.8
                ax.set_xlim(min(bbox_x) - pad, max(bbox_x) + pad)
                ax.set_ylim(min(bbox_y) - pad, max(bbox_y) + pad)
            if has_function:
                x_min, x_max = ax.get_xlim()
                y_min, y_max = ax.get_ylim()
                axis_color = line_color
                ax.plot([x_min, x_max], [0.0, 0.0], '-', lw=1.1, color=axis_color, alpha=0.9, zorder=0.9)
                ax.plot([0.0, 0.0], [y_min, y_max], '-', lw=1.1, color=axis_color, alpha=0.9, zorder=0.9)
                x_tip = x_max
                x_tail = x_max - (x_max - x_min) * 0.06
                y_tip = y_max
                y_tail = y_max - (y_max - y_min) * 0.06
                ax.annotate(
                    "", xy=(x_tip, 0.0), xytext=(x_tail, 0.0),
                    arrowprops=dict(arrowstyle="->", color=axis_color, lw=1.2),
                    zorder=1.0,
                )
                ax.annotate(
                    "", xy=(0.0, y_tip), xytext=(0.0, y_tail),
                    arrowprops=dict(arrowstyle="->", color=axis_color, lw=1.2),
                    zorder=1.0,
                )
                span_x = x_max - x_min
                span_y = y_max - y_min
                ax.text(x_tip - span_x * 0.04, 0.0 + span_y * 0.03, "x",
                        fontsize=9, color=axis_color, ha="left", va="bottom")
                ax.text(0.0 + span_x * 0.03, y_tip - span_y * 0.04, "y",
                        fontsize=9, color=axis_color, ha="left", va="top")
            if minimal:
                ax.axis('off')
                fig.tight_layout(pad=0.25)
        if not minimal:
            ax.set_title("VeriGraph Rendering")
            fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=240)
        if show:
            plt.show()
        plt.close(fig)
    # === Hard/soft constraint passes ===
    def VeriSolve_incidence(self, tol: Optional[float] = None, ignore_protection: bool = False):
        """Resolve high-priority incidence constraints in one shot to remove ordering effects."""
        import math
        from itertools import combinations
        eff_tol = tol if tol is not None else getattr(CONFIG, "analytic_tol", 1e-6)

        # On the Author path, enable analytic geometry solving to avoid iterative point pushing
        if self.meta.get("author_strict_mode"):
            ok = AnalyticSolver(self, tol=eff_tol, ignore_protection=ignore_protection).solve()
            if not ok:
                raise RuntimeError("AnalyticSolver failed in strict mode")
            return

        point_lookup = _build_point_lookup(self)
        # Reset semantic priority before each solve to ensure overlay rules are recomputed reproducibly
        self.semantic_priority = {
            pid: 0 for pid, ent in self.entities.items() if ent.type == "Point"
        }
        circle_members: Dict[str, set] = defaultdict(set)
        for cons in self.constraints.values():
            if cons.type in ("OnCircle", "PointOnCircle") and len(cons.scope) >= 2:
                p_tok = None; cir = None
                for tok in cons.scope:
                    ent = self.entities.get(tok)
                    if p_tok is None and ent and ent.type == "Point":
                        p_tok = tok
                    elif cir is None and ent and ent.type == "Circle":
                        cir = tok
                if p_tok and cir:
                    circle_members[cir].add(p_tok)
            elif cons.type == "Incidence" and len(cons.scope) >= 2:
                pts = [tok for tok in cons.scope if _is_point_entity(self, tok)]
                ids = [tok for tok in cons.scope if tok not in pts]
                for lid in ids:
                    ent = self.entities.get(lid)
                    if ent and ent.type == "Circle":
                        for p in pts:
                            circle_members[lid].add(p)
        hard_locked = set(self.meta.get("hard_locked_points") or [])
        used_targets: Dict[str, Tuple[float, float]] = {}

        def _record_used(pid: str, xy: Tuple[float, float]):
            used_targets[pid] = (float(xy[0]), float(xy[1]))

        def _lock(pid: str):
            hard_locked.add(pid)
            xy = _get_xy(pid)
            if xy:
                _record_used(pid, xy)

        def _get_xy(pid: str) -> Optional[Tuple[float, float]]:
            try:
                return self.get_point_xy(pid)
            except Exception:
                return None

        def _refresh_incident_entities(pid: str):
            # Update cached directions for any segment/line/ray/vector containing this point, to avoid using stale coordinates
            for ent in self.entities.values():
                if ent.type not in ("Segment", "Line", "Ray", "Vector"):
                    continue
                pts = ent.attrs.data.get("points") or ent.attrs.data.get("verts")
                if not (isinstance(pts, list) and len(pts) >= 2 and pid in pts):
                    continue
                try:
                    p0 = self.get_point_xy(pts[0])
                    p1 = self.get_point_xy(pts[1])
                except Exception:
                    continue
                if not (p0 and p1):
                    continue
                ent.attrs.data.setdefault("points", list(pts))
                ent.attrs.data["cached_coords"] = [
                    {"x": float(p0[0]), "y": float(p0[1])},
                    {"x": float(p1[0]), "y": float(p1[1])},
                ]
                dir_vec = (float(p1[0] - p0[0]), float(p1[1] - p0[1]))
                ent.attrs.data["dir"] = {"dx": dir_vec[0], "dy": dir_vec[1]}
                seg_len = math.hypot(dir_vec[0], dir_vec[1])
                ent.attrs.data["length"] = seg_len
                if seg_len > 1e-12:
                    ent.attrs.data["normal"] = {"dx": -dir_vec[1] / seg_len, "dy": dir_vec[0] / seg_len}
                else:
                    ent.attrs.data["normal"] = {"dx": 0.0, "dy": 0.0}
                if ent.type in ("Line", "Ray", "Vector"):
                    ent.attrs.data["direction"] = {"dx": dir_vec[0], "dy": dir_vec[1]}

        def _set_xy(pid: str, xy: Tuple[float, float], priority: int = 0, lock_point: bool = True) -> bool:
            if pid not in self.entities or self.entities[pid].type != "Point":
                return False
            if not ignore_protection and self._is_protected_point(pid):
                return False
            old_priority = self.semantic_priority.get(pid, 0)
            if priority > 0:
                if priority <= old_priority:
                    return False
            elif old_priority > 0:
                return False
            ent = self.entities[pid]
            ent.attrs.data.setdefault("coord", {"x": 0.0, "y": 0.0})
            ent.attrs.data["coord"]["x"] = float(xy[0])
            ent.attrs.data["coord"]["y"] = float(xy[1])
            if priority > 0:
                self.semantic_priority[pid] = priority
            if lock_point:
                _lock(pid)
            else:
                _record_used(pid, xy)
            _refresh_incident_entities(pid)
            return True

        def _line_points_with_bound(token: str) -> Optional[Tuple[Tuple[str, str], bool]]:
            ent = self.entities.get(token)
            pts: Optional[Tuple[str, str]] = None
            bounded = False
            if ent:
                if ent.type == "Segment":
                    try:
                        pts = _segment_endpoints_for_VeriScript(self, token)
                        bounded = True
                    except Exception:
                        pts = None
                elif ent.type in ("Line", "Ray"):
                    verts = ent.attrs.data.get("points") or ent.attrs.data.get("verts")
                    if isinstance(verts, list) and len(verts) >= 2:
                        pts = (verts[0], verts[1])
            if pts is None and isinstance(token, str):
                upper = token.upper()
                if upper.startswith("SEG_"):
                    bounded = True
                    decoded = _split_token_into_points(point_lookup, upper[4:], 2)
                    if decoded:
                        pts = decoded
            return (pts, bounded) if pts else None

        def _line_intersection(p1: Tuple[float, float], p2: Tuple[float, float],
                               p3: Tuple[float, float], p4: Tuple[float, float]) -> Optional[Tuple[float, float]]:
            x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < eff_tol:
                return None
            det1 = x1 * y2 - y1 * x2
            det2 = x3 * y4 - y3 * x4
            px = (det1 * (x3 - x4) - (x1 - x2) * det2) / denom
            py = (det1 * (y3 - y4) - (y1 - y2) * det2) / denom
            if not (math.isfinite(px) and math.isfinite(py)):
                return None
            return (px, py)

        def _circle_circle_intersections(c1: Tuple[float, float, float, str],
                                         c2: Tuple[float, float, float, str]) -> List[Tuple[float, float]]:
            x0, y0, r0, _ = c1; x1, y1, r1, _ = c2
            dx = x1 - x0; dy = y1 - y0
            d = math.hypot(dx, dy)
            if d < eff_tol:
                return []
            if d > r0 + r1 + eff_tol or d < abs(r0 - r1) - eff_tol:
                return []
            a = (r0 * r0 - r1 * r1 + d * d) / (2 * d)
            h_sq = r0 * r0 - a * a
            if h_sq < -eff_tol:
                return []
            h_sq = max(h_sq, 0.0)
            h = math.sqrt(h_sq)
            xm = x0 + a * dx / d
            ym = y0 + a * dy / d
            rx = -dy * (h / d)
            ry = dx * (h / d)
            pts = [(xm + rx, ym + ry), (xm - rx, ym - ry)]
            if h < eff_tol:
                return [pts[0]]
            return pts

        def _choose_nearest(candidates: List[Tuple[float, float]], ref: Optional[Tuple[float, float]],
                            current_pid: Optional[str] = None,
                            used: Optional[Dict[str, Tuple[float, float]]] = None,
                            unique_tol: float = 1e-6) -> Optional[Tuple[float, float]]:
            if not candidates:
                return None
            used = used or {}
            def _dist2(p, q):
                return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2
            sorted_cands = sorted(candidates, key=lambda pt: _dist2(pt, ref) if ref else 0.0)
            for pt in sorted_cands:
                conflict = False
                for pid, used_xy in used.items():
                    if current_pid is not None and pid == current_pid:
                        continue
                    if _dist2(pt, used_xy) <= unique_tol * unique_tol:
                        conflict = True
                        break
                if conflict:
                    continue
                return pt
            if ref is None:
                return candidates[0]
            return min(candidates, key=lambda pt: _dist2(pt, ref))

        def _on_segment(a: Tuple[float, float], b: Tuple[float, float], q: Tuple[float, float], bound: bool) -> bool:
            if not bound:
                return True
            t = _line_param(a, b, q)
            return t is not None and -1e-6 <= t <= 1.0 + 1e-6

        point_lines: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        point_circles: Dict[str, List[Tuple[float, float, float, str]]] = defaultdict(list)

        for cons in self.constraints.values():
            if cons.type in ("PointOnLine", "PointOnSegment") and len(cons.scope) >= 2:
                # Point on line/segment, order-independent
                p_tok = None
                line_token = None
                for tok in cons.scope:
                    if p_tok is None and _is_point_entity(self, tok):
                        p_tok = tok
                    elif line_token is None:
                        line_token = tok
                if p_tok is None:
                    p_tok, line_token = cons.scope[0], cons.scope[1]
                lp = _line_points_with_bound(line_token) if line_token else None
                if not lp:
                    continue
                pts, bounded = lp
                point_lines[p_tok].append({"pts": pts, "bounded": bounded, "id": line_token})
            elif cons.type == "Incidence" and len(cons.scope) >= 2:
                # Incidence point-line/segment/circle
                pts = [tok for tok in cons.scope if _is_point_entity(self, tok)]
                ids = [tok for tok in cons.scope if tok not in pts]
                if not pts or not ids:
                    continue
                for p_tok in pts:
                    for lid in ids:
                        ent = self.entities.get(lid)
                        if ent and ent.type == "Circle":
                            params = self._circle_params(lid)
                            if params:
                                cx, cy, r, _ = params
                                if r > eff_tol:
                                    point_circles[p_tok].append((cx, cy, r, lid))
                            continue
                        lp = _line_points_with_bound(lid)
                        if not lp:
                            continue
                        pts_pair, bounded = lp
                        point_lines[p_tok].append({"pts": pts_pair, "bounded": bounded, "id": lid})
            elif cons.type in ("OnCircle", "PointOnCircle") and len(cons.scope) >= 2:
                # Point on circle, order-independent
                p_tok = None
                cir = None
                for tok in cons.scope:
                    ent = self.entities.get(tok)
                    if p_tok is None and ent and ent.type == "Point":
                        p_tok = tok
                    elif cir is None and ent and ent.type == "Circle":
                        cir = tok
                if p_tok is None or cir is None:
                    p_tok, cir = cons.scope[0], cons.scope[1]
                params = self._circle_params(cir)
                if params:
                    cx, cy, r, _ = params
                    if r > eff_tol:
                        point_circles[p_tok].append((cx, cy, r, cir))
        target_points = set(point_lines.keys()) | set(point_circles.keys())

        for pid in sorted(target_points):
            if pid not in self.entities or self.entities[pid].type != "Point":
                continue
            current = _get_xy(pid)
            updated = False

            if point_lines.get(pid) and point_circles.get(pid):
                candidates: List[Tuple[float, float]] = []
                for line in point_lines[pid]:
                    pa, pb = _get_xy(line["pts"][0]), _get_xy(line["pts"][1])
                    if not pa or not pb:
                        continue
                    for cir in point_circles[pid]:
                        cx, cy, r, _ = cir
                        for pt in _circle_line_intersections(cx, cy, r, pa, pb):
                            if _on_segment(pa, pb, pt, line["bounded"]):
                                candidates.append(pt)
                target = _choose_nearest(candidates, current, pid, used_targets)
                if target and _set_xy(pid, target):
                    updated = True

            if not updated and len(point_lines.get(pid, [])) >= 2:
                candidates: List[Tuple[float, float]] = []
                for l1, l2 in combinations(point_lines[pid], 2):
                    pa1, pb1 = _get_xy(l1["pts"][0]), _get_xy(l1["pts"][1])
                    pa2, pb2 = _get_xy(l2["pts"][0]), _get_xy(l2["pts"][1])
                    if not (pa1 and pb1 and pa2 and pb2):
                        continue
                    pt = _line_intersection(pa1, pb1, pa2, pb2)
                    if pt and _on_segment(pa1, pb1, pt, l1["bounded"]) and _on_segment(pa2, pb2, pt, l2["bounded"]):
                        candidates.append(pt)
                target = _choose_nearest(candidates, current, pid, used_targets)
                if target and _set_xy(pid, target):
                    updated = True

            if not updated and len(point_circles.get(pid, [])) >= 2:
                candidates: List[Tuple[float, float]] = []
                for c1, c2 in combinations(point_circles[pid], 2):
                    for pt in _circle_circle_intersections(c1, c2):
                        candidates.append(pt)
                target = _choose_nearest(candidates, current, pid, used_targets)
                if target:
                    _set_xy(pid, target)

        # pattern-based semantic intersections (collinear/perp/etc.)
        self._solve_semantic_intersections(
            hard_locked,
            _get_xy,
            _set_xy,
            _line_intersection,
            _circle_line_intersections,
            _circle_circle_intersections,
            _on_segment,
            used_targets,
            tol=eff_tol,
            ignore_protection=ignore_protection,
        )

        self.meta["hard_locked_points"] = sorted(hard_locked)

    def _solve_author_analytic(self, ignore_protection: bool = False, tol: Optional[float] = None):
        """Backwards-compatible wrapper for strict analytic solver."""
        eff_tol = tol if tol is not None else getattr(CONFIG, "analytic_tol", 1e-6)
        ok = AnalyticSolver(self, tol=eff_tol, ignore_protection=ignore_protection).solve()
        if not ok:
            raise RuntimeError("AnalyticSolver failed in author analytic mode")
        return ok


    def _solve_semantic_intersections(
        self,
        hard_locked: set,
        _get_xy,
        _set_xy,
        _line_intersection,
        _circle_line_intersections,
        _circle_circle_intersections,
        _on_segment,
        used_targets: Dict[str, Tuple[float, float]],
        tol: float = 1e-6,
        ignore_protection: bool = False,
    ):
        """Solve semantic points across four Passes using analytic-geometry formulas; locked key points are no longer overwritten."""
        import math

        if self.meta.get("author_strict_mode"):
            return

        point_lookup = _build_point_lookup(self)

        semantic_targets: set = set()
        for cons in self.constraints.values():
            if cons.type == "Midpoint" and cons.scope:
                semantic_targets.add(cons.scope[0])
            elif cons.type == "AngleBisector" and cons.scope:
                try:
                    semantic_targets.update(_segment_endpoints_for_VeriScript(self, cons.scope[0]))
                except Exception:
                    pass
            elif cons.type == "Concyclic" and len(cons.scope) >= 4:
                semantic_targets.add(cons.scope[3])
            elif cons.type in ("Parallel", "Perpendicular") and cons.scope:
                for tok in cons.scope[:2]:
                    try:
                        semantic_targets.update(_segment_endpoints_for_VeriScript(self, tok))
                    except Exception:
                        pass
            elif cons.type == "Collinear":
                for tok in cons.scope:
                    if _is_point_entity(self, tok):
                        semantic_targets.add(tok)

        def _is_semantic_target(pid: str) -> bool:
            return pid in semantic_targets

        def _is_locked(pid: str) -> bool:
            return pid in hard_locked or (not ignore_protection and self._is_protected_point(pid))

        def _is_used_by_other(pt: Tuple[float, float], pid: Optional[str]) -> bool:
            x, y = pt
            for other_pid, (ux, uy) in used_targets.items():
                if pid is not None and other_pid == pid:
                    continue
                if math.hypot(x - ux, y - uy) < tol:
                    return True
            return False

        def _unit(vec: Tuple[float, float]) -> Optional[Tuple[float, float]]:
            vx, vy = vec
            n = math.hypot(vx, vy)
            if n < tol:
                return None
            return (vx / n, vy / n)

        def _pid(label: str) -> Optional[str]:
            upper = label.upper()
            cand = point_lookup.get(upper)
            if cand:
                return cand
            return label if label in self.entities else None

        def _xy(pid: Optional[str]) -> Optional[Tuple[float, float]]:
            if not pid:
                return None
            try:
                return self.get_point_xy(pid)
            except Exception:
                return None

        def _project_point_to_line(pt: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> Optional[Tuple[float, float]]:
            ab = (b[0] - a[0], b[1] - a[1])
            denom = ab[0] * ab[0] + ab[1] * ab[1]
            if denom < tol * tol:
                return None
            t = ((pt[0] - a[0]) * ab[0] + (pt[1] - a[1]) * ab[1]) / denom
            return (a[0] + ab[0] * t, a[1] + ab[1] * t)

        def _circle_from_abc() -> Optional[Tuple[float, float, float]]:
            A_id, B_id, C_id = _pid("A"), _pid("B"), _pid("C")
            A, B, C = _xy(A_id), _xy(B_id), _xy(C_id)
            if not (A and B and C):
                return None
            center = _circumcenter_from_points(A, B, C)
            if not center:
                return None
            cx, cy = center
            r = math.hypot(cx - A[0], cy - A[1])
            if not math.isfinite(r) or r < tol:
                return None
            return (cx, cy, r)

        def _set_locked(pid: Optional[str], xy: Optional[Tuple[float, float]], priority: int) -> bool:
            if pid is None or xy is None:
                return False
            ok = _set_xy(pid, xy, priority=priority, lock_point=True)
            return ok

        def _apply_direction_constraint(target_seg: str, ref_seg: str, priority: int, is_perp: bool = False,
                                        require_semantic: bool = False) -> bool:
            try:
                u, v = _segment_endpoints_for_VeriScript(self, target_seg)
            except Exception:
                return False
            anchor, mover = u, v
            u_locked, v_locked = _is_locked(u), _is_locked(v)
            u_sem, v_sem = _is_semantic_target(u), _is_semantic_target(v)
            u_pri, v_pri = self.semantic_priority.get(u, 0), self.semantic_priority.get(v, 0)
            if require_semantic:
                if u_sem and not u_locked:
                    anchor, mover = v, u
                elif v_sem and not v_locked:
                    anchor, mover = u, v
                else:
                    return False
            else:
                if u_locked and not v_locked:
                    anchor, mover = u, v
                elif v_locked and not u_locked:
                    anchor, mover = v, u
                elif u_pri > v_pri and not v_locked:
                    anchor, mover = u, v
                elif v_pri > u_pri and not u_locked:
                    anchor, mover = v, u
                elif u_sem and not u_locked and not v_sem:
                    anchor, mover = v, u
                elif v_sem and not v_locked and not u_sem:
                    anchor, mover = u, v
                else:
                    anchor, mover = (u, v) if u < v else (v, u)
            if _is_locked(mover):
                return False
            anchor_xy = _get_xy(anchor)
            if not anchor_xy:
                return False
            try:
                a_ref, b_ref = _segment_endpoints_for_VeriScript(self, ref_seg)
                ref_a, ref_b = _get_xy(a_ref), _get_xy(b_ref)
            except Exception:
                return False
            if not (ref_a and ref_b):
                return False
            ref_dir = (ref_b[0] - ref_a[0], ref_b[1] - ref_a[1])
            if is_perp:
                ref_dir = (-ref_dir[1], ref_dir[0])
            dir_unit = _unit(ref_dir)
            if not dir_unit:
                return False
            mover_xy = _get_xy(mover)
            length = None
            if mover_xy:
                length = math.hypot(mover_xy[0] - anchor_xy[0], mover_xy[1] - anchor_xy[1])
            if not length or length < tol:
                length = 1.0
            tgt = (anchor_xy[0] + dir_unit[0] * length, anchor_xy[1] + dir_unit[1] * length)
            if _is_used_by_other(tgt, mover):
                return False
            return _set_xy(mover, tgt, priority=priority, lock_point=False)

        # Pass 1 — Midpoint + base directions (anchors fixed)
        for cons in self.constraints.values():
            if cons.type != "Midpoint" or len(cons.scope) < 2:
                continue
            m, seg = cons.scope[0], cons.scope[1]
            try:
                u, v = _segment_endpoints_for_VeriScript(self, seg)
                uxy, vxy = _get_xy(u), _get_xy(v)
            except Exception:
                continue
            if not (uxy and vxy):
                continue
            mid = ((uxy[0] + vxy[0]) / 2.0, (uxy[1] + vxy[1]) / 2.0)
            _set_locked(m, mid, priority=1)

        for cons in self.constraints.values():
            if cons.type not in ("Parallel", "Perpendicular") or len(cons.scope) < 2:
                continue
            ok = _apply_direction_constraint(cons.scope[0], cons.scope[1], priority=1,
                                             is_perp=(cons.type == "Perpendicular"), require_semantic=True)
            if not ok:
                _apply_direction_constraint(cons.scope[1], cons.scope[0], priority=1,
                                            is_perp=(cons.type == "Perpendicular"), require_semantic=True)

        # Pass 2 — foot of perpendicular / angle bisector
        A_id, B_id, C_id = _pid("A"), _pid("B"), _pid("C")
        H_id, L_id = _pid("H"), _pid("L")
        A_xy, B_xy, C_xy = _xy(A_id), _xy(B_id), _xy(C_id)
        if A_xy and B_xy and C_xy:
            if H_id and _is_semantic_target(H_id) and not _is_locked(H_id):
                foot = _project_point_to_line(A_xy, B_xy, C_xy)
                _set_locked(H_id, foot, priority=3)
            if L_id and _is_semantic_target(L_id) and not _is_locked(L_id):
                dir_ab = _unit((B_xy[0] - A_xy[0], B_xy[1] - A_xy[1]))
                dir_ac = _unit((C_xy[0] - A_xy[0], C_xy[1] - A_xy[1]))
                if dir_ab and dir_ac:
                    bis = (dir_ab[0] + dir_ac[0], dir_ab[1] + dir_ac[1])
                    bis_unit = _unit(bis)
                    if bis_unit:
                        tgt = _line_intersection(A_xy, (A_xy[0] + bis_unit[0], A_xy[1] + bis_unit[1]), B_xy, C_xy)
                        _set_locked(L_id, tgt, priority=3)

        # Pass 3 — second intersection P / tangent intersection S
        P_id = _pid("P")
        L_xy = _xy(L_id)
        if P_id and _is_semantic_target(P_id) and A_xy and B_xy and C_xy and L_xy and not _is_locked(P_id):
            circle_params = None
            # Prefer existing circles
            for cid, ent in self.entities.items():
                if ent.type != "Circle":
                    continue
                params = self._circle_params(cid)
                if params:
                    circle_params = params
                    break
            if circle_params is None:
                circle_params = _circle_from_abc()
            if circle_params:
                if len(circle_params) == 4:
                    cx, cy, r, _ = circle_params
                else:
                    cx, cy, r = circle_params
                inters = _circle_line_intersections(cx, cy, r, A_xy, L_xy)
                chosen = None
                for pt in inters:
                    if (pt[0] - A_xy[0]) ** 2 + (pt[1] - A_xy[1]) ** 2 <= tol * tol:
                        continue
                    chosen = pt
                    break
                if chosen and not _is_used_by_other(chosen, P_id):
                    _set_locked(P_id, chosen, priority=3)

        O_id, S_id, R_id = _pid("O"), _pid("S"), _pid("R")
        O_xy = _xy(O_id)
        if A_xy and B_xy and C_xy and O_xy:
            ao = (O_xy[0] - A_xy[0], O_xy[1] - A_xy[1])
            tan_dir = _unit((-ao[1], ao[0]))
            if tan_dir:
                scale = max(math.hypot(B_xy[0] - C_xy[0], B_xy[1] - C_xy[1]), math.hypot(ao[0], ao[1]), 1.0)
                if R_id and _is_semantic_target(R_id) and not _is_locked(R_id):
                    r_pt = (A_xy[0] + tan_dir[0] * scale * 0.5, A_xy[1] + tan_dir[1] * scale * 0.5)
                    _set_xy(R_id, r_pt, priority=1, lock_point=False)
                if S_id and _is_semantic_target(S_id) and not _is_locked(S_id):
                    s_pt = _line_intersection(A_xy, (A_xy[0] + tan_dir[0], A_xy[1] + tan_dir[1]), B_xy, C_xy)
                    _set_locked(S_id, s_pt, priority=3)

        # Pass 4 — global direction correction (remaining parallel/perpendicular; anchor priority controls single-end movement)
        for cons in self.constraints.values():
            if cons.type not in ("Parallel", "Perpendicular") or len(cons.scope) < 2:
                continue
            updated = _apply_direction_constraint(
                cons.scope[0], cons.scope[1],
                priority=2,
                is_perp=(cons.type == "Perpendicular"),
                require_semantic=False
            )
            if not updated:
                _apply_direction_constraint(
                    cons.scope[1], cons.scope[0],
                    priority=2,
                    is_perp=(cons.type == "Perpendicular"),
                    require_semantic=False
                )


    def polish_soft_constraints(self, max_iter: int = 8, tol: float = 1e-4, ignore_protection: bool = False):
        """Gently polish soft constraints without moving protected or hard-locked points."""
        import math
        from itertools import combinations

        hard_locked = set(self.meta.get("hard_locked_points") or [])
        point_lookup = _build_point_lookup(self)

        def _is_locked(pid: str) -> bool:
            return pid in hard_locked or self._is_protected_point(pid)

        def _can_move(pid: str) -> bool:
            if pid not in self.entities or self.entities[pid].type != "Point":
                return False
            if _is_locked(pid):
                return False
            return True

        def _get_xy(pid: str) -> Optional[Tuple[float, float]]:
            try:
                return self.get_point_xy(pid)
            except Exception:
                return None

        def _move(pid: str, target: Tuple[float, float], blend: float = 0.35) -> float:
            if not _can_move(pid):
                return 0.0
            ent = self.entities[pid]
            ox = ent.attrs.data["coord"]["x"]
            oy = ent.attrs.data["coord"]["y"]
            nx = ox + (target[0] - ox) * blend
            ny = oy + (target[1] - oy) * blend
            ent.attrs.data["coord"]["x"] = float(nx)
            ent.attrs.data["coord"]["y"] = float(ny)
            return abs(nx - ox) + abs(ny - oy)

        def _dist(p, q):
            return math.hypot(p[0] - q[0], p[1] - q[1])

        def _angle(u, v):
            nu = math.hypot(u[0], u[1]); nv = math.hypot(v[0], v[1])
            if nu < 1e-12 or nv < 1e-12:
                return None
            c = max(-1.0, min(1.0, (u[0] * v[0] + u[1] * v[1]) / (nu * nv)))
            return math.degrees(math.acos(c))

        def _get_seg_endpoints(seg_id: str) -> Optional[Tuple[str, str]]:
            try:
                return _segment_endpoints_for_VeriScript(self, seg_id)
            except Exception:
                return None

        def _angle_vertices(token: str) -> Optional[Tuple[str, str, str]]:
            return self._angle_vertices_from_token(token, point_lookup)

        def _angle_measure(verts: Tuple[str, str, str]) -> Optional[float]:
            try:
                a, b, c = verts
                ax, ay = self.get_point_xy(a)
                bx, by = self.get_point_xy(b)
                cx, cy = self.get_point_xy(c)
            except Exception:
                return None
            v1 = (bx - ax, by - ay)
            v2 = (bx - cx, by - cy)
            return _angle(v1, v2)

        def _fit_circle_from_points(pid_list: list):
            coords = []
            for pid in pid_list:
                if pid not in self.entities or self.entities[pid].type != "Point":
                    continue
                try:
                    coords.append((pid, self.get_point_xy(pid)))
                except Exception:
                    continue
            if len(coords) < 2:
                return None, None
            for (pid1, p1), (pid2, p2), (pid3, p3) in combinations(coords, 3):
                center = _circumcenter_from_points(p1, p2, p3)
                if center is None:
                    continue
                radius = math.hypot(center[0] - p1[0], center[1] - p1[1])
                if radius > 1e-9 and math.isfinite(radius):
                    return center, radius
            (pid1, p1), (pid2, p2) = coords[0], coords[1]
            mid = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
            radius = _dist(p1, p2) / 2.0
            if radius > 1e-9:
                return mid, radius
            return None, None

        def _project_soft(center, radius, point_ids, blend=0.45):
            moved = 0.0
            if center is None or radius is None or radius < 1e-9:
                return moved
            cx, cy = center
            for pid in point_ids:
                if not _can_move(pid):
                    continue
                try:
                    px, py = self.get_point_xy(pid)
                except Exception:
                    continue
                vx, vy = px - cx, py - cy
                norm = math.hypot(vx, vy)
                if norm < 1e-9:
                    continue
                target = (cx + vx / norm * radius, cy + vy / norm * radius)
                moved += _move(pid, target, blend=blend)
            return moved

        def _constraints_sorted():
            return sorted(self.constraints.values(), key=lambda c: (c.type, str(c.id)))

        for _ in range(max_iter):
            total = 0.0
            for c in _constraints_sorted():
                if c.type == "EqualLength":
                    s1, s2 = c.scope
                    ep1 = _get_seg_endpoints(s1); ep2 = _get_seg_endpoints(s2)
                    if not ep1 or not ep2:
                        continue
                    A, B = ep1; C, D = ep2
                    pa, pb = self.get_point_xy(A), self.get_point_xy(B)
                    pc, pd = self.get_point_xy(C), self.get_point_xy(D)
                    L1, L2 = _dist(pa, pb), _dist(pc, pd)
                    if L1 < 1e-9 or L2 < 1e-9:
                        continue
                    v = (pd[0] - pc[0], pd[1] - pc[1])
                    nv = math.hypot(v[0], v[1])
                    if nv < 1e-9:
                        continue
                    unit = (v[0] / nv, v[1] / nv)
                    targetD = (pc[0] + unit[0] * L1, pc[1] + unit[1] * L1)
                    if _can_move(D):
                        total += _move(D, targetD, blend=0.4)
                    elif _can_move(C):
                        targetC = (pd[0] - unit[0] * L1, pd[1] - unit[1] * L1)
                        total += _move(C, targetC, blend=0.4)

                elif c.type == "Parallel":
                    s1, s2 = c.scope
                    ep1 = _get_seg_endpoints(s1); ep2 = _get_seg_endpoints(s2)
                    if not ep1 or not ep2:
                        continue
                    A, B = ep1; C, D = ep2
                    pa, pb = self.get_point_xy(A), self.get_point_xy(B)
                    pc, pd = self.get_point_xy(C), self.get_point_xy(D)
                    v1 = (pb[0] - pa[0], pb[1] - pa[1])
                    v2 = (pd[0] - pc[0], pd[1] - pc[1])
                    ang = _angle(v1, v2)
                    if ang is None:
                        continue
                    err = min(ang, abs(180.0 - ang))
                    if err > 0.1:
                        L = _dist(pc, pd)
                        n1 = math.hypot(v1[0], v1[1])
                        if n1 < 1e-9:
                            continue
                        unit1 = (v1[0] / n1, v1[1] / n1)
                        targetD = (pc[0] + unit1[0] * L, pc[1] + unit1[1] * L)
                        if _can_move(D):
                            total += _move(D, targetD, blend=0.35)
                        elif _can_move(C):
                            targetC = (pd[0] - unit1[0] * L, pd[1] - unit1[1] * L)
                            total += _move(C, targetC, blend=0.35)

                elif c.type == "Perpendicular":
                    s1, s2 = c.scope
                    ep1 = _get_seg_endpoints(s1); ep2 = _get_seg_endpoints(s2)
                    if not ep1 or not ep2:
                        continue
                    A, B = ep1; C, D = ep2
                    pa, pb = self.get_point_xy(A), self.get_point_xy(B)
                    pc, pd = self.get_point_xy(C), self.get_point_xy(D)
                    v1 = (pb[0] - pa[0], pb[1] - pa[1])
                    v2 = (pd[0] - pc[0], pd[1] - pc[1])
                    ang = _angle(v1, v2)
                    if ang is None:
                        continue
                    err = abs(90.0 - ang)
                    if err > 0.2:
                        L = _dist(pc, pd)
                        n1 = math.hypot(v1[0], v1[1])
                        if n1 < 1e-9:
                            continue
                        vn = (-v1[1] / n1, v1[0] / n1)
                        if _can_move(D):
                            targetD = (pc[0] + vn[0] * L, pc[1] + vn[1] * L)
                            total += _move(D, targetD, blend=0.35)
                        elif _can_move(C):
                            targetC = (pd[0] - vn[0] * L, pd[1] - vn[1] * L)
                            total += _move(C, targetC, blend=0.35)

                elif c.type == "EqualAngle" and len(c.scope) >= 2:
                    verts1 = _angle_vertices(c.scope[0])
                    verts2 = _angle_vertices(c.scope[1])
                    if not (verts1 and verts2):
                        continue
                    target_ang = _angle_measure(verts2)
                    current_ang = _angle_measure(verts1)
                    if target_ang is None or current_ang is None:
                        continue
                    diff = target_ang - current_ang
                    if abs(diff) < 1e-3:
                        continue
                    A, B, C = verts1
                    movable = A if _can_move(A) else (C if _can_move(C) else None)
                    if movable is None:
                        continue
                    center_xy = _get_xy(B)
                    anchor_xy = _get_xy(C if movable == A else A)
                    mv_xy = _get_xy(movable)
                    if not (center_xy and anchor_xy and mv_xy):
                        continue
                    vec = (mv_xy[0] - center_xy[0], mv_xy[1] - center_xy[1])
                    if math.hypot(vec[0], vec[1]) < 1e-9:
                        continue
                    step_rad = math.radians(diff) * 0.35
                    step_rad = max(-math.pi / 6, min(math.pi / 6, step_rad))
                    cos_t = math.cos(step_rad); sin_t = math.sin(step_rad)
                    rotated = (vec[0] * cos_t - vec[1] * sin_t, vec[0] * sin_t + vec[1] * cos_t)
                    target = (center_xy[0] + rotated[0], center_xy[1] + rotated[1])
                    total += _move(movable, target, blend=0.45)

                elif c.type == "Concyclic":
                    pts = [p for p in c.scope if p in self.entities and self.entities[p].type == "Point"]
                    if len(pts) < 3:
                        continue
                    center, radius = _fit_circle_from_points(pts)
                    total += _project_soft(center, radius, pts, blend=0.4)

            if total < tol:
                break

    def enforce_constraints(self, max_iter=8, tol=1e-4, ignore_protection: bool = False):
        """Soft/heuristic polishing only. Call VeriSolve_incidence() separately for exact placement."""
        self.polish_soft_constraints(max_iter=max_iter, tol=tol, ignore_protection=ignore_protection)

    def verify_constraints(self, ang_tol_deg=0.5, len_tol=1e-3, circle_tol=1e-3):
        import math
        def _dist(P, Q):
            return math.hypot(P[0] - Q[0], P[1] - Q[1])
        def _vector(u, v):
            x1, y1 = self.get_point_xy(u)
            x2, y2 = self.get_point_xy(v)
            return (x2 - x1, y2 - y1)
        def _vec_err(u, v):
            return math.hypot(u[0] - v[0], u[1] - v[1])

        # Segment endpoint lookup
        def _seg_endpoints(sid):
            incs = [c for c in self.constraints.values() if c.type == "Incidence" and sid in c.scope]
            pts = [s for c in incs for s in c.scope
                   if s != sid and s in self.entities and self.entities[s].type == "Point"]
            if len(pts) < 2: return None
            xy = [self.get_point_xy(p) for p in pts]
            i1 = int(min(range(len(pts)), key=lambda i: xy[i][0] + xy[i][1]))
            i2 = int(max(range(len(pts)), key=lambda i: xy[i][0] + xy[i][1]))
            return pts[i1], pts[i2]

        def _angle(u, v):
            nu = math.hypot(u[0], u[1]); 
            nv = math.hypot(v[0], v[1])
            if nu < 1e-12 or nv < 1e-12: return None
            c = max(-1.0, min(1.0, (u[0] * v[0] + u[1] * v[1]) / (nu * nv)))
            return math.degrees(math.acos(c))

        point_lookup = _build_point_lookup(self)

        def _angle_vertices_from_token(token: str) -> Optional[Tuple[str, str, str]]:
            return self._angle_vertices_from_token(token, point_lookup)

        def _angle_measure(vertices: Tuple[str, str, str]) -> Optional[float]:
            try:
                a, b, c = vertices
                v1 = _vector(b, a)
                v2 = _vector(b, c)
            except Exception:
                return None
            return _angle(v1, v2)

        def _segment_points(seg_id: str) -> List[str]:
            pts: List[str] = []
            for cons in self.constraints.values():
                if cons.type != "Incidence" or seg_id not in cons.scope:
                    continue
                for tok in cons.scope:
                    if tok == seg_id:
                        continue
                    ent = self.entities.get(tok)
                    if ent and ent.type == "Point" and tok not in pts:
                        pts.append(tok)
            return pts

        def _resolve_line_points(line_id: str) -> Optional[Tuple[str, str]]:
            ent = self.entities.get(line_id)
            if ent:
                if ent.type == "Segment":
                    ep = _seg_endpoints(line_id)
                    if ep:
                        return ep
                elif ent.type == "Line":
                    verts = ent.attrs.data.get("points") or ent.attrs.data.get("verts")
                    if isinstance(verts, list) and len(verts) >= 2:
                        return verts[0], verts[1]
            token = (line_id or "").strip()
            if not token:
                return None
            upper = token.upper()
            if upper.startswith("SEG_"):
                body = upper[4:]
                pts = _split_token_into_points(point_lookup, body, 2)
                if pts:
                    return pts[0], pts[1]
            return None

        def _point_line_distance(pid: str, A: str, B: str) -> Optional[float]:
            try:
                px, py = self.get_point_xy(pid)
                ax, ay = self.get_point_xy(A)
                bx, by = self.get_point_xy(B)
            except Exception:
                return None
            vx, vy = bx - ax, by - ay
            norm = math.hypot(vx, vy)
            if norm < 1e-9:
                return None
            cross = abs((px - ax) * vy - (py - ay) * vx)
            return cross / norm

        def _projection_param(pid: str, A: str, B: str) -> Optional[float]:
            try:
                px, py = self.get_point_xy(pid)
                ax, ay = self.get_point_xy(A)
                bx, by = self.get_point_xy(B)
            except Exception:
                return None
            vx, vy = bx - ax, by - ay
            denom = vx * vx + vy * vy
            if denom < 1e-9:
                return None
            return ((px - ax) * vx + (py - ay) * vy) / denom

        symbolic_helper = SymbolicConstraintHelper(
            self,
            lambda sid: _segment_points(sid) or [],
            lambda lid: _resolve_line_points(lid),
        )

        def _symbolic_rescue(current_ok, current_msg, constraint_type, scope_tokens):
            if current_ok is False and constraint_type in SymbolicConstraintHelper.SUPPORTED:
                sym_ok, sym_msg = symbolic_helper.verify(constraint_type, list(scope_tokens))
                if sym_ok is not None:
                    new_msg = current_msg or ""
                    if sym_msg:
                        new_msg = (new_msg + " | " if new_msg else "") + f"symbolic:{sym_msg}"
                    return sym_ok, new_msg
            return current_ok, current_msg

        def _triangle_vertices_from_token(token: str) -> Optional[Tuple[str, str, str]]:
            tok = (token or "").strip()
            if not tok:
                return None
            ent = self.entities.get(tok)
            if ent and ent.type == "Triangle":
                verts = ent.attrs.data.get("verts") or []
                if isinstance(verts, list) and len(verts) >= 3:
                    return tuple(verts[:3])
            upper = tok.upper()
            body = upper.split("_", 1)[1] if upper.startswith("TRI_") and len(upper) > 4 else upper
            pts = _split_token_into_points(point_lookup, body, 3)
            if pts:
                return pts
            if len(body) >= 3:
                guess = tuple(point_lookup.get(ch, ch) for ch in body[:3])
                return guess
            return None

        def _triangle_side_lengths(verts: Tuple[str, str, str]) -> Optional[Tuple[float, float, float]]:
            try:
                A, B, C = verts
                return (
                    _dist(self.get_point_xy(A), self.get_point_xy(B)),
                    _dist(self.get_point_xy(B), self.get_point_xy(C)),
                    _dist(self.get_point_xy(C), self.get_point_xy(A)),
                )
            except Exception:
                return None

        def _triangle_angles(verts: Tuple[str, str, str]) -> Optional[Tuple[float, float, float]]:
            angles = []
            order = [(1, 0, 2), (0, 1, 2), (0, 2, 1)]
            for idxs in order:
                triplet = (verts[idxs[0]], verts[idxs[1]], verts[idxs[2]])
                ang = _angle_measure(triplet)
                if ang is None:
                    return None
                angles.append(ang)
            return tuple(angles)

        circle_point_cache: Dict[str, List[str]] = {}

        def _points_on_circle(circle_id: str) -> List[str]:
            if circle_id in circle_point_cache:
                return circle_point_cache[circle_id]
            pts: List[str] = []
            for cons in self.constraints.values():
                if cons.type not in ("OnCircle", "Concyclic") or circle_id not in cons.scope:
                    continue
                for tok in cons.scope:
                    if tok == circle_id:
                        continue
                    ent = self.entities.get(tok)
                    if ent and ent.type == "Point" and tok not in pts:
                        pts.append(tok)
            circle_point_cache[circle_id] = pts
            return pts

        report = {}  # cid -> {ok:bool, msg:str}

        for cid, c in list(self.constraints.items()):
            t, sc = c.type, c.scope
            ok, msg = True, ""
            extra_info = None

            if t in ("Concyclic", "OnCircle"):
                # Old data may contain multi-point Concyclic constraints (scope of 3/4 points); the original logic that defaulted to unpacking only two items would raise directly.
                if len(sc) != 2:
                    ok, msg = True, f"skip-{t}-unsupported-scope-{len(sc)}"
                else:
                    pid, cir = sc
                    params = self._circle_params(cir)
                    if not params:
                        ok, msg = False, "circle-missing"
                    else:
                        cx, cy, r, _ = params
                        px, py = self.get_point_xy(pid)
                        ok = abs(_dist((px, py), (cx, cy)) - r) <= circle_tol
                        msg = f"Δr={abs(_dist((px, py), (cx, cy)) - r):.3e}"

            elif t == "EqualLength":
                s1, s2 = sc
                ep1, ep2 = _seg_endpoints(s1), _seg_endpoints(s2)
                if not ep1 or not ep2:
                    ok, msg = False, "segment-endpoints-missing"
                else:
                    A, B = ep1;
                    C, D = ep2
                    L1 = _dist(self.get_point_xy(A), self.get_point_xy(B))
                    L2 = _dist(self.get_point_xy(C), self.get_point_xy(D))
                    ok = abs(L1 - L2) <= len_tol
                    msg = f"|ΔL|={abs(L1 - L2):.3e}"

            elif t == "Parallel":
                s1, s2 = sc
                ep1, ep2 = _seg_endpoints(s1), _seg_endpoints(s2)
                if not ep1 or not ep2:
                    ok, msg = False, "segment-endpoints-missing"
                else:
                    A, B = ep1;
                    C, D = ep2
                    v1 = (self.get_point_xy(B)[0] - self.get_point_xy(A)[0],
                          self.get_point_xy(B)[1] - self.get_point_xy(A)[1])
                    v2 = (self.get_point_xy(D)[0] - self.get_point_xy(C)[0],
                          self.get_point_xy(D)[1] - self.get_point_xy(C)[1])
                    ang = _angle(v1, v2)
                    if ang is None:
                        ok, msg = False, "degenerate"
                    else:
                        err = min(ang, abs(180.0 - ang))
                        ok = err <= ang_tol_deg
                        msg = f"∥ err={err:.2e}°"

            elif t == "Perpendicular":
                s1, s2 = sc
                ep1, ep2 = _seg_endpoints(s1), _seg_endpoints(s2)
                if not ep1 or not ep2:
                    ok, msg = False, "segment-endpoints-missing"
                else:
                    A, B = ep1;
                    C, D = ep2
                    v1 = (self.get_point_xy(B)[0] - self.get_point_xy(A)[0],
                          self.get_point_xy(B)[1] - self.get_point_xy(A)[1])
                    v2 = (self.get_point_xy(D)[0] - self.get_point_xy(C)[0],
                          self.get_point_xy(D)[1] - self.get_point_xy(C)[1])
                    ang = _angle(v1, v2)
                    if ang is None:
                        ok, msg = False, "degenerate"
                    else:
                        err = abs(90.0 - ang)
                        ok = err <= ang_tol_deg
                        msg = f"⟂ err={err:.2e}°"

            elif t == "VectorSum":
                if len(sc) != 6:
                    ok, msg = False, "scope-mismatch"
                else:
                    try:
                        target = _vector(sc[0], sc[1])
                        left = _vector(sc[2], sc[3])
                        right = _vector(sc[4], sc[5])
                    except Exception:
                        ok, msg = False, "point-missing"
                    else:
                        sum_vec = (left[0] + right[0], left[1] + right[1])
                        err = _vec_err(target, sum_vec)
                        ok = err <= len_tol
                        msg = f"|Δv|={err:.3e}"

            elif t == "VectorDiff":
                if len(sc) != 6:
                    ok, msg = False, "scope-mismatch"
                else:
                    try:
                        target = _vector(sc[0], sc[1])
                        left = _vector(sc[2], sc[3])
                        right = _vector(sc[4], sc[5])
                    except Exception:
                        ok, msg = False, "point-missing"
                    else:
                        diff_vec = (left[0] - right[0], left[1] - right[1])
                        err = _vec_err(target, diff_vec)
                        ok = err <= len_tol
                        msg = f"|Δv|={err:.3e}"

            elif t == "VectorDotZero":
                if len(sc) != 4:
                    ok, msg = False, "scope-mismatch"
                else:
                    try:
                        v1 = _vector(sc[0], sc[1])
                        v2 = _vector(sc[2], sc[3])
                    except Exception:
                        ok, msg = False, "point-missing"
                    else:
                        dot = v1[0] * v2[0] + v1[1] * v2[1]
                        ok = abs(dot) <= len_tol
                        msg = f"|dot|={abs(dot):.3e}"
            elif t == "VectorLinear":
                if len(sc) < 5 or (len(sc) - 2) % 3 != 0:
                    ok, msg = False, "scope-mismatch"
                else:
                    try:
                        target = _vector(sc[0], sc[1])
                        acc = (0.0, 0.0)
                        for idx in range(2, len(sc), 3):
                            k = _author_parse_float(sc[idx], "coeff")
                            vec = _vector(sc[idx + 1], sc[idx + 2])
                            acc = (acc[0] + k * vec[0], acc[1] + k * vec[1])
                        err = _vec_err(target, acc)
                        ok = err <= len_tol
                        msg = f"|Δv|={err:.3e}"
                    except Exception:
                        ok, msg = False, "point-or-coeff-missing"

            elif t == "EqualAngle":
                if len(sc) != 2:
                    ok, msg = False, "scope-mismatch"
                else:
                    verts1 = _angle_vertices_from_token(sc[0])
                    verts2 = _angle_vertices_from_token(sc[1])
                    if not verts1 or not verts2:
                        ok, msg = False, "angle-missing"
                    else:
                        ang1 = _angle_measure(verts1)
                        ang2 = _angle_measure(verts2)
                        if ang1 is None or ang2 is None:
                            ok, msg = False, "degenerate"
                        else:
                            err = abs(ang1 - ang2)
                            ok = err <= ang_tol_deg
                            msg = f"∠ err={err:.2e}°"

            elif t == "Midpoint":
                m, seg = sc
                ep = _seg_endpoints(seg)
                if not ep:
                    ok, msg = False, "segment-endpoints-missing"
                else:
                    U, V = ep
                    MU = _dist(self.get_point_xy(m), self.get_point_xy(U))
                    MV = _dist(self.get_point_xy(m), self.get_point_xy(V))
                    ok = abs(MU - MV) <= len_tol
                    msg = f"|MU-MV|={abs(MU - MV):.3e}"

            elif t == "AngleBisector":
                # seg E? , angle ∠XYZ
                seg, ang = sc
                name = self.entities[ang].attrs.data.get("name", "")
                if len(name) == 4 and name[0] == "∠":
                    X, Y, Z = name[1], name[2], name[3]
                    # Find the dividing point on the angle bisector (the point connected to Y)
                    ep = _seg_endpoints(seg)
                    if not ep:
                        ok, msg = False, "bisector-seg-missing"
                    else:
                        U, V = ep
                        # Pick the endpoint that is not Y as E
                        E = U if U != Y else V
                        a1 = _angle(
                            (self.get_point_xy(X)[0] - self.get_point_xy(Y)[0],
                             self.get_point_xy(X)[1] - self.get_point_xy(Y)[1]),
                            (self.get_point_xy(E)[0] - self.get_point_xy(Y)[0],
                             self.get_point_xy(E)[1] - self.get_point_xy(Y)[1])
                        )
                        a2 = _angle(
                            (self.get_point_xy(E)[0] - self.get_point_xy(Y)[0],
                             self.get_point_xy(E)[1] - self.get_point_xy(Y)[1]),
                            (self.get_point_xy(Z)[0] - self.get_point_xy(Y)[0],
                             self.get_point_xy(Z)[1] - self.get_point_xy(Y)[1])
                        )
                        if a1 is None or a2 is None:
                            ok, msg = False, "degenerate"
                        else:
                            err = abs(a1 - a2)
                            ok = err <= ang_tol_deg
                            msg = f"bisect err={err:.2e}°"

            elif t == "Collinear":
                if len(sc) != 3:
                    ok, msg = False, "scope-mismatch"
                else:
                    try:
                        pts = [self.get_point_xy(p) for p in sc]
                    except Exception:
                        ok, msg = False, "point-missing"
                    else:
                        area2 = abs(
                            (pts[1][0] - pts[0][0]) * (pts[2][1] - pts[0][1])
                            - (pts[1][1] - pts[0][1]) * (pts[2][0] - pts[0][0])
                        )
                        base = max(
                            math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]),
                            math.hypot(pts[2][0] - pts[0][0], pts[2][1] - pts[0][1]),
                            math.hypot(pts[2][0] - pts[1][0], pts[2][1] - pts[1][1]),
                            1.0,
                        )
                        ok = area2 <= len_tol * base
                        msg = f"|area2|={area2:.3e}"

            elif t == "PointOnLine":
                if len(sc) != 2:
                    ok, msg = False, "scope-mismatch"
                else:
                    point, line_id = sc
                    refs = _resolve_line_points(line_id)
                    if not refs:
                        ok, msg = False, "line-endpoints-missing"
                    else:
                        dist_line = _point_line_distance(point, refs[0], refs[1])
                        if dist_line is None:
                            ok, msg = False, "degenerate"
                        else:
                            ok = dist_line <= len_tol
                            msg = f"dist={dist_line:.3e}"

            elif t == "PointOnSegment":
                if len(sc) != 2:
                    ok, msg = False, "scope-mismatch"
                else:
                    point, seg_id = sc
                    refs = _resolve_line_points(seg_id)
                    if not refs:
                        ok, msg = False, "segment-endpoints-missing"
                    else:
                        dist_line = _point_line_distance(point, refs[0], refs[1])
                        param = _projection_param(point, refs[0], refs[1])
                        if dist_line is None or param is None:
                            ok, msg = False, "degenerate"
                        else:
                            within = -1e-3 <= param <= 1 + 1e-3
                            ok = dist_line <= len_tol and within
                            msg = f"dist={dist_line:.3e}, t={param:.3f}"

            elif t == "SimilarTriangles":
                if len(sc) != 2:
                    ok, msg = False, "scope-mismatch"
                else:
                    tri1 = _triangle_vertices_from_token(sc[0])
                    tri2 = _triangle_vertices_from_token(sc[1])
                    if not tri1 or not tri2:
                        ok, msg = False, "triangle-missing"
                    else:
                        lengths1 = _triangle_side_lengths(tri1)
                        lengths2 = _triangle_side_lengths(tri2)
                        angles1 = _triangle_angles(tri1)
                        angles2 = _triangle_angles(tri2)
                        if not lengths1 or not lengths2 or not angles1 or not angles2:
                            ok, msg = False, "degenerate"
                        elif any(L <= len_tol for L in lengths1 + lengths2):
                            ok, msg = False, "degenerate"
                        else:
                            ratios = [l1 / l2 for l1, l2 in zip(lengths1, lengths2)]
                            avg_ratio = sum(ratios) / 3.0
                            rel_err = max(abs(r - avg_ratio) for r in ratios) / max(abs(avg_ratio), 1e-6)
                            ang_err = max(abs(a - b) for a, b in zip(angles1, angles2))
                            ok = rel_err <= 5e-3 and ang_err <= ang_tol_deg
                            msg = f"rel_err={rel_err:.2e}, Δang={ang_err:.2e}°"

            elif t == "CongruentTriangles":
                if len(sc) != 2:
                    ok, msg = False, "scope-mismatch"
                else:
                    tri1 = _triangle_vertices_from_token(sc[0])
                    tri2 = _triangle_vertices_from_token(sc[1])
                    if not tri1 or not tri2:
                        ok, msg = False, "triangle-missing"
                    else:
                        lengths1 = _triangle_side_lengths(tri1)
                        lengths2 = _triangle_side_lengths(tri2)
                        if not lengths1 or not lengths2:
                            ok, msg = False, "degenerate"
                        else:
                            diffs = [abs(a - b) for a, b in zip(lengths1, lengths2)]
                            max_diff = max(diffs)
                            ok = max_diff <= len_tol
                            msg = f"max|ΔL|={max_diff:.3e}"

            elif t == "Tangent":
                if len(sc) < 3:
                    ok, msg = False, "scope-mismatch"
                else:
                    seg_id, circle_id, tp = sc[:3]
                    params = self._circle_params(circle_id)
                    if not params:
                        ok, msg = False, "circle-missing"
                    elif tp not in self.entities:
                        ok, msg = False, "tangent-point-missing"
                    else:
                        cx, cy, radius, _ = params
                        tx, ty = self.get_point_xy(tp)
                        dist_r = abs(_dist((tx, ty), (cx, cy)) - radius)
                        if dist_r > circle_tol:
                            ok, msg = False, f"point-off-circle Δr={dist_r:.3e}"
                        else:
                            seg_pts = _segment_points(seg_id)
                            if tp not in seg_pts:
                                ok, msg = False, "point-not-on-seg"
                            else:
                                others = [pid for pid in seg_pts if pid != tp]
                                if not others:
                                    ok, msg = False, "segment-endpoints-missing"
                                else:
                                    other = max(
                                        others,
                                        key=lambda pid: _dist(self.get_point_xy(pid), (tx, ty)),
                                    )
                                    vx, vy = self.get_point_xy(other)[0] - tx, self.get_point_xy(other)[1] - ty
                                    rad_vec = (tx - cx, ty - cy)
                                    ang = _angle(rad_vec, (vx, vy))
                                    if ang is None:
                                        ok, msg = False, "degenerate"
                                    else:
                                        err = abs(90.0 - ang)
                                        ok = err <= ang_tol_deg
                                        msg = f"|90-θ|={err:.2e}°"

            elif t == "PowerEquality":
                if len(sc) != 4:
                    ok, msg = False, "scope-mismatch"
                else:
                    P_id, circle_id, secant_seg, tangent_seg = sc
                    params = self._circle_params(circle_id)
                    if not params:
                        ok, msg = False, "circle-missing"
                    elif P_id not in self.entities:
                        ok, msg = False, "point-missing"
                    else:
                        Px, Py = self.get_point_xy(P_id)
                        circle_pts = set(_points_on_circle(circle_id))
                        sec_pts = [pid for pid in _segment_points(secant_seg) if pid in circle_pts]
                        if len(sec_pts) < 2:
                            ok, msg = False, "secant-circle-points-missing"
                        else:
                            sec_pts = sorted(sec_pts, key=lambda pid: _dist((Px, Py), self.get_point_xy(pid)))
                            Uxy = self.get_point_xy(sec_pts[0])
                            Vxy = self.get_point_xy(sec_pts[1])
                            d1 = _dist((Px, Py), Uxy)
                            d2 = _dist((Px, Py), Vxy)
                            tan_pts = _segment_points(tangent_seg)
                            tan_circle_pts = [pid for pid in tan_pts if pid in circle_pts]
                            if P_id not in tan_pts or not tan_circle_pts:
                                ok, msg = False, "tangent-point-missing"
                            else:
                                tpxy = self.get_point_xy(tan_circle_pts[0])
                                tan_len = _dist((Px, Py), tpxy)
                                prod = d1 * d2
                                tan_sq = tan_len * tan_len
                                err = abs(prod - tan_sq)
                                tol_val = max(len_tol * max(prod, tan_sq, 1.0), len_tol)
                                ok = err <= tol_val
                                msg = f"|sec-prod - tan^2|={err:.3e}"

            elif t == "ExprConstraint":
                expr_meta = c.params or {}
                lhs = expr_meta.get("lhs")
                rhs = expr_meta.get("rhs")
                relation = expr_meta.get("relation") or "="
                tol = expr_meta.get("tolerance")
                try:
                    tol_val = float(tol) if tol is not None else None
                except Exception:
                    tol_val = None
                if not isinstance(lhs, str) or not isinstance(rhs, str):
                    ok, msg = False, "expr-missing"
                else:
                    try:
                        ok, msg, details = evaluate_expr_relation(self, lhs, rhs, relation, tol_val)
                        extra_info = {"expr": {**expr_meta, **details}}
                    except Exception as exc:
                        ok, msg = False, f"expr-error: {exc}"

            # Other types (Similar/Congruent/PowerEquality/Tangent/EqualAngle, etc.) are first marked unknown
            # to avoid "verbal facts" polluting the truth; you can supplement numeric verification as needed.
            else:
                ok, msg = None, "unchecked"

            ok, msg = _symbolic_rescue(ok, msg, t, sc)
            payload = {"ok": ok, "msg": msg, "type": t, "scope": sc, "source": c.source}
            if extra_info:
                payload.update(extra_info)
            report[cid] = payload
        return report


class SymbolicConstraintHelper:
    SUPPORTED = {"EqualLength", "Parallel", "Perpendicular", "Collinear", "PointOnLine", "PointOnSegment"}

    def __init__(self, VeriGraph, segment_points_fn, line_points_fn):
        self.VeriGraph = VeriGraph
        self._segment_points_fn = segment_points_fn
        self._line_points_fn = line_points_fn
        self._point_cache: Dict[str, Tuple[Fraction, Fraction]] = {}

    def _point(self, pid: str) -> Tuple[Fraction, Fraction]:
        if pid not in self._point_cache:
            x, y = self.VeriGraph.get_point_xy(pid)
            fx = Fraction(str(x)).limit_denominator(1024)
            fy = Fraction(str(y)).limit_denominator(1024)
            self._point_cache[pid] = (fx, fy)
        return self._point_cache[pid]

    def _vec(self, p: str, q: str) -> Tuple[Fraction, Fraction]:
        (px, py) = self._point(p)
        (qx, qy) = self._point(q)
        return (qx - px, qy - py)

    def _segment_endpoints(self, seg_id: str) -> Optional[Tuple[str, str]]:
        pts = self._segment_points_fn(seg_id) or []
        if len(pts) < 2:
            return None
        return (pts[0], pts[1])

    def _line_reference(self, line_id: str) -> Optional[Tuple[str, str]]:
        return self._line_points_fn(line_id)

    def verify(self, typ: str, scope: List[str]) -> Tuple[Optional[bool], str]:
        if typ not in self.SUPPORTED:
            return None, ""
        try:
            if typ == "EqualLength":
                seg1, seg2 = scope
                e1 = self._segment_endpoints(seg1)
                e2 = self._segment_endpoints(seg2)
                if not e1 or not e2:
                    return None, ""
                v1 = self._vec(e1[0], e1[1])
                v2 = self._vec(e2[0], e2[1])
                len1 = v1[0] ** 2 + v1[1] ** 2
                len2 = v2[0] ** 2 + v2[1] ** 2
                return len1 == len2, f"squared={len1} vs {len2}"
            if typ in ("Parallel", "Perpendicular"):
                seg1, seg2 = scope
                e1 = self._segment_endpoints(seg1)
                e2 = self._segment_endpoints(seg2)
                if not e1 or not e2:
                    return None, ""
                v1 = self._vec(e1[0], e1[1])
                v2 = self._vec(e2[0], e2[1])
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                dot = v1[0] * v2[0] + v1[1] * v2[1]
                if typ == "Parallel":
                    return cross == 0, f"cross={cross}"
                return dot == 0, f"dot={dot}"
            if typ == "Collinear":
                a, b, c = scope
                ax, ay = self._point(a)
                bx, by = self._point(b)
                cx, cy = self._point(c)
                area2 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
                return area2 == 0, f"area2={area2}"
            if typ == "PointOnLine":
                pid, lid = scope
                ref = self._line_reference(lid)
                if not ref:
                    return None, ""
                v = self._vec(ref[0], ref[1])
                pv = self._vec(ref[0], pid)
                cross = v[0] * pv[1] - v[1] * pv[0]
                return cross == 0, f"cross={cross}"
            if typ == "PointOnSegment":
                pid, sid = scope
                ref = self._segment_endpoints(sid)
                if not ref:
                    return None, ""
                v = self._vec(ref[0], ref[1])
                pv = self._vec(ref[0], pid)
                cross = v[0] * pv[1] - v[1] * pv[0]
                denom = v[0] * v[0] + v[1] * v[1]
                if denom == 0:
                    return None, ""
                t = (pv[0] * v[0] + pv[1] * v[1]) / denom
                on_seg = cross == 0 and 0 <= t <= 1
                return on_seg, f"cross={cross}, t={t}"
        except Exception:
            return None, ""
        return None, ""



def _draw_angle_arc(ax, A, B, C, radius=0.6, color="#000000"):
    v1 = (B[0]-A[0], B[1]-A[1]); v2 = (C[0]-A[0], C[1]-A[1])
    a1 = math.degrees(math.atan2(v1[1], v1[0])); a2 = math.degrees(math.atan2(v2[1], v2[0]))
    start = a1; sweep = (a2-a1) % 360
    if sweep>180: start, sweep = a2, (a1-a2) % 360
    arc = Arc((A[0],A[1]), width=2*radius, height=2*radius,
              angle=0, theta1=start, theta2=start+sweep, color=color, lw=1.4, zorder=4,
              capstyle="round", joinstyle="round")
    ax.add_patch(arc)


# ================================ Theorem Library ================================
class TheoremLibrary:
    # ---------- Parallel-through-point construction: through point P, draw a line parallel to known segment base_seg, returning the new segment new_seg_id ----------
    @staticmethod
    def parallel_through_point(g: "GeometryGraph", base_seg: str, P: str, new_seg_id: str) -> Dict[str, Any]:
        ends = NaturalLanguageDescriber._segment_endpoints(g)
        assert base_seg in ends and P in g.entities
        u, v = ends[base_seg]                          # base_seg = UV
        Uxy, Vxy = g.get_point_xy(u), g.get_point_xy(v)
        Pxy = g.get_point_xy(P)
        dx, dy = (Vxy[0]-Uxy[0], Vxy[1]-Uxy[1])

        # Take Q = P + (dx, dy) and add to the graph (auto-rename if Q's point name is taken)
        qid = "Q"
        k = 0
        while qid in g.entities:  # avoid name collision
            k += 1; qid = f"Q{k}"
        Qxy = (Pxy[0] + dx, Pxy[1] + dy)
        g.add_entity("Point", id=qid, attrs=Attrs.point(*Qxy, provenance="derived"))

        # New segment
        if new_seg_id not in g.entities:
            g.add_entity("Segment", id=new_seg_id)
        g.add_constraint("Incidence", [P, new_seg_id])
        g.add_constraint("Incidence", [qid, new_seg_id])
        g.add_constraint("Parallel", [new_seg_id, base_seg], source="proof")
        return {"point_Q": qid, "segment": new_seg_id}

    # ---------- Construct parallelogram ABCD from three points (given A,B,C, construct D so that AB ∥ CD and BC ∥ AD) ----------
    @staticmethod
    def parallelogram_from_ABC(g: "GeometryGraph", A: str, B: str, C: str, quad_id: str = "QUAD_ABCD") -> Dict[str, Any]:
        Axy, Bxy, Cxy = g.get_point_xy(A), g.get_point_xy(B), g.get_point_xy(C)
        # D = A + (C - B)
        Dxy = (Axy[0] + (Cxy[0]-Bxy[0]), Axy[1] + (Cxy[1]-Bxy[1]))
        if "D" not in g.entities:
            g.add_entity("Point", id="D", attrs=Attrs.point(*Dxy, provenance="derived"))

        # Edge segments
        for sid in ["SEG_AB","SEG_BC","SEG_CD","SEG_DA","SEG_AC","SEG_BD"]:
            if sid not in g.entities: g.add_entity("Segment", id=sid)
        for u, v, sid in [("A","B","SEG_AB"),("B","C","SEG_BC"),("C","D","SEG_CD"),("D","A","SEG_DA"),
                          ("A","C","SEG_AC"),("B","D","SEG_BD")]:
            g.add_constraint("Incidence",[u,sid]); g.add_constraint("Incidence",[v,sid])

        # Parallel relations
        g.add_constraint("Parallel", ["SEG_AB","SEG_CD"], source="problem")
        g.add_constraint("Parallel", ["SEG_BC","SEG_DA"], source="problem")

        # Quadrilateral entity
        if quad_id not in g.entities:
            g.add_entity("Quadrilateral", id=quad_id, attrs=Attrs.quad(["A","B","C","D"], kind="Parallelogram"))
        return {"quad": quad_id}

    # ---------- On top of the parallelogram, construct a rectangle (add one right angle) ----------
    @staticmethod
    def rectangle_from_parallelogram(g: "GeometryGraph", A="A", B="B", C="C", D="D",
                                     quad_id: str = "QUAD_RECT") -> Dict[str, Any]:
        # Add right angle (AB ⟂ BC)
        if "SEG_AB" in g.entities and "SEG_BC" in g.entities:
            g.add_constraint("Perpendicular", ["SEG_AB","SEG_BC"], source="problem")
        # Store an additional rectangle entity (annotated kind=Rectangle)
        if quad_id not in g.entities:
            g.add_entity("Quadrilateral", id=quad_id, attrs=Attrs.quad([A,B,C,D], kind="Rectangle", provenance="derived"))
        return {"quad": quad_id}

    # ---------- On top of the parallelogram, construct a rhombus (four equal sides) ----------
    @staticmethod
    def rhombus_from_parallelogram(g: "GeometryGraph", quad_id: str = "QUAD_RHOMBUS") -> Dict[str, Any]:
        # declare |AB|=|BC|=|CD|=|DA|
        for s1, s2 in [("SEG_AB","SEG_BC"),("SEG_BC","SEG_CD"),("SEG_CD","SEG_DA")]:
            g.add_constraint("EqualLength", [s1,s2], source="problem")
        if quad_id not in g.entities:
            g.add_entity("Quadrilateral", id=quad_id, attrs=Attrs.quad(["A","B","C","D"], kind="Rhombus", provenance="derived"))
        return {"quad": quad_id}

    # ---------- Similarity (AA) and congruence (SSS / SAS) relation declarations ----------
    @staticmethod
    def triangles_similar_AA(g: "GeometryGraph", tri1: str, tri2: str, angles: List[Tuple[str,str]] = None, source="problem"):
        """
        For example pass `angles=[("∠A","∠D"),("∠B","∠E")]` (names by your convention); this engine just records the Similar tag.
        """
        return {"cid": g.add_constraint("SimilarTriangles", [tri1, tri2], source=source, params={"evidence": "AA"})}

    @staticmethod
    def triangles_congruent_SSS(g: "GeometryGraph", tri1: str, tri2: str, source="problem"):
        return {"cid": g.add_constraint("CongruentTriangles", [tri1, tri2], source=source, params={"evidence": "SSS"})}

    @staticmethod
    def angle_bisector(g: GeometryGraph, seg_id: str, angle_entity: str, source="problem"):
        g.add_constraint("AngleBisector", [seg_id, angle_entity], source=source)
        name = g.entities[angle_entity].attrs.data["name"]  # ∠BAC
        X,Y,Z = name[1], name[2], name[3]
        ang1 = f"∠{X}{Y}E"; ang2 = f"∠E{Y}{Z}"
        if ang1 not in g.entities: g.add_entity("Angle", id=ang1, attrs=Attrs.angle(ang1, provenance="derived"))
        if ang2 not in g.entities: g.add_entity("Angle", id=ang2, attrs=Attrs.angle(ang2, provenance="derived"))
        g.add_constraint("EqualAngle", [ang1, ang2], source="derived")
        return {"angles":[ang1,ang2]}

    @staticmethod
    def perpendicular(g: GeometryGraph, seg1: str, seg2: str, source="problem"):
        return {"cid": g.add_constraint("Perpendicular", [seg1, seg2], source=source)}

    @staticmethod
    def parallel(g: GeometryGraph, seg1: str, seg2: str, source="problem"):
        return {"cid": g.add_constraint("Parallel", [seg1, seg2], source=source)}

    @staticmethod
    def midpoint(g: GeometryGraph, point_id: str, seg_id: str, source="problem"):
        g.add_constraint("Midpoint", [point_id, seg_id], source=source); return {"mid": point_id}

    @staticmethod
    def equal_length(g: GeometryGraph, seg1: str, seg2: str, source="problem"):
        return {"cid": g.add_constraint("EqualLength", [seg1, seg2], source=source)}

    @staticmethod
    def similar(g: GeometryGraph, tri1: str, tri2: str, source="problem"):
        return {"cid": g.add_constraint("SimilarTriangles", [tri1, tri2], source=source)}

    @staticmethod
    def concyclic(g: GeometryGraph, circle_id: str, pts: List[str], source="problem"):
        for p in pts:
            g.add_constraint("Concyclic", [p, circle_id], source=source)
        return {"circle": circle_id, "points": pts}
    # ---------- Trapezoid: given A, B, C, construct D such that AB ∥ CD, yielding trapezoid ABCD ----------
    @staticmethod
    def trapezoid_from_ABC(g: "GeometryGraph", A="A", B="B", C="C",
                           quad_id: str = "QUAD_TRAP") -> Dict[str, Any]:
        Axy, Bxy, Cxy = g.get_point_xy(A), g.get_point_xy(B), g.get_point_xy(C)
        s = _quantized_sample((0.6, 1.4))
        Dxy = (Cxy[0] + s*(Bxy[0]-Axy[0]), Cxy[1] + s*(Bxy[1]-Axy[1]))
        if "D" not in g.entities:
            g.add_entity("Point", id="D", attrs=Attrs.point(*Dxy, provenance="derived"))

        # Reorder A/B/C/D to avoid self-intersection: A/B on the left, D/C on the right (bottom/top)
        coords = {
            "A": g.get_point_xy("A"),
            "B": g.get_point_xy("B"),
            "C": g.get_point_xy("C"),
            "D": g.get_point_xy("D"),
        }
        xs = sorted(v[0] for v in coords.values())
        ys = sorted(v[1] for v in coords.values())
        left_x = xs[0]
        right_x = xs[-1]
        bottom_y = ys[0]
        top_y = ys[-1]
        if abs(right_x - left_x) < 0.8:
            right_x = left_x + 2.0
        if abs(top_y - bottom_y) < 0.8:
            top_y = bottom_y + 2.0
        target_positions = {
            "A": (left_x, bottom_y),
            "B": (left_x, top_y),
            "C": (right_x, top_y),
            "D": (right_x, bottom_y),
        }
        for pid, (tx, ty) in target_positions.items():
            prov = g.entities.get(pid, Entity(pid, "Point", Attrs())).attrs.data.get("provenance", "derived")
            g.add_entity("Point", id=pid, attrs=Attrs.point(tx, ty, provenance=prov))

        for sid in ["SEG_AB","SEG_BC","SEG_CD","SEG_DA"]:
            if sid not in g.entities: g.add_entity("Segment", id=sid)
        for u,v,sid in [("A","B","SEG_AB"),("B","C","SEG_BC"),("C","D","SEG_CD"),("D","A","SEG_DA")]:
            g.add_constraint("Incidence",[u,sid]); g.add_constraint("Incidence",[v,sid])
        g.add_constraint("Parallel", ["SEG_AB","SEG_CD"], source="problem")
        if quad_id not in g.entities:
            g.add_entity("Quadrilateral", id=quad_id, attrs=Attrs.quad(["A","B","C","D"], kind="Trapezoid"))
        return {"quad": quad_id}

    # ---------- Isosceles Trapezoid: on top of trapezoid, add AD = BC ----------
    @staticmethod
    def isosceles_trapezoid(g: "GeometryGraph", quad_id: str = "QUAD_ISO_TRAP") -> Dict[str, Any]:
        g.add_constraint("EqualLength", ["SEG_DA", "SEG_BC"], source="problem")
        if quad_id not in g.entities:
            g.add_entity("Quadrilateral", id=quad_id,
                         attrs=Attrs.quad(["A","B","C","D"], kind="IsoscelesTrapezoid", provenance="derived"))
        return {"quad": quad_id}

    # ---------- Kite: AB=AD, CB=CD ----------
    @staticmethod
    def kite_from_ABC(g: "GeometryGraph", A="A", B="B", C="C",
                      quad_id: str = "QUAD_KITE"):
        """
        Construct a kite shape: AB = AD, BC = CD, with diagonals perpendicular.
        """

        # Helper: auto-name segments to ensure unique ordering
        def _seg_id(u: str, v: str) -> str:
            return f"SEG_{u}{v}" if u < v else f"SEG_{v}{u}"

        # Construct point D
        pA = g.get_point_xy(A)
        pB = g.get_point_xy(B)
        pC = g.get_point_xy(C)
        import math
        # Place D on the extension of BC, forming the diagonal structure
        dx = pC[0] - pB[0]
        dy = pC[1] - pB[1]
        D = "D"
        g.add_entity("Point", id=D, attrs=Attrs.point(pC[0] + 1.5 * dx, pC[1] + 1.5 * dy))

        # Four sides
        seg_AB = _seg_id(A, B)
        seg_BC = _seg_id(B, C)
        seg_CD = _seg_id(C, D)
        seg_DA = _seg_id(D, A)
        for sid, (u, v) in [(seg_AB, (A, B)), (seg_BC, (B, C)),
                            (seg_CD, (C, D)), (seg_DA, (D, A))]:
            g.add_entity("Segment", id=sid)
            g.add_constraint("Incidence", [u, sid])
            g.add_constraint("Incidence", [v, sid])

        # Equal-length relations (the kite's two pairs of adjacent sides)
        g.add_constraint("EqualLength", [seg_AB, seg_DA], source="problem")
        g.add_constraint("EqualLength", [seg_BC, seg_CD], source="problem")

        # Diagonals (used for the perpendicular constraint)
        seg_AC = _seg_id(A, C)
        seg_BD = _seg_id(B, D)
        for sid, (u, v) in [(seg_AC, (A, C)), (seg_BD, (B, D))]:
            g.add_entity("Segment", id=sid)
            g.add_constraint("Incidence", [u, sid])
            g.add_constraint("Incidence", [v, sid])

        # Diagonals are perpendicular
        g.add_constraint("Perpendicular", [seg_AC, seg_BD], source="proof")

        # Record the kite element
        g.add_entity("Quadrilateral", id=quad_id)
        for s in [A, B, C, D]:
            g.add_constraint("Incidence", [s, quad_id])

    # ---------- Cyclic Quadrilateral: four points on a circle ----------
    @staticmethod
    def cyclic_quadrilateral_from_circle(
        g: "GeometryGraph",
        center_id: str,
        radius: float,
        circle_id: Optional[str] = "CIR_BASE",
        name_pts=("P","Q","R","S"),
        quad_id="QUAD_CYCLIC"
    ) -> Dict[str, Any]:
        cx,cy = g.get_point_xy(center_id)
        circle_id = circle_id or f"CIR_{center_id}"
        if circle_id not in g.entities or g.entities[circle_id].type != "Circle":
            g.add_entity("Circle", id=circle_id, attrs=Attrs.circle(center_id, radius, provenance="derived"))

        pts = []
        needed = len(name_pts)
        dirs = random.sample(_PYTHAGOREAN_DIRECTIONS, k=min(needed, len(_PYTHAGOREAN_DIRECTIONS)))
        if len(dirs) < needed:
            repeat = needed // len(dirs) + 1
            dirs = (dirs * repeat)[:needed]
        for i, nm in enumerate(name_pts):
            dx, dy = dirs[i]
            length = math.hypot(dx, dy)
            if length == 0:
                continue
            x = cx + radius * dx / length
            y = cy + radius * dy / length
            if nm in g.entities: nm = f"{nm}{random.randint(2,9)}"
            g.add_entity("Point", id=nm, attrs=Attrs.point(x,y, provenance="derived"))
            pts.append(nm)
        for nm in pts:
            g.add_constraint("Concyclic", [nm, circle_id], source="problem")
        for sid,u,v in [("SEG_PQ",pts[0],pts[1]),("SEG_QR",pts[1],pts[2]),
                        ("SEG_RS",pts[2],pts[3]),("SEG_SP",pts[3],pts[0])]:
            if sid not in g.entities: g.add_entity("Segment", id=sid)
            g.add_constraint("Incidence",[u,sid]); g.add_constraint("Incidence",[v,sid])
        if quad_id not in g.entities:
            g.add_entity("Quadrilateral", id=quad_id, attrs=Attrs.quad(pts, kind="Cyclic"))
        return {"quad": quad_id, "points": pts}

    # ---------- Tangents from External Point: two tangents from P to a circle ----------
    @staticmethod
    def circle_tangents_from_point(g: "GeometryGraph", circle_id: str, P_id: str,
                                   seg_t="SEG_PT", seg_s="SEG_PS") -> Dict[str, Any]:
        c = g.entities[circle_id].attrs.data
        O = c["center"]; r = float(c["radius"])
        Ox,Oy = g.get_point_xy(O); Px,Py = g.get_point_xy(P_id)
        vx, vy = Px-Ox, Py-Oy; d2 = vx*vx + vy*vy; d = math.sqrt(d2)
        if d <= r + 1e-9:
            scale = (r + 0.5) / max(d, 1e-6)
            Px,Py = Ox + vx*scale, Oy + vy*scale
            g.entities[P_id].attrs = Attrs.point(Px, Py, provenance="aux")
        k = (r*r)/d2
        h = (r/d2) * math.sqrt(max(d2 - r*r, 0.0))
        rx, ry = -vy, vx
        Tx = Ox + k*vx + h*rx; Ty = Oy + k*vy + h*ry
        Sx = Ox + k*vx - h*rx; Sy = Oy + k*vy - h*ry
        for tid,(x,y) in [("T",(Tx,Ty)),("S",(Sx,Sy))]:
            if tid not in g.entities:
                g.add_entity("Point", id=tid, attrs=Attrs.point(x,y, provenance="derived"))
        for sid,u,v in [(seg_t,"P","T"),(seg_s,"P","S")]:
            if sid not in g.entities: g.add_entity("Segment", id=sid)
            g.add_constraint("Incidence",[u,sid]); g.add_constraint("Incidence",[v,sid])
            g.add_constraint("Tangent", [sid, circle_id, "T" if sid==seg_t else "S"], source="problem")
        g.add_constraint("EqualLength", [seg_t, seg_s], source="proof")
        return {"tangent_points": ("T","S")}

    # ---------- Power of a Point: PA*PB = PT^2 ----------
    @staticmethod
    def power_of_point(g: "GeometryGraph", circle_id: str, P_id: str,
                       secant_seg="SEG_PU", tangent_seg="SEG_PT") -> Dict[str, Any]:
        return {"cid": g.add_constraint("PowerEquality", [P_id, circle_id, secant_seg, tangent_seg], source="proof")}
    # === Natural-derivation similar/congruent construction functions ===
    # [SIM-AA-PARALLEL]
    @staticmethod
    def build_similarity_via_parallel(g):
        """Generate AA similarity via parallel lines"""
        # △ABC already exists
        # Through C draw CE ∥ AB and pick D on AC
        # Constraints: Parallel, Incidence
        g.add_entity("Point", id="D", attrs=Attrs.point(2, 1))
        g.add_entity("Point", id="E", attrs=Attrs.point(3, 2))
        g.add_entity("Segment", id="SEG_CE")
        g.add_constraint("Incidence", ["C", "SEG_CE"])
        g.add_constraint("Incidence", ["E", "SEG_CE"])
        g.add_constraint("Parallel", ["SEG_AB", "SEG_CE"])
        g.add_constraint("SimilarTriangles", ["TRI_CDE", "TRI_CAB"],
                         source="derived", params={"reason": "AA-via-parallel"})
        return {"tri1": "TRI_CDE", "tri2": "TRI_CAB"}

    # [SIM-ANGLE-BISECTOR]
    @staticmethod
    def build_bisector_facts(g):
        """Reliable consequences of an angle bisector: equal angles + ratio"""
        # Semantics: AE is the bisector of ∠BAC
        g.add_constraint("AngleBisector", ["SEG_AE", "∠BAC"])
        # Derive equal angles
        g.add_constraint("EqualAngle", ["∠BAE", "∠EAC"], source="derived")
        # Angle bisector theorem: BE/CE = BA/CA
        # The exact constraint name depends on your system; if you don't have EqualRatio, use Proportional/SegmentRatio etc.
        g.add_constraint("EqualRatio", ["SEG_BE", "SEG_CE", "SEG_BA", "SEG_CA"],
                         source="derived", params={"reason": "Angle bisector theorem"})
        # Optional: E on BC (default in many problem setups)
        g.add_constraint("PointOnSegment", ["E", "SEG_BC"])
        return {"ratios": ["BE:CE = BA:CA"], "angles": ["∠BAE = ∠EAC"]}


# [DERIVED-RULE-ENGINE] Safe version: auto-derive AA similarity (angle bisector / parallel) + numeric fallback
class DerivedRuleEngine:
    ANG_TOL_DEG = 1.0   # Numeric angle equality threshold (degrees)

    # ---------- Small utility: safe add ----------
    @staticmethod
    def _safe_add_constraint(g, type_, scope, source="derived", params=None):
        for s in scope:
            if s not in g.entities:
                return False
        cid = g.add_constraint(type_, scope, params=params or {}, source=source)
        return True

    @staticmethod
    def _seg_id(g, u, v):
        """Return existing segment ID: SEG_uv or SEG_vu; return None if not found"""
        cand = [f"SEG_{u}{v}", f"SEG_{v}{u}"]
        for sid in cand:
            if sid in g.entities and g.entities[sid].type == "Segment":
                return sid
        return None

    @staticmethod
    def _tri_id(verts):
        """Generate canonical triangle ID from three points (does not strictly require existence)"""
        a,b,c = verts
        return f"TRI_{a}{b}{c}"

    @staticmethod
    def _ensure_triangle_entity(g, verts, provenance="derived"):
        """If the triangle entity does not exist and all three points exist, add the Triangle entity"""
        a,b,c = verts
        for p in (a,b,c):
            if p not in g.entities or g.entities[p].type != "Point":
                return None
        tid = DerivedRuleEngine._tri_id(verts)
        if tid not in g.entities:
            g.add_entity("Triangle", id=tid, attrs=Attrs.triangle([a,b,c], provenance=provenance))
        return tid

    @staticmethod
    def _angle_deg(g, X, Y, Z):
        """Numeric angle ∠XYZ (Y is vertex); return None if any point is missing"""
        try:
            Ax,Ay = g.get_point_xy(X); Bx,By = g.get_point_xy(Y); Cx,Cy = g.get_point_xy(Z)
        except Exception:
            return None
        v1 = (Ax-Bx, Ay-By); v2 = (Cx-Bx, Cy-By)
        n1 = (v1[0]**2+v1[1]**2) ** 0.5; n2 = (v2[0]**2+v2[1]**2) ** 0.5
        if n1 == 0 or n2 == 0: return None
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        import math
        cosv = max(-1.0, min(1.0, dot/(n1*n2)))
        return math.degrees(math.acos(cosv))

    @staticmethod
    def _two_angles_close(a, b, tol=1.0):
        return (a is not None) and (b is not None) and (abs(a-b) <= tol)

    @staticmethod
    def _triangle_vertices_from_entity(g, tri_id):
        e = g.entities.get(tri_id)
        if not e or e.type != "Triangle": return None
        verts = e.attrs.data.get("verts")
        if not (isinstance(verts, list) and len(verts) == 3): return None
        return tuple(verts)

    @staticmethod
    def _all_triangles(g):
        """Return list of all triangle IDs in the graph"""
        return [eid for eid, e in g.entities.items() if e.type == "Triangle"]

    @staticmethod
    def _parallel_pairs(g):
        """Return list of parallel segment pairs [(sid1,sid2), ...] (Segment only)"""
        pairs = []
        for c in g.constraints.values():
            if c.type == "Parallel":
                s1, s2 = c.scope
                if s1 in g.entities and s2 in g.entities \
                   and g.entities[s1].type == "Segment" and g.entities[s2].type == "Segment":
                    pairs.append((s1, s2))
        return pairs

    @staticmethod
    def _triangle_edge_segments(g, verts):
        """Given the three vertices of a triangle, return the three corresponding existing Segment IDs (some may be None)"""
        a,b,c = verts
        return (
            DerivedRuleEngine._seg_id(g, a, b),
            DerivedRuleEngine._seg_id(g, b, c),
            DerivedRuleEngine._seg_id(g, c, a),
        )

    @staticmethod
    def _has_parallel_between(g, segA, segB, parallel_pairs):
        if segA is None or segB is None: return False
        for x,y in parallel_pairs:
            if {x,y} == {segA, segB}: return True
        return False

    # ---------- Rule 1: angle bisector → induce AA similarity (△ABE ∼ △AEC) ----------
        # ---------- Rule 2: two pairs of parallel sides → AA similarity ----------
    @staticmethod
    def _derive_from_parallel(g):
        changed = False
        tri_ids = DerivedRuleEngine._all_triangles(g)
        if len(tri_ids) < 2: return False
        parallel_pairs = DerivedRuleEngine._parallel_pairs(g)

        # Enumerate triangle pairs
        for i in range(len(tri_ids)):
            for j in range(i+1, len(tri_ids)):
                t1, t2 = tri_ids[i], tri_ids[j]
                V1 = DerivedRuleEngine._triangle_vertices_from_entity(g, t1)
                V2 = DerivedRuleEngine._triangle_vertices_from_entity(g, t2)
                if not V1 or not V2: continue

                e1 = DerivedRuleEngine._triangle_edge_segments(g, V1)  # (sAB, sBC, sCA)
                e2 = DerivedRuleEngine._triangle_edge_segments(g, V2)

                # Count "parallel pairs": edge pair (e1[k], e2[m]) parallel
                cnt = 0
                for sA in e1:
                    for sB in e2:
                        if DerivedRuleEngine._has_parallel_between(g, sA, sB, parallel_pairs):
                            cnt += 1
                if cnt >= 2:
                    ok = DerivedRuleEngine._safe_add_constraint(
                        g, "SimilarTriangles", [t1, t2], source="derived",
                        params={"reason":"AA-via-parallel"}
                    )
                    changed = changed or ok
        return changed

    # ---------- Rule 3: numeric AA fallback (two angles approximately equal) ----------
    @staticmethod
    def _derive_numeric_AA(g):
        changed = False
        tri_ids = DerivedRuleEngine._all_triangles(g)
        if len(tri_ids) < 2: return False

        for i in range(len(tri_ids)):
            for j in range(i+1, len(tri_ids)):
                t1, t2 = tri_ids[i], tri_ids[j]
                V1 = DerivedRuleEngine._triangle_vertices_from_entity(g, t1)
                V2 = DerivedRuleEngine._triangle_vertices_from_entity(g, t2)
                if not V1 or not V2: continue
                a1,b1,c1 = V1; a2,b2,c2 = V2

                # Three vertex angles: ∠A, ∠B, ∠C
                A1 = DerivedRuleEngine._angle_deg(g, b1, a1, c1)
                B1 = DerivedRuleEngine._angle_deg(g, c1, b1, a1)
                C1 = DerivedRuleEngine._angle_deg(g, a1, c1, b1)

                A2 = DerivedRuleEngine._angle_deg(g, b2, a2, c2)
                B2 = DerivedRuleEngine._angle_deg(g, c2, b2, a2)
                C2 = DerivedRuleEngine._angle_deg(g, a2, c2, b2)

                # Any two angles approximately equal triggers AA
                eq_pairs = 0
                if DerivedRuleEngine._two_angles_close(A1, A2, DerivedRuleEngine.ANG_TOL_DEG): eq_pairs += 1
                if DerivedRuleEngine._two_angles_close(B1, B2, DerivedRuleEngine.ANG_TOL_DEG): eq_pairs += 1
                if DerivedRuleEngine._two_angles_close(C1, C2, DerivedRuleEngine.ANG_TOL_DEG): eq_pairs += 1

                if eq_pairs >= 2:
                    ok = DerivedRuleEngine._safe_add_constraint(
                        g, "SimilarTriangles", [t1, t2], source="derived",
                        params={"reason":"AA-numeric"}
                    )
                    changed = changed or ok
        return changed

    @staticmethod
    def apply(g: "GeometryGraph"):
        """
        Overall flow:
        1) Angle-bisector derives △ABE ∼ △AEC (auto-create triangles and add similarity if entities are complete)
        2) Two pairs of parallel sides derive AA similarity
        3) Numeric AA fallback (two angles approximately equal)
        Note: all additions perform existence checks and will not raise exceptions.
        """
        changed = False
        try:
            changed |= DerivedRuleEngine._derive_from_parallel(g)
        except Exception:
            pass
        try:
            changed |= DerivedRuleEngine._derive_numeric_AA(g)
        except Exception:
            pass
        return g



# ================================ NL Describer ================================
class NaturalLanguageDescriber:
    @staticmethod
    def _segment_endpoints(g: GeometryGraph) -> Dict[str, Tuple[str,str]]:
        """Infer the two endpoints of each segment via Incidence (take the two points with maximum distance)"""
        points = {eid for eid,e in g.entities.items() if e.type=="Point"}
        incid: Dict[str, List[str]] = {}
        for c in g.constraints.values():
            if c.type == "Incidence":
                p = [s for s in c.scope if s in points]
                s = [s for s in c.scope if s not in points]
                for sid in s:
                    incid.setdefault(sid, []).extend(p)
        ends = {}
        for sid, plist in incid.items():
            if g.entities.get(sid, Entity("", "", Attrs())).type != "Segment": continue
            if len(plist) < 2: continue
            unique_pts = []
            for p in plist:
                if p not in unique_pts:
                    unique_pts.append(p)
            if len(unique_pts) < 2:
                continue
            best_pair = None
            best_dist = -1.0
            cached = {}
            for idx, pid in enumerate(unique_pts):
                try:
                    cached[pid] = g.get_point_xy(pid)
                except Exception:
                    cached[pid] = None
            for i in range(len(unique_pts)):
                pi = unique_pts[i]
                xy_i = cached.get(pi)
                if xy_i is None:
                    continue
                for j in range(i + 1, len(unique_pts)):
                    pj = unique_pts[j]
                    xy_j = cached.get(pj)
                    if xy_j is None:
                        continue
                    dist = (xy_i[0] - xy_j[0]) ** 2 + (xy_i[1] - xy_j[1]) ** 2
                    if dist > best_dist:
                        best_dist = dist
                        best_pair = (pi, pj)
            if best_pair:
                ends[sid] = best_pair
        return ends

    @staticmethod
    def describe(g: "GeometryGraph", verify: Dict[str, dict] = None, only_true: bool = True) -> str:
        """
        Generate English description: by default outputs only facts that pass numeric verification (ensures consistency with the figure).
        If only_true=False, displays ✓/✗/– and the error for each constraint.
        """
        lines = []
        # 1) Point coordinates
        pts = [eid for eid, e in g.entities.items() if e.type == "Point"];
        pts.sort()
        if pts:
            lines.append("Point coordinates:")
            for p in pts:
                x, y = g.get_point_xy(p)
                fx, fy = _format_coord_xy(x, y)
                lines.append(f"  - {p}({fx}, {fy})")

        # 2) Segment lengths
        ends = NaturalLanguageDescriber._segment_endpoints(g)
        tri_edges = set(g.meta.get("triangle_edges", []))
        segs = sorted(ends.keys())
        if segs:
            lines.append("Segment lengths:")
            for sid in segs:
                u, v = ends[sid]
                ux, uy = g.get_point_xy(u)
                vx, vy = g.get_point_xy(v)
                sq = (ux - vx) ** 2 + (uy - vy) ** 2
                L = _format_sqrt_from_square(sq)
                tag = "(triangle edge)" if sid in tri_edges else ""
                entry = f"  - |{sid}| = {L}"
                if tag:
                    entry += f" {tag}"
                lines.append(entry)

        # 3) Circle parameters
        circles = [eid for eid, e in g.entities.items() if e.type == "Circle"]
        if circles:
            lines.append("Circle parameters:")
            for cid in sorted(circles):
                c = g.entities[cid].attrs.data
                center = c["center"];
                r = float(c["radius"])
                cx, cy = g.get_point_xy(center)
                fx, fy = _format_coord_xy(cx, cy)
                lines.append(f"  - {cid}: center {center}({fx},{fy}), radius r={_format_fraction(r)}")

        # 4) Relations: only true facts (or with status)
        if g.constraints:
            lines.append("Geometric relations:")
            ver = verify or g.verify_constraints()
            for cid, c in g.constraints.items():
                t, sc = c.type, c.scope
                st = ver.get(cid, {"ok": None, "msg": "", "type": t})
                ok = st["ok"]
                mark = "✓" if ok is True else ("✗" if ok is False else "–")
                # Only output true facts
                if only_true and ok is not True:
                    continue
                src = c.source
                if t == "AngleBisector":
                    seg, ang = sc
                    name = g.entities[ang].attrs.data.get("name", ang)
                    lines.append(
                        f"  - {seg} is the angle bisector of {name} (source: {src})" + ("" if only_true else f" [{mark} {st['msg']}]"))
                elif t == "EqualAngle":
                    a1, a2 = sc
                    lines.append(f"  - {a1} = {a2} (source: {src})" + ("" if only_true else f" [{mark}]"))
                elif t == "EqualLength":
                    s1, s2 = sc
                    lines.append(f"  - {s1} = {s2} (source: {src})" + ("" if only_true else f" [{mark} {st['msg']}]"))
                elif t == "Midpoint":
                    m, seg = sc
                    lines.append(
                        f"  - {m} is the midpoint of {seg} (source: {src})" + ("" if only_true else f" [{mark} {st['msg']}]"))
                elif t == "Perpendicular":
                    s1, s2 = sc
                    lines.append(f"  - {s1} ⟂ {s2} (source: {src})" + ("" if only_true else f" [{mark} {st['msg']}]"))
                elif t == "Parallel":
                    s1, s2 = sc
                    lines.append(f"  - {s1} ∥ {s2} (source: {src})" + ("" if only_true else f" [{mark} {st['msg']}]"))
                elif t == "OnCircle":
                    if len(sc) == 2:
                        p, cir = sc
                        lines.append(
                            f"  - point {p} on circle {cir} (source: {src})" + ("" if only_true else f" [{mark} {st['msg']}]"))
                    else:
                        lines.append(
                            f"  - points {', '.join(map(str, sc))} on the same circle (source: {src})" + ("" if only_true else f" [{mark} {st['msg']}]"))
                elif t == "Concyclic":
                    if len(sc) >= 3:
                        lines.append(
                            f"  - points {', '.join(map(str, sc))} concyclic (source: {src})" + ("" if only_true else f" [{mark} {st['msg']}]"))
                    elif len(sc) == 2:
                        p, cir = sc
                        lines.append(
                            f"  - point {p} on circle {cir} (source: {src})" + ("" if only_true else f" [{mark} {st['msg']}]"))
                    else:
                        lines.append(
                            f"  - concyclic relation (source: {src})" + ("" if only_true else f" [{mark} {st['msg']}]"))
                elif t == "SimilarTriangles":
                    t1, t2 = sc
                    lines.append(f"  - triangle {t1} ∼ {t2} (source: {src})" + ("" if only_true else " [–]"))
                elif t == "CongruentTriangles":
                    t1, t2 = sc
                    lines.append(f"  - triangle {t1} ≅ {t2} (source: {src})" + ("" if only_true else " [–]"))
                elif t == "RightAngle":
                    s1, s2 = sc
                    lines.append(f"  - {s1} and {s2} form a right angle (source: {src})" + ("" if only_true else f" [{mark}]"))
                elif t == "Tangent":
                    seg, cir, tp = sc
                    lines.append(
                        f"  - segment {seg} is tangent to circle {cir} at point {tp} (source: {src})" + ("" if only_true else " [–]"))
                elif t == "PowerEquality":
                    P, cir, sec, tan = sc
                    lines.append(f"  - Power of a point: power of point {P} with respect to circle {cir} satisfies secant product |{sec}| = |{tan}|^2 (source: {src})" + (
                        "" if only_true else " [–]"))
                elif t == "VectorSum":
                    A, B, C, D, E, F = sc
                    lines.append(
                        f"  - vector →{A}{B} = →{C}{D} + →{E}{F} (source: {src})" + (
                            "" if only_true else f" [{mark} {st['msg']}]"))
                elif t == "VectorDiff":
                    A, B, C, D, E, F = sc
                    lines.append(
                        f"  - vector →{A}{B} = →{C}{D} - →{E}{F} (source: {src})" + (
                            "" if only_true else f" [{mark} {st['msg']}]"))
                elif t == "VectorDotZero":
                    A, B, C, D = sc
                    lines.append(
                        f"  - vector →{A}{B} · →{C}{D} = 0 (source: {src})" + (
                            "" if only_true else f" [{mark} {st['msg']}]"))
                elif t == "VectorLinear":
                    if len(sc) >= 5 and (len(sc) - 2) % 3 == 0:
                        target = f"→{sc[0]}{sc[1]}"
                        parts = []
                        for idx in range(2, len(sc), 3):
                            k = sc[idx]
                            parts.append(f"{k}·→{sc[idx + 1]}{sc[idx + 2]}")
                        rhs = " + ".join(parts) if parts else "0"
                        lines.append(
                            f"  - vector {target} = {rhs} (source: {src})" + (
                                "" if only_true else f" [{mark} {st['msg']}]"))
                    else:
                        lines.append(f"  - vector linear relation (source: {src})" + (
                            "" if only_true else f" [{mark} {st['msg']}]"))

        return "\n".join(lines)


# ================================ Demo ================================


def _save_json(path: str, payload: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def _build_qa_payload(
    author_output: Optional[Dict[str, Any]],
    solver_output: Optional[Dict[str, Any]],
    problem_fallback: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    problem = None
    if isinstance(author_output, dict):
        problem = author_output.get("problem")
    if isinstance(problem, dict):
        problem = problem.get("problem")
    if problem is None and problem_fallback is not None:
        problem = problem_fallback.get("problem") if isinstance(problem_fallback, dict) else problem_fallback
        if isinstance(problem, dict):
            problem = problem.get("problem")
    proof = None
    if isinstance(solver_output, dict):
        proof = solver_output.get("proof")
    return {"problem": problem, "proof": proof}


def _write_qa_json(
    job_dir: str,
    author_output: Optional[Dict[str, Any]],
    solver_output: Optional[Dict[str, Any]],
    problem_fallback: Optional[Dict[str, Any]] = None,
):
    qa_payload = _build_qa_payload(author_output, solver_output, problem_fallback=problem_fallback)
    _save_json(os.path.join(job_dir, "QA.json"), qa_payload)


_GENERATION_MEMORY_FILENAME = "generation_memory.json"
_GENERATION_LOCK_FILENAME = ".generation.lock"
_GENERATION_MEMORY_LOCK = threading.Lock()


def _now_timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _sanitize_folder_component(value: str) -> str:
    cleaned = (value or "").strip()
    if not cleaned:
        return "unspecified"
    invalid = '<>:"/\\|?*'
    return "".join("_" if ch in invalid else ch for ch in cleaned)


def _resolve_output_root(difficulty: str, family: Optional[str]) -> str:
    configured = (CONFIG.output_root or "").strip()
    if configured:
        return configured
    fam = _sanitize_folder_component(family or "unspecified")
    diff = _sanitize_folder_component(difficulty or CONFIG.difficulty or "hard")
    return os.path.join(os.getcwd(), f"{fam}_{diff}")


def _generation_memory_path(out_root: str) -> str:
    return os.path.join(out_root, _GENERATION_MEMORY_FILENAME)


def _scan_existing_problem_indices(out_root: str) -> List[int]:
    try:
        entries = os.listdir(out_root)
    except FileNotFoundError:
        return []
    indices: List[int] = []
    for name in entries:
        match = re.match(r"^problem_(\d+)", name, flags=re.IGNORECASE)
        if not match:
            continue
        try:
            idx = int(match.group(1))
        except Exception:
            continue
        if os.path.isdir(os.path.join(out_root, name)):
            indices.append(idx)
    return indices


def _scan_error_attempt_dirs(out_root: str) -> int:
    error_root = os.path.join(out_root, "error")
    try:
        entries = os.listdir(error_root)
    except FileNotFoundError:
        return 0
    count = 0
    for name in entries:
        if not re.match(r"^problem_\d+", name, flags=re.IGNORECASE):
            continue
        if os.path.isdir(os.path.join(error_root, name)):
            count += 1
    return count


def _default_generation_memory(family: str, difficulty: str) -> Dict[str, Any]:
    return {
        "schema_version": 2,
        "family": family,
        "difficulty": difficulty,
        "total_generation_attempts": 0,
        "directly_certified_no_repair": 0,
        "certified_after_repair": 0,
        "rejected_after_verification": 0,
        "veriscript_intercepted_count": 0,
        "verisolve_intercepted_count": 0,
        "numeric_reflection_count": 0,
        "logic_reflection_count": 0,
        "next_problem_index": 0,
        "available_indices": [],
        "updated_at": _now_timestamp(),
    }


_LEGACY_GENERATION_KEYS = (
    "generated_success_count",
    "logic_reflection_rounds",
    "numeric_reflection_rounds",
    "repaired_success_count",
    "failed_final_count",
    "VeriScript_failure_count",
    "VeriSolve_failure_count",
    "author_reflection_rounds",
    "solver_numeric_reflection_rounds",
    "solver_logic_reflection_rounds",
)


def _migrate_generation_memory(memory: Dict[str, Any], family: str, difficulty: str) -> Dict[str, Any]:
    if not isinstance(memory, dict):
        return _default_generation_memory(family, difficulty)
    memory.setdefault("schema_version", 2)
    memory.setdefault("family", family)
    memory.setdefault("difficulty", difficulty)

    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            return default

    legacy_generated = _safe_int(memory.get("generated_success_count", 0))
    legacy_repaired = _safe_int(memory.get("repaired_success_count", 0))
    legacy_failed = _safe_int(memory.get("failed_final_count", 0))
    legacy_veriscript = _safe_int(memory.get("VeriScript_failure_count", 0))
    legacy_verisolve = _safe_int(memory.get("VeriSolve_failure_count", 0))
    legacy_numeric_rounds = _safe_int(memory.get("numeric_reflection_rounds", 0))
    legacy_logic_rounds = _safe_int(memory.get("logic_reflection_rounds", 0))
    legacy_author_rounds = _safe_int(memory.get("author_reflection_rounds", 0))
    legacy_solver_logic = _safe_int(memory.get("solver_logic_reflection_rounds", 0))
    legacy_solver_numeric = _safe_int(memory.get("solver_numeric_reflection_rounds", 0))

    if "total_generation_attempts" not in memory:
        memory["total_generation_attempts"] = legacy_generated + legacy_failed
    if "certified_after_repair" not in memory:
        memory["certified_after_repair"] = legacy_repaired
    if "directly_certified_no_repair" not in memory:
        direct = legacy_generated - legacy_repaired
        memory["directly_certified_no_repair"] = max(0, direct)
    if "rejected_after_verification" not in memory:
        memory["rejected_after_verification"] = legacy_failed
    if "veriscript_intercepted_count" not in memory:
        memory["veriscript_intercepted_count"] = legacy_veriscript
    if "verisolve_intercepted_count" not in memory:
        memory["verisolve_intercepted_count"] = legacy_verisolve
    if "numeric_reflection_count" not in memory:
        memory["numeric_reflection_count"] = legacy_numeric_rounds or legacy_solver_numeric
    if "logic_reflection_count" not in memory:
        if legacy_logic_rounds:
            memory["logic_reflection_count"] = legacy_logic_rounds
        else:
            memory["logic_reflection_count"] = legacy_author_rounds + legacy_solver_logic

    for key in _LEGACY_GENERATION_KEYS:
        memory.pop(key, None)

    memory["schema_version"] = max(int(memory.get("schema_version", 2) or 2), 2)
    return memory


def _load_generation_memory(out_root: str, family: str, difficulty: str) -> Dict[str, Any]:
    mem_path = _generation_memory_path(out_root)
    memory: Dict[str, Any]
    if os.path.exists(mem_path):
        try:
            with open(mem_path, "r", encoding="utf-8") as fh:
                memory = json.load(fh)
        except Exception:
            memory = _default_generation_memory(family, difficulty)
    else:
        memory = _default_generation_memory(family, difficulty)

    if not isinstance(memory, dict):
        memory = _default_generation_memory(family, difficulty)

    memory = _migrate_generation_memory(memory, family, difficulty)
    for key in (
        "total_generation_attempts",
        "directly_certified_no_repair",
        "certified_after_repair",
        "rejected_after_verification",
        "veriscript_intercepted_count",
        "verisolve_intercepted_count",
        "numeric_reflection_count",
        "logic_reflection_count",
        "next_problem_index",
    ):
        try:
            memory[key] = int(memory.get(key, 0))
        except Exception:
            memory[key] = 0

    available_raw = memory.get("available_indices")
    if not isinstance(available_raw, list):
        available_raw = []
    available_clean: List[int] = []
    for item in available_raw:
        try:
            val = int(item)
        except Exception:
            continue
        if val > 0:
            available_clean.append(val)
    memory["available_indices"] = sorted(set(available_clean))

    existing_indices = _scan_existing_problem_indices(out_root)
    if existing_indices:
        existing_count = len(existing_indices)
        max_existing = max(existing_indices)
        existing_set = set(existing_indices)
        missing = [idx for idx in range(1, max_existing + 1) if idx not in existing_set]
        if missing:
            memory["available_indices"] = sorted(set(memory["available_indices"]) | set(missing))
        if memory["next_problem_index"] < max_existing:
            memory["next_problem_index"] = max_existing
        if memory["available_indices"]:
            memory["available_indices"] = [idx for idx in memory["available_indices"] if idx not in existing_indices]

    return memory


def _write_generation_memory(out_root: str, payload: Dict[str, Any]):
    mem_path = _generation_memory_path(out_root)
    tmp_path = mem_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    os.replace(tmp_path, mem_path)


def _acquire_generation_lock(lock_path: str, timeout_secs: float = 10.0, stale_after_secs: float = 600.0):
    start = time.time()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as fh:
                fh.write(str(time.time()))
            return
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(lock_path) > stale_after_secs:
                    os.remove(lock_path)
                    continue
            except Exception:
                pass
            if time.time() - start > timeout_secs:
                raise TimeoutError(f"Timeout waiting for memory lock: {lock_path}")
            time.sleep(0.05)


def _with_generation_memory_lock(out_root: str, fn):
    os.makedirs(out_root, exist_ok=True)
    lock_path = os.path.join(out_root, _GENERATION_LOCK_FILENAME)
    with _GENERATION_MEMORY_LOCK:
        _acquire_generation_lock(lock_path)
        try:
            return fn()
        finally:
            try:
                os.remove(lock_path)
            except Exception:
                pass


def _reserve_problem_index(out_root: str, family: str, difficulty: str) -> int:
    def _reserve():
        memory = _load_generation_memory(out_root, family, difficulty)
        available = list(memory.get("available_indices") or [])
        existing = set(_scan_existing_problem_indices(out_root))
        assigned = None
        available.sort()
        while available:
            candidate = available.pop(0)
            if candidate <= 0 or candidate in existing:
                continue
            assigned = candidate
            break
        if assigned is None:
            next_index = max(int(memory.get("next_problem_index", 0)), 0)
            assigned = next_index + 1
            while assigned in existing:
                assigned += 1
            memory["next_problem_index"] = assigned
        memory["available_indices"] = available
        memory["updated_at"] = _now_timestamp()
        _write_generation_memory(out_root, memory)
        return assigned
    return _with_generation_memory_lock(out_root, _reserve)


def _note_generation_attempt_start(out_root: str, family: str, difficulty: str):
    def _update():
        memory = _load_generation_memory(out_root, family, difficulty)
        memory["total_generation_attempts"] = int(memory.get("total_generation_attempts", 0)) + 1
        memory["updated_at"] = _now_timestamp()
        _write_generation_memory(out_root, memory)
    _with_generation_memory_lock(out_root, _update)


def _summarize_author_stats(stage_outputs: Optional[Dict[str, Any]]) -> Tuple[int, int]:
    if not isinstance(stage_outputs, dict):
        return 0, 0
    stats = stage_outputs.get("author_stats") or {}
    try:
        reflection_rounds = int(stats.get("reflection_rounds", 0))
    except Exception:
        reflection_rounds = 0
    try:
        failed_attempts = int(stats.get("failed_attempts", 0))
    except Exception:
        failed_attempts = 0
    return reflection_rounds, failed_attempts


def _summarize_solver_attempts(solver_output: Optional[Dict[str, Any]]) -> Tuple[int, int, bool]:
    if not isinstance(solver_output, dict):
        return 0, 0, False
    records = solver_output.get("_attempt_records") or []
    numeric_reflection_rounds = 0
    ok_count = 0
    had_failure = False
    for record in records:
        if not isinstance(record, dict):
            continue
        status = record.get("status")
        if status == "numeric_fail":
            numeric_reflection_rounds += 1
            had_failure = True
        elif status == "ok":
            ok_count += 1
        else:
            if status:
                had_failure = True
    logic_reflection_rounds = max(0, ok_count - 1)
    return numeric_reflection_rounds, logic_reflection_rounds, had_failure


def _scan_logs_for_failures(logs: Optional[List[Dict[str, Any]]]) -> Tuple[bool, bool]:
    VeriScript_failed = False
    VeriSolve_failed = False
    if not logs:
        return VeriScript_failed, VeriSolve_failed
    for entry in logs:
        if not isinstance(entry, dict):
            continue
        op = entry.get("op") or ""
        ok_flag = entry.get("ok")
        geom_ok = entry.get("geom_ok")
        if op == "VeriSolve" and ok_flag is False:
            VeriSolve_failed = True
        if ok_flag is False or geom_ok is False:
            VeriScript_failed = True
    return VeriScript_failed, VeriSolve_failed


def _detect_VeriScript_and_VeriSolve_failures(
    stage_outputs: Optional[Dict[str, Any]],
    solver_output: Optional[Dict[str, Any]],
    error_text: Optional[str] = None,
) -> Tuple[bool, bool]:
    VeriScript_failed = False
    VeriSolve_failed = False

    if isinstance(stage_outputs, dict):
        author_combo = (stage_outputs.get("author") or {}).get("combo") or {}
        combo_logs = author_combo.get("logs") if isinstance(author_combo, dict) else None
        combo_VeriScript, combo_solve = _scan_logs_for_failures(combo_logs)
        VeriScript_failed |= combo_VeriScript
        VeriSolve_failed |= combo_solve

        author_stats = stage_outputs.get("author_stats") or {}
        failed_steps = author_stats.get("failed_steps") or []
        for step in failed_steps:
            if not isinstance(step, dict):
                continue
            step_logs = step.get("logs")
            step_VeriScript, step_solve = _scan_logs_for_failures(step_logs)
            VeriScript_failed |= step_VeriScript
            VeriSolve_failed |= step_solve

    if isinstance(solver_output, dict):
        exec_logs = solver_output.get("execution_logs")
        exec_VeriScript, exec_solve = _scan_logs_for_failures(exec_logs)
        VeriScript_failed |= exec_VeriScript
        VeriSolve_failed |= exec_solve

        attempt_records = solver_output.get("_attempt_records") or []
        for record in attempt_records:
            if not isinstance(record, dict):
                continue
            status = record.get("status")
            if status == "numeric_fail":
                VeriScript_failed = True
            record_logs = record.get("logs")
            rec_VeriScript, rec_solve = _scan_logs_for_failures(record_logs)
            VeriScript_failed |= rec_VeriScript
            VeriSolve_failed |= rec_solve

    if error_text:
        if not VeriSolve_failed and ("VeriSolve" in error_text or "VeriSolve" in error_text):
            VeriSolve_failed = True
        if not VeriScript_failed and (
            "VeriScript" in error_text
            or "Numeric / intersection" in error_text
            or "Numeric / intersection issues still present after Author self-check" in error_text
            or "VeriScript generation failed" in error_text
        ):
            VeriScript_failed = True

    return VeriScript_failed, VeriSolve_failed


def _update_generation_memory_success(
    out_root: str,
    family: str,
    difficulty: str,
    *,
    directly_certified_no_repair: bool,
    certified_after_repair: bool,
    veriscript_intercepted: bool,
    verisolve_intercepted: bool,
    numeric_reflection: bool,
    logic_reflection: bool,
    problem_index: int,
):
    def _update():
        memory = _load_generation_memory(out_root, family, difficulty)
        available = memory.get("available_indices") or []
        if problem_index in available:
            memory["available_indices"] = [idx for idx in available if idx != problem_index]
        if directly_certified_no_repair:
            memory["directly_certified_no_repair"] = int(memory.get("directly_certified_no_repair", 0)) + 1
        if certified_after_repair:
            memory["certified_after_repair"] = int(memory.get("certified_after_repair", 0)) + 1
        if veriscript_intercepted:
            memory["veriscript_intercepted_count"] = int(memory.get("veriscript_intercepted_count", 0)) + 1
        if verisolve_intercepted:
            memory["verisolve_intercepted_count"] = int(memory.get("verisolve_intercepted_count", 0)) + 1
        if certified_after_repair and numeric_reflection:
            memory["numeric_reflection_count"] = int(memory.get("numeric_reflection_count", 0)) + 1
        if certified_after_repair and logic_reflection:
            memory["logic_reflection_count"] = int(memory.get("logic_reflection_count", 0)) + 1
        memory["updated_at"] = _now_timestamp()
        _write_generation_memory(out_root, memory)
    _with_generation_memory_lock(out_root, _update)


def _update_generation_memory_failure(
    out_root: str,
    family: str,
    difficulty: str,
    *,
    veriscript_intercepted: bool,
    verisolve_intercepted: bool,
    numeric_reflection: bool,
    logic_reflection: bool,
    problem_index: int,
):
    def _update():
        memory = _load_generation_memory(out_root, family, difficulty)
        memory["rejected_after_verification"] = int(memory.get("rejected_after_verification", 0)) + 1
        if problem_index > 0:
            available = memory.get("available_indices") or []
            if problem_index not in available:
                available.append(problem_index)
            memory["available_indices"] = sorted(set(int(x) for x in available if int(x) > 0))
        if veriscript_intercepted:
            memory["veriscript_intercepted_count"] = int(memory.get("veriscript_intercepted_count", 0)) + 1
        if verisolve_intercepted:
            memory["verisolve_intercepted_count"] = int(memory.get("verisolve_intercepted_count", 0)) + 1
        memory["updated_at"] = _now_timestamp()
        _write_generation_memory(out_root, memory)
    _with_generation_memory_lock(out_root, _update)


def _move_failed_job_dir(out_root: str, job_dir: str, problem_index: int, seed: Optional[int] = None) -> str:
    error_root = os.path.join(out_root, "error")
    os.makedirs(error_root, exist_ok=True)
    base_name = f"problem_{problem_index}"
    target = os.path.join(error_root, base_name)
    if os.path.exists(target):
        suffix_parts = [time.strftime("%Y%m%d_%H%M%S", time.localtime())]
        if seed is not None:
            suffix_parts.insert(0, f"seed_{seed}")
        suffix = "_".join(suffix_parts)
        target = os.path.join(error_root, f"{base_name}_{suffix}")
    shutil.move(job_dir, target)
    return target


def _run_single_problem_job(
    job_index: int,
    seed: int,
    out_root: str,
    difficulty: str,
    *,
    enable_show: bool = False,
    family_override: Optional[str] = None,
) -> str:
    raw_family = family_override if family_override is not None else CONFIG.blueprint_family
    family = _sanitize_folder_component(raw_family or "unspecified")
    problem_index = _reserve_problem_index(out_root, family, difficulty)
    job_dir = os.path.join(out_root, f"problem_{problem_index}")
    os.makedirs(job_dir, exist_ok=True)
    stage_outputs: Dict[str, Any] = {}
    solver_output: Optional[Dict[str, Any]] = None
    start_wall = time.time()
    start_ts = _now_timestamp_ms()
    _reset_llm_metrics()
    print(f"[JOB][START] {start_ts} slot={job_index} problem={problem_index} seed={seed} {_worker_label()} out={job_dir}")
    prev_override = _get_blueprint_family_override()
    try:
        _set_blueprint_family_override(raw_family)
        _note_generation_attempt_start(out_root, family, difficulty)
        g = GeometryGraph()
        save_original = not AUTHOR_FROM_SCRATCH
        if save_original:
            original_img = os.path.join(job_dir, "graph_original.png")
            g.render(save_path=original_img, show=False)

        author_img = os.path.join(job_dir, "graph_author.png")
        problem_json = generate_problem_from_VeriGraph(
            g,
            strict_validate=False,
            allow_author_edits=True,
            preview_path=author_img,
            stage_outputs=stage_outputs,
        )
        g.render(save_path=author_img, show=False)

        _save_json(os.path.join(job_dir, "problem.json"), problem_json)
        if stage_outputs:
            _save_json(os.path.join(job_dir, "author_output.json"), stage_outputs)
        else:
            _save_json(os.path.join(job_dir, "author_output.json"), {"note": "author stage skipped"})

        solver_output = solve_problem_with_llm(g, problem_json)
        actions = [VeriScriptAction(**a) for a in solver_output.get("actions", [])]
        solver_logs = _apply_VeriScript_on_VeriGraph(g, actions, verify_only=True)
        if solver_logs:
            solver_output = dict(solver_output)
            solver_output["execution_logs"] = solver_logs
        _save_json(os.path.join(job_dir, "solver_output.json"), solver_output)
        _write_qa_json(job_dir, stage_outputs, solver_output, problem_fallback=problem_json)

        solver_img = os.path.join(job_dir, "graph_solver.png")
        g.render(save_path=solver_img, show=False)
        if enable_show:
            switched = False
            try:
                if GUI_MPL_BACKEND and plt.get_backend().lower() != GUI_MPL_BACKEND.lower():
                    plt.switch_backend(GUI_MPL_BACKEND)
                    switched = True
            except Exception as backend_err:
                print(f"[WARN] Backend switch before figure display failed: {backend_err}")
            try:
                g.render(show=True)
            except Exception as show_err:
                print(f"[WARN] Figure display failed: {show_err}")
            finally:
                if switched:
                    try:
                        plt.switch_backend(HEADLESS_ACTIVE_BACKEND)
                    except Exception as backend_err:
                        print(f"[WARN] Restoring render backend failed: {backend_err}")

        if CONFIG.enable_teacher_agent:
            ver2 = g.verify_constraints()
            nl2 = NaturalLanguageDescriber.describe(g, verify=ver2, only_true=False)
            teacher_text = generate_solver_proof_narration(problem_json, solver_output, nl2)
            with open(os.path.join(job_dir, "teacher_output.txt"), "w", encoding="utf-8") as fh:
                fh.write(teacher_text or "")
        else:
            print("[TEACHER] Skipping teacher agent (disabled in configuration)")

        author_reflection_rounds, author_failed_attempts = _summarize_author_stats(stage_outputs)
        solver_numeric_reflection_rounds, solver_logic_reflection_rounds, solver_had_failure = _summarize_solver_attempts(solver_output)
        VeriScript_failed, VeriSolve_failed = _detect_VeriScript_and_VeriSolve_failures(stage_outputs, solver_output)
        author_modified = bool((problem_json.get("meta") or {}).get("reflection_modified")) if isinstance(problem_json, dict) else False
        solver_modified = bool((solver_output or {}).get("reflection_modified")) if isinstance(solver_output, dict) else False
        has_intercept = VeriScript_failed or VeriSolve_failed
        numeric_reflection = has_intercept
        logic_reflection = (not has_intercept) and (author_modified or solver_modified)
        directly_certified = (not has_intercept) and (not author_modified) and (not solver_modified)
        certified_after_repair = not directly_certified
        _update_generation_memory_success(
            out_root,
            family,
            difficulty,
            directly_certified_no_repair=directly_certified,
            certified_after_repair=certified_after_repair,
            veriscript_intercepted=VeriScript_failed,
            verisolve_intercepted=VeriSolve_failed,
            numeric_reflection=numeric_reflection,
            logic_reflection=logic_reflection,
            problem_index=problem_index,
        )

        end_ts = _now_timestamp_ms()
        duration = time.time() - start_wall
        _log_llm_metrics_summary(f"slot={job_index} problem={problem_index} seed={seed}")
        print(f"[JOB][END] {end_ts} slot={job_index} problem={problem_index} seed={seed} duration={duration:.2f}s {_worker_label()}")
        return job_dir
    except Exception:
        error_text = traceback.format_exc()
        VeriScript_failed, VeriSolve_failed = _detect_VeriScript_and_VeriSolve_failures(stage_outputs, solver_output, error_text=error_text)
        author_reflection_rounds, _ = _summarize_author_stats(stage_outputs)
        solver_numeric_reflection_rounds, solver_logic_reflection_rounds, _ = _summarize_solver_attempts(solver_output)
        solver_modified = bool((solver_output or {}).get("reflection_modified")) if isinstance(solver_output, dict) else False
        has_intercept = VeriScript_failed or VeriSolve_failed
        numeric_reflection = has_intercept
        logic_reflection = (not has_intercept) and solver_modified
        _update_generation_memory_failure(
            out_root,
            family,
            difficulty,
            veriscript_intercepted=VeriScript_failed,
            verisolve_intercepted=VeriSolve_failed,
            numeric_reflection=numeric_reflection,
            logic_reflection=logic_reflection,
            problem_index=problem_index,
        )
        err_path = os.path.join(job_dir, "error.txt")
        with open(err_path, "w", encoding="utf-8") as fh:
            fh.write(error_text)
        try:
            _move_failed_job_dir(out_root, job_dir, problem_index, seed=seed)
        except Exception as move_err:
            print(f"[WARN] Error moving failure directory to error: {move_err}")
        end_ts = _now_timestamp_ms()
        duration = time.time() - start_wall
        _log_llm_metrics_summary(f"slot={job_index} problem={problem_index} seed={seed} status=error")
        print(f"[JOB][END] {end_ts} slot={job_index} problem={problem_index} seed={seed} duration={duration:.2f}s status=error {_worker_label()}")
        raise
    finally:
        _set_blueprint_family_override(prev_override)


def _print_batch_summary(
    success_records: List[Tuple[Any, ...]],
    failure_records: List[Tuple[Any, ...]],
    retry_attempts: Optional[List[Tuple[Any, ...]]] = None,
):
    print("[BATCH] All tasks processed.")
    if success_records:
        print("[BATCH] Successful problems:")
        for rec in success_records:
            if len(rec) >= 5:
                idx, seed, fam, diff, folder = rec[:5]
                print(f"  - #{idx} ({fam}/{diff}, seed={seed}) -> {folder}")
            else:
                idx, seed, folder = rec[:3]
                print(f"  - #{idx} (seed={seed}) -> {folder}")
    if failure_records:
        print("[BATCH] Failed problems:")
        for rec in failure_records:
            if len(rec) >= 5:
                idx, seed, fam, diff, err = rec[:5]
                print(f"  - #{idx} ({fam}/{diff}, seed={seed}) -> {err}")
            else:
                idx, seed, err = rec[:3]
                print(f"  - #{idx} (seed={seed}) -> {err}")
    if retry_attempts:
        print("[BATCH] Auto-replaced failure attempts:")
        for rec in retry_attempts:
            if len(rec) >= 5:
                idx, seed, fam, diff, err = rec[:5]
                print(f"  - #{idx} ({fam}/{diff}, seed={seed}) -> {err}")
            else:
                idx, seed, err = rec[:3]
                print(f"  - #{idx} (seed={seed}) -> {err}")
    if not failure_records:
        print("[BATCH] All problems generated successfully.")


def run_parallel_author_solver_teacher(
    num_problems: Optional[int] = None,
    base_seed: int = 5656,
    difficulty: Optional[str] = None,
    family: Optional[str] = None,
    show_when_single: Optional[bool] = None,
):
    cfg_parallel = max(1, int(CONFIG.batch_parallel_jobs or 1))
    show_single = show_when_single if show_when_single is not None else bool(CONFIG.show_single_job_plot)
    if num_problems is None:
        num_problems = cfg_parallel
    num_problems = max(1, int(num_problems))
    if num_problems <= 0:
        print("[BATCH] No problem count specified; skipping run")
        return
    target_difficulty = difficulty or CONFIG.difficulty or "hard"
    target_family = family if family is not None else CONFIG.blueprint_family
    out_root = _resolve_output_root(target_difficulty, target_family)
    os.makedirs(out_root, exist_ok=True)
    max_workers = min(num_problems, cfg_parallel)
    total = num_problems
    completed_success = 0
    success_records: List[Tuple[int, int, str]] = []
    failure_records: List[Tuple[int, int, str]] = []
    retry_attempts: List[Tuple[int, int, str]] = []
    next_seed = base_seed

    def _print_progress():
        if total <= 0:
            return
        bar_len = 30
        ratio = completed_success / total
        filled = min(bar_len, int(bar_len * ratio))
        bar = "#" * filled + "-" * (bar_len - filled)
        print(f"[BATCH][PROGRESS] [{bar}] {completed_success}/{total}")

    if max_workers <= 1:
        while completed_success < total:
            idx = completed_success
            seed = next_seed
            next_seed += 1
            try:
                job_dir = _run_single_problem_job(
                    idx, seed, out_root, target_difficulty,
                    enable_show=(show_single and num_problems == 1),
                    family_override=target_family,
                )
                success_records.append((idx + 1, seed, job_dir))
                completed_success += 1
                print(f"[BATCH] Problem {idx + 1} (seed={seed}) completed; output dir: {job_dir}")
            except Exception as exc:
                retry_attempts.append((idx + 1, seed, str(exc)))
                print(f"[BATCH][ERROR] Problem {idx + 1} (seed={seed}) failed: {exc}; will try regenerating with a new seed")
            _print_progress()
        _print_batch_summary(success_records, failure_records, retry_attempts=retry_attempts)
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: Dict[Any, Tuple[int, int]] = {}
        pending_slots = list(range(num_problems))
        completed_slots = set()
        inflight_slots = set()

        def _submit_slot(slot_idx: int):
            nonlocal next_seed
            seed = next_seed
            next_seed += 1
            fut = executor.submit(
                _run_single_problem_job,
                slot_idx,
                seed,
                out_root,
                target_difficulty,
                family_override=target_family,
            )
            futures[fut] = (slot_idx, seed)
            inflight_slots.add(slot_idx)

        def _refill_slots():
            while len(futures) < max_workers and pending_slots:
                slot_idx = pending_slots.pop(0)
                if slot_idx in completed_slots or slot_idx in inflight_slots:
                    continue
                _submit_slot(slot_idx)

        _refill_slots()
        while completed_success < total:
            if not futures:
                _refill_slots()
                if not futures:
                    break
            done, _ = wait(list(futures.keys()), return_when=FIRST_COMPLETED)
            for fut in done:
                idx, seed = futures.pop(fut)
                inflight_slots.discard(idx)
                try:
                    job_dir = fut.result()
                    success_records.append((idx + 1, seed, job_dir))
                    completed_slots.add(idx)
                    completed_success += 1
                    print(f"[BATCH] Problem {idx + 1} (seed={seed}) completed; output dir: {job_dir}")
                except Exception as exc:
                    retry_attempts.append((idx + 1, seed, str(exc)))
                    print(f"[BATCH][ERROR] Problem {idx + 1} (seed={seed}) failed: {exc}; will try regenerating with a new seed")
                    if idx not in completed_slots:
                        pending_slots.insert(0, idx)
                _print_progress()
            _refill_slots()

    _print_batch_summary(success_records, failure_records, retry_attempts=retry_attempts)


def run_parallel_author_solver_teacher_matrix(
    num_per_combo: Optional[int] = None,
    *,
    counts: Optional[Dict[str, Dict[str, int]]] = None,
    combo_counts: Optional[List[Tuple[str, str, int]]] = None,
    families: Optional[List[str]] = None,
    difficulties: Optional[List[str]] = None,
    base_seed: int = 5656,
    show_when_single: Optional[bool] = None,
    max_workers: Optional[int] = None,
):
    def _normalize_list(values: Optional[List[str]], fallback: List[str]) -> List[str]:
        if values is None:
            return list(fallback)
        if isinstance(values, str):
            items = [v.strip() for v in values.split(",")]
            return [v for v in items if v]
        return [str(v).strip() for v in values if str(v).strip()]

    fam_list = _normalize_list(families, ["euclid", "vector_coord", "function_calculus"])
    diff_list = _normalize_list(difficulties, ["easy", "medium", "hard"])

    pending_jobs: List[Tuple[int, str, str]] = []
    job_id = 0
    total = 0

    if combo_counts:
        for fam, diff, cnt in combo_counts:
            try:
                cnt = int(cnt)
            except Exception:
                cnt = 0
            if cnt <= 0:
                continue
            for _ in range(cnt):
                job_id += 1
                pending_jobs.append((job_id, str(fam), str(diff)))
            total += cnt
    elif counts:
        for fam, diff_map in counts.items():
            if not isinstance(diff_map, dict):
                continue
            for diff, cnt in diff_map.items():
                try:
                    cnt = int(cnt)
                except Exception:
                    cnt = 0
                if cnt <= 0:
                    continue
                for _ in range(cnt):
                    job_id += 1
                    pending_jobs.append((job_id, str(fam), str(diff)))
                total += cnt
    else:
        try:
            num_per_combo = int(num_per_combo or 0)
        except Exception:
            num_per_combo = 0
        if num_per_combo <= 0 or not fam_list or not diff_list:
            print("[BATCH] No valid problem count/family/difficulty specified; skipping run")
            return
        combos = [(fam, diff) for fam in fam_list for diff in diff_list]
        total = num_per_combo * len(combos)
        for fam, diff in combos:
            for _ in range(num_per_combo):
                job_id += 1
                pending_jobs.append((job_id, fam, diff))

    if total <= 0:
        print("[BATCH] Combination is empty; skipping run")
        return

    cfg_parallel = max(1, int(CONFIG.batch_parallel_jobs or 1))
    show_single = show_when_single if show_when_single is not None else bool(CONFIG.show_single_job_plot)
    show_single = show_single and total == 1
    max_workers = max_workers if max_workers is not None else cfg_parallel
    max_workers = max(1, min(int(max_workers), total))

    completed_success = 0
    success_records: List[Tuple[int, int, str, str, str]] = []
    failure_records: List[Tuple[int, int, str, str, str]] = []
    retry_attempts: List[Tuple[int, int, str, str, str]] = []
    next_seed = base_seed

    def _print_progress():
        if total <= 0:
            return
        bar_len = 30
        ratio = completed_success / total
        filled = min(bar_len, int(bar_len * ratio))
        bar = "#" * filled + "-" * (bar_len - filled)
        print(f"[BATCH][PROGRESS] [{bar}] {completed_success}/{total}")

    if max_workers <= 1:
        while completed_success < total and pending_jobs:
            job_idx, fam, diff = pending_jobs.pop(0)
            seed = next_seed
            next_seed += 1
            out_root = _resolve_output_root(diff, fam)
            os.makedirs(out_root, exist_ok=True)
            try:
                job_dir = _run_single_problem_job(
                    job_idx,
                    seed,
                    out_root,
                    diff,
                    enable_show=show_single,
                    family_override=fam,
                )
                success_records.append((job_idx, seed, fam, diff, job_dir))
                completed_success += 1
                print(f"[BATCH] Problem {job_idx} ({fam}/{diff}, seed={seed}) completed; output dir: {job_dir}")
            except Exception as exc:
                retry_attempts.append((job_idx, seed, fam, diff, str(exc)))
                print(f"[BATCH][ERROR] Problem {job_idx} ({fam}/{diff}, seed={seed}) failed: {exc}; will try regenerating with a new seed")
                pending_jobs.insert(0, (job_idx, fam, diff))
            _print_progress()
        _print_batch_summary(success_records, failure_records, retry_attempts=retry_attempts)
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: Dict[Any, Tuple[int, str, str, int, str]] = {}
        inflight_jobs: set = set()

        def _submit_job(job: Tuple[int, str, str]):
            nonlocal next_seed
            job_idx, fam, diff = job
            seed = next_seed
            next_seed += 1
            out_root = _resolve_output_root(diff, fam)
            os.makedirs(out_root, exist_ok=True)
            fut = executor.submit(
                _run_single_problem_job,
                job_idx,
                seed,
                out_root,
                diff,
                family_override=fam,
                enable_show=show_single,
            )
            futures[fut] = (job_idx, fam, diff, seed, out_root)
            inflight_jobs.add(job_idx)

        def _refill_jobs():
            while len(futures) < max_workers and pending_jobs:
                job = pending_jobs.pop(0)
                if job[0] in inflight_jobs:
                    continue
                _submit_job(job)

        _refill_jobs()
        while completed_success < total:
            if not futures:
                _refill_jobs()
                if not futures:
                    break
            done, _ = wait(list(futures.keys()), return_when=FIRST_COMPLETED)
            for fut in done:
                job_idx, fam, diff, seed, out_root = futures.pop(fut)
                inflight_jobs.discard(job_idx)
                try:
                    job_dir = fut.result()
                    success_records.append((job_idx, seed, fam, diff, job_dir))
                    completed_success += 1
                    print(f"[BATCH] Problem {job_idx} ({fam}/{diff}, seed={seed}) completed; output dir: {job_dir}")
                except Exception as exc:
                    retry_attempts.append((job_idx, seed, fam, diff, str(exc)))
                    print(f"[BATCH][ERROR] Problem {job_idx} ({fam}/{diff}, seed={seed}) failed: {exc}; will try regenerating with a new seed")
                    pending_jobs.insert(0, (job_idx, fam, diff))
                _print_progress()
            _refill_jobs()

    _print_batch_summary(success_records, failure_records, retry_attempts=retry_attempts)


def run_generation_from_config():
    if CONFIG.matrix_enable or CONFIG.matrix_counts or CONFIG.matrix_combo_counts:
        run_parallel_author_solver_teacher_matrix(
            num_per_combo=CONFIG.matrix_num_per_combo,
            counts=CONFIG.matrix_counts,
            combo_counts=CONFIG.matrix_combo_counts,
            families=CONFIG.matrix_families,
            difficulties=CONFIG.matrix_difficulties,
            base_seed=CONFIG.matrix_base_seed,
            max_workers=CONFIG.matrix_max_workers,
            show_when_single=CONFIG.show_single_job_plot,
        )
        return
    run_parallel_author_solver_teacher(base_seed=10)


# CLI entry point moved to verigeo/cli.py — use `python -m verigeo`.
