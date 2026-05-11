#!/usr/bin/env python3
"""evaluate_build123d_exec.py — Execution-based eval of base vs tuned models.

Companion to evaluate_build123d.py. That one does cheap static proxy metrics;
this one runs every generated code block in a subprocess and measures the
geometric outcome. Slower (~30-90 s per generation) but answers the question
"does it actually produce a valid 3D model?" rather than "is the syntax
plausible?".

Scoring layers per generation (each subsumes the previous):
  L1 syntax_ok      — ast.parse() succeeds on the extracted ```python block
  L2 api_ok         — exec'd code does NOT raise NameError/ImportError/
                      AttributeError (i.e. only references real symbols)
  L3 exec_ok        — exec completes, top-level `root_part` is defined and
                      is a build123d Part
  L4 step_export_ok — export_step(root_part, ...) writes a non-empty file
  L5 invariants_ok  — (optional, requires --ref-dir) volume, bbox dims, and
                      face count of the generated part are within --tol of
                      the reference STEP file's invariants

Input format: the picked-subset JSONL produced by the eval-picker, i.e. rows
shaped {orig_index, task_type, messages, tools, metadata, ...}. We use the
system message + first user message as the prompt (single-turn).

Output:
  --out/per_example/{idx:04d}.json — full record per row (prompts, completions,
                                     scores, errors, query/exec timings)
  --out/summary.json               — aggregate counts + rates per side per layer
  stdout: markdown table

The eval-picker file build123d-eval-picks-v1.jsonl is reproducible — it
derives from the trainer's holdout (seed=42, test_size=0.10) and then picks
100 with random.Random(42). See /mnt/tank/eval/build123d-eval-picks-v1.indices.json
for provenance.

Usage (HTTP mode against two deployed vLLM endpoints):

    python evaluate_build123d_exec.py \\
        --base-url    http://192.168.44.36:8001/v1 \\
        --base-model  Qwen/Qwen3.6-27B \\
        --ft-url      http://192.168.44.36:8002/v1 \\
        --ft-model    qwen3.6-27b-base-lora-attn-mlp \\
        --eval-set    /mnt/tank/eval/build123d-eval-picks-v1.jsonl \\
        --out         /mnt/tank/eval/results/run-$(date +%Y%m%d-%H%M)

Optional flags:
    --limit N            — score only the first N rows (debug)
    --start-from N       — resume from index N (idempotent: skips rows whose
                           per_example/{idx:04d}.json already exists)
    --side base|ft|both  — score only one side (default both)
    --max-tokens 4096    — completion cap
    --temperature 0.0    — deterministic by default
    --exec-timeout 90    — subprocess timeout in seconds
    --concurrency 4      — parallel HTTP requests per side (NOT execution —
                           execution is always serial, since CAD ops are
                           CPU-heavy and parallel runs would just contend)
    --ref-dir /path      — directory of reference STEP files for L5
                           (matched by metadata.example_id; if absent, L5 is
                           reported as 'no_reference' and treated as a miss)

The script is dependency-light: stdlib only for the harness itself. The
subprocess that executes generated code needs `build123d` (and optionally
`bd_warehouse`, `gridfinity_build123d`) installed in the same Python the
harness uses. If those are missing, the subprocess will fail at import time
and every L2 check will fail — run on a node that has them.
"""

import argparse
import ast
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)

# The user's production template (per the system prompt the dataset embeds)
# wraps generated code with these imports + a final export_step. We replicate
# it so that L3/L4 measure what the user actually deploys, not bare exec.
# bd_warehouse / gridfinity_build123d are optional; if not installed, the
# try/except keeps the import phase silent and any code that uses them will
# raise NameError → L2 fail (the correct outcome for a model that hallucinated
# a class that doesn't exist in this environment).
EXEC_TEMPLATE = r"""
from build123d import *
try:
    from bd_warehouse.thread import IsoThread, AcmeThread, MetricTrapezoidalThread
except ImportError: pass
try:
    from bd_warehouse.fastener import (
        CounterSunkScrew, HexHeadScrew, SocketHeadCapScrew,
        ButtonHeadScrew, PanHeadScrew, FlatHeadScrew, ChamferedScrew,
    )
except ImportError: pass
try:
    from bd_warehouse.bearing import SingleRowDeepGrooveBallBearing
except ImportError: pass
try:
    from bd_warehouse.gear import SpurGear
except ImportError: pass
try:
    from bd_warehouse.pipe import Pipe, PipeSection
except ImportError: pass
try:
    from gridfinity_build123d import (
        Bin, Base, BaseEqual, BasePlate, BasePlateEqual,
        Compartment, CompartmentsEqual,
    )
except ImportError: pass

# --- BEGIN USER CODE ---
{user_code}
# --- END USER CODE ---

import json as _json, os as _os, sys as _sys, tempfile as _tempfile

if "root_part" not in dir():
    raise NameError("root_part not defined at module top level")

_rp = root_part
try:
    _vol = float(_rp.volume) if hasattr(_rp, "volume") else None
except Exception:
    _vol = None
try:
    _bb = _rp.bounding_box() if hasattr(_rp, "bounding_box") else None
    _bbox = ([_bb.min.X, _bb.min.Y, _bb.min.Z],
             [_bb.max.X, _bb.max.Y, _bb.max.Z]) if _bb else None
except Exception:
    _bbox = None
try:
    _faces = len(_rp.faces()) if hasattr(_rp, "faces") else None
except Exception:
    _faces = None

_step_ok = False
_step_size = 0
_tmp_path = None
try:
    _tmp = _tempfile.NamedTemporaryFile(suffix=".step", delete=False)
    _tmp.close()
    _tmp_path = _tmp.name
    export_step(_rp, _tmp_path)
    _step_size = _os.path.getsize(_tmp_path)
    _step_ok = _step_size > 0
except Exception as _e:
    _step_err = type(_e).__name__ + ": " + str(_e)[:200]
finally:
    if _tmp_path and _os.path.exists(_tmp_path):
        try: _os.unlink(_tmp_path)
        except Exception: pass

_sys.stderr.write("__EVAL_RESULT__" + _json.dumps({
    "root_part_ok": True,
    "volume": _vol,
    "bbox": _bbox,
    "face_count": _faces,
    "step_export_ok": _step_ok,
    "step_size_bytes": _step_size,
}) + "\n")
"""


def extract_code_block(text: str) -> str:
    """Return the first ```python``` (or just ```) fenced block, or '' if none."""
    if not text:
        return ""
    m = CODE_BLOCK_RE.search(text)
    if m:
        return m.group(1)
    # Fallback: some models drop the fence. Treat the whole response as code
    # only if it has at least one obvious build123d hint, otherwise return ''
    # so L1 fails cleanly rather than mis-attributing a chatty refusal as code.
    if "build123d" in text or "BuildPart" in text or "root_part" in text:
        return text
    return ""


def check_syntax(code: str):
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"


def execute_in_subprocess(code: str, timeout: int):
    """Run wrapped code in a fresh Python subprocess. Returns a dict whose
    `category` is one of:
        ok           — completed; result fields populated
        api_error    — NameError/ImportError/AttributeError before/during exec
        runtime      — any other Exception during exec
        timeout      — subprocess killed by deadline
        no_code      — input code was empty
        no_result    — process exited 0 but didn't emit __EVAL_RESULT__"""
    if not code.strip():
        return {"category": "no_code"}

    script = EXEC_TEMPLATE.replace("{user_code}", code)
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"category": "timeout", "error": f"timeout after {timeout}s"}

    if result.returncode == 0:
        for line in (result.stderr or "").splitlines():
            if line.startswith("__EVAL_RESULT__"):
                return {"category": "ok", **json.loads(line[len("__EVAL_RESULT__"):])}
        return {"category": "no_result", "stderr_tail": (result.stderr or "")[-500:]}

    err = result.stderr or ""
    # Categorize by the first error type token in the traceback's last frames
    tail = err[-3000:]
    if any(t in tail for t in ("NameError", "ImportError", "ModuleNotFoundError", "AttributeError")):
        cat = "api_error"
    elif "SyntaxError" in tail and "<string>" in tail:
        cat = "syntax_error"  # rare — would normally be caught at L1
    else:
        cat = "runtime"

    # Last non-traceback-frame line is the exception summary
    err_line = ""
    for line in reversed(tail.splitlines()):
        s = line.strip()
        if s and not s.startswith("File ") and not s.startswith("Traceback"):
            err_line = s[:300]
            break
    return {"category": cat, "error": err_line, "stderr_tail": tail[-600:]}


def score_completion(completion: str, exec_timeout: int) -> dict:
    """Compute L1-L4 for a completion. L5 is layered on later if --ref-dir."""
    code = extract_code_block(completion)
    has_code = bool(code.strip())

    syntax_ok, syntax_err = (check_syntax(code) if has_code else (False, "no python code block"))

    result = {
        "has_code_block": has_code,
        "L1_syntax_ok": syntax_ok,
        "L2_api_ok": False,
        "L3_exec_ok": False,
        "L4_step_ok": False,
        "syntax_error": syntax_err,
        "exec_category": None,
        "exec_error": None,
        "volume": None,
        "face_count": None,
        "bbox": None,
        "step_size_bytes": 0,
    }
    if not syntax_ok:
        return result

    exec_t0 = time.time()
    er = execute_in_subprocess(code, timeout=exec_timeout)
    result["exec_seconds"] = round(time.time() - exec_t0, 2)
    result["exec_category"] = er.get("category")
    result["exec_error"] = er.get("error")
    result["exec_stderr_tail"] = er.get("stderr_tail")

    # L2 fails on api_error (NameError/AttributeError/Import); passes on
    # everything else that got past L1 (including runtime errors).
    result["L2_api_ok"] = er.get("category") != "api_error"
    result["L3_exec_ok"] = (
        er.get("category") == "ok" and bool(er.get("root_part_ok"))
    )
    result["L4_step_ok"] = bool(er.get("step_export_ok"))
    result["volume"] = er.get("volume")
    result["face_count"] = er.get("face_count")
    result["bbox"] = er.get("bbox")
    result["step_size_bytes"] = er.get("step_size_bytes", 0)
    return result


def compare_invariants(gen: dict, ref: dict, tol: float) -> dict:
    """L5: compare generated vs reference geometric invariants. Returns
    {L5_invariants_ok, details} where details is per-invariant pass/fail.
    Reference dict is expected to have the same shape as gen (volume,
    bbox, face_count)."""
    out = {"L5_invariants_ok": True, "details": {}}

    def _close(g, r, rel_tol):
        if g is None or r is None:
            return None
        if rel_tol == 0.0:
            return g == r
        return abs(g - r) <= rel_tol * max(abs(r), 1e-9)

    # Volume — relative tol
    v_g, v_r = gen.get("volume"), ref.get("volume")
    v_ok = _close(v_g, v_r, tol)
    out["details"]["volume"] = {"g": v_g, "r": v_r, "ok": v_ok}
    if v_ok is not True:
        out["L5_invariants_ok"] = False

    # Face count — exact match
    f_g, f_r = gen.get("face_count"), ref.get("face_count")
    f_ok = _close(f_g, f_r, 0.0)
    out["details"]["face_count"] = {"g": f_g, "r": f_r, "ok": f_ok}
    if f_ok is not True:
        out["L5_invariants_ok"] = False

    # Bounding box dimensions
    g_bb, r_bb = gen.get("bbox"), ref.get("bbox")
    if g_bb and r_bb:
        g_dims = tuple(g_bb[1][i] - g_bb[0][i] for i in range(3))
        r_dims = tuple(r_bb[1][i] - r_bb[0][i] for i in range(3))
        dim_oks = [_close(g, r, tol) for g, r in zip(g_dims, r_dims)]
        all_ok = all(x is True for x in dim_oks)
        out["details"]["bbox_dims"] = {"g": list(g_dims), "r": list(r_dims),
                                       "per_axis_ok": dim_oks, "ok": all_ok}
        if not all_ok:
            out["L5_invariants_ok"] = False
    else:
        out["details"]["bbox_dims"] = {"g": g_bb, "r": r_bb, "ok": None}
        out["L5_invariants_ok"] = False
    return out


def load_reference_invariants(ref_dir: Path, example_id: str, exec_timeout: int) -> dict:
    """Load reference STEP for example_id and compute its invariants. Returns
    a dict shaped like score_completion's output, or {} if no reference."""
    if not ref_dir or not example_id:
        return {}
    for ext in (".step", ".stp"):
        path = ref_dir / f"{example_id}{ext}"
        if path.exists():
            # Run a small script that imports the STEP and prints invariants
            probe = (
                "from build123d import import_step\n"
                f"_rp = import_step('{path}')\n"
                "import json, sys\n"
                "_bb = _rp.bounding_box()\n"
                "sys.stderr.write('__EVAL_RESULT__' + json.dumps({\n"
                "    'volume': float(_rp.volume),\n"
                "    'bbox': ([_bb.min.X, _bb.min.Y, _bb.min.Z],\n"
                "             [_bb.max.X, _bb.max.Y, _bb.max.Z]),\n"
                "    'face_count': len(_rp.faces()),\n"
                "}))\n"
            )
            try:
                r = subprocess.run(
                    [sys.executable, "-c", probe],
                    capture_output=True, text=True, timeout=exec_timeout,
                )
                for line in (r.stderr or "").splitlines():
                    if line.startswith("__EVAL_RESULT__"):
                        return json.loads(line[len("__EVAL_RESULT__"):])
            except subprocess.TimeoutExpired:
                pass
            return {}
    return {}


def query_chat(url: str, model: str, system_msg: str, user_msg: str,
               max_tokens: int, temperature: float, api_key: str | None,
               timeout: int) -> tuple[str, float, str | None]:
    """POST to OpenAI-compatible /v1/chat/completions. Returns
    (completion_text, elapsed_s, error_or_none)."""
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = Request(url.rstrip("/") + "/chat/completions", data=body, headers=headers)
    t0 = time.time()
    try:
        with urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
        text = data["choices"][0]["message"]["content"] or ""
        return text, round(time.time() - t0, 2), None
    except HTTPError as e:
        body_tail = ""
        try:
            body_tail = e.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        return "", round(time.time() - t0, 2), f"HTTP {e.code}: {body_tail}"
    except URLError as e:
        return "", round(time.time() - t0, 2), f"URLError: {e.reason}"
    except (json.JSONDecodeError, KeyError) as e:
        return "", round(time.time() - t0, 2), f"BadResponse: {e}"


def first_messages(messages: list) -> tuple[str, str]:
    """Return (system_content, first_user_content). Multi-turn convos are
    collapsed to single-turn at the eval boundary."""
    sys_msg = ""
    user_msg = ""
    for m in messages or []:
        role = m.get("role")
        if role == "system" and not sys_msg:
            sys_msg = m.get("content") or ""
        elif role == "user" and not user_msg:
            user_msg = m.get("content") or ""
            break
    return sys_msg, user_msg


def evaluate_row(idx: int, row: dict, args, ref_invariants_cache: dict | None = None) -> dict:
    """Run one row through both sides (or one side, per args.side) and
    return the full record."""
    sys_msg, user_msg = first_messages(row.get("messages", []))
    meta = row.get("metadata") or {}
    record = {
        "idx": idx,
        "orig_index": row.get("orig_index"),
        "category": meta.get("category"),
        "example_id": meta.get("example_id") or meta.get("prompt_id"),
        "user_msg_preview": (user_msg or "")[:300],
        "system_msg_preview": (sys_msg or "")[:200],
    }

    sides = []
    if args.side in ("base", "both"):
        sides.append(("base", args.base_url, args.base_model))
    if args.side in ("ft", "both"):
        sides.append(("ft", args.ft_url, args.ft_model))

    for label, url, model in sides:
        if not url or not model:
            record[label] = {"skipped": True, "reason": "no url/model provided"}
            continue
        text, q_seconds, q_err = query_chat(
            url, model, sys_msg, user_msg,
            args.max_tokens, args.temperature, args.api_key,
            timeout=args.http_timeout,
        )
        if q_err:
            record[label] = {
                "query_error": q_err, "query_seconds": q_seconds,
                "completion": "", "has_code_block": False,
                "L1_syntax_ok": False, "L2_api_ok": False,
                "L3_exec_ok": False, "L4_step_ok": False,
            }
            continue
        scores = score_completion(text, exec_timeout=args.exec_timeout)
        scores["completion"] = text
        scores["query_seconds"] = q_seconds
        record[label] = scores

    # L5 (optional): compare ft.invariants against reference STEP, if any
    if args.ref_dir and record.get("example_id"):
        ref = ref_invariants_cache.get(record["example_id"]) if ref_invariants_cache is not None else None
        if ref is None:
            ref = load_reference_invariants(
                Path(args.ref_dir), record["example_id"], exec_timeout=args.exec_timeout,
            )
            if ref_invariants_cache is not None:
                ref_invariants_cache[record["example_id"]] = ref
        record["reference_invariants"] = ref if ref else None
        for label in ("base", "ft"):
            s = record.get(label)
            if not s or s.get("skipped") or not s.get("L3_exec_ok"):
                continue
            if not ref:
                s["L5_invariants_ok"] = False
                s["L5_reason"] = "no_reference"
                continue
            s["L5"] = compare_invariants(s, ref, tol=args.tol)
            s["L5_invariants_ok"] = s["L5"]["L5_invariants_ok"]

    return record


def aggregate(records: list, side: str) -> dict:
    """Sum L1-L5 booleans across records for one side."""
    out = {f"L{i}": 0 for i in range(1, 6)}
    out["has_code_block"] = 0
    out["queried"] = 0
    out["query_error"] = 0
    for r in records:
        s = r.get(side, {})
        if not s or s.get("skipped"):
            continue
        if s.get("query_error"):
            out["query_error"] += 1
            continue
        out["queried"] += 1
        if s.get("has_code_block"):
            out["has_code_block"] += 1
        for L in range(1, 6):
            key = {
                1: "L1_syntax_ok",
                2: "L2_api_ok",
                3: "L3_exec_ok",
                4: "L4_step_ok",
                5: "L5_invariants_ok",
            }[L]
            if s.get(key):
                out[f"L{L}"] += 1
    return out


def print_summary(records: list, args):
    base_agg = aggregate(records, "base")
    ft_agg = aggregate(records, "ft")
    total = max(base_agg["queried"], ft_agg["queried"], 1)

    print("\n" + "=" * 64)
    print(f"  RESULTS ({total} examples; ref_dir={'yes' if args.ref_dir else 'no'})")
    print("=" * 64)
    print(f"  {'Metric':<22} {'base':>10} {'ft':>10}  {'delta':>8}")
    print(f"  {'-' * 22} {'-' * 10} {'-' * 10}  {'-' * 8}")
    for label, key in [
        ("has_code_block",     "has_code_block"),
        ("L1 syntax_ok",       "L1"),
        ("L2 api_ok",          "L2"),
        ("L3 exec_ok",         "L3"),
        ("L4 step_export_ok",  "L4"),
        ("L5 invariants_ok",   "L5"),
        ("query_error",        "query_error"),
    ]:
        b = base_agg[key]
        f = ft_agg[key]
        delta = f - b
        b_pct = 100 * b / max(total, 1)
        f_pct = 100 * f / max(total, 1)
        print(f"  {label:<22} {b:>3}/{total} ({b_pct:>4.0f}%) {f:>3}/{total} ({f_pct:>4.0f}%)  {delta:>+4d}")
    print("=" * 64)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--base-url",     help="OpenAI-compatible base URL, e.g. http://host:port/v1")
    ap.add_argument("--base-model",   help="model name vLLM serves on --base-url")
    ap.add_argument("--ft-url",       help="OpenAI-compatible base URL for the fine-tuned model")
    ap.add_argument("--ft-model",     help="model name vLLM serves on --ft-url")
    ap.add_argument("--api-key",      default=None, help="optional bearer token for both endpoints")
    ap.add_argument("--eval-set",     default="/mnt/tank/eval/build123d-eval-picks-v1.jsonl",
                    help="path to picked eval JSONL (one full row per line)")
    ap.add_argument("--out",          required=True, help="output directory")
    ap.add_argument("--limit",        type=int, default=0, help="0=all rows")
    ap.add_argument("--start-from",   type=int, default=0)
    ap.add_argument("--side",         choices=["base", "ft", "both"], default="both")
    ap.add_argument("--max-tokens",   type=int, default=4096)
    ap.add_argument("--temperature",  type=float, default=0.0)
    ap.add_argument("--exec-timeout", type=int, default=90,
                    help="subprocess timeout for executing generated code (s)")
    ap.add_argument("--http-timeout", type=int, default=600,
                    help="HTTP timeout for the completion request (s)")
    ap.add_argument("--concurrency",  type=int, default=1,
                    help="parallel HTTP requests (only HTTP — exec is always serial)")
    ap.add_argument("--ref-dir",      default=None,
                    help="directory of reference STEP files (matched by metadata.example_id) for L5")
    ap.add_argument("--tol",          type=float, default=0.10,
                    help="relative tolerance for volume + bbox dims (face count is exact)")
    args = ap.parse_args()

    if args.side in ("base", "both") and not (args.base_url and args.base_model):
        raise SystemExit("--side base/both requires --base-url and --base-model")
    if args.side in ("ft", "both") and not (args.ft_url and args.ft_model):
        raise SystemExit("--side ft/both requires --ft-url and --ft-model")

    out_dir = Path(args.out)
    (out_dir / "per_example").mkdir(parents=True, exist_ok=True)

    rows = []
    with open(args.eval_set) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.limit > 0:
        rows = rows[:args.limit]

    print(f"Loaded {len(rows)} rows from {args.eval_set}")
    print(f"Mode: side={args.side}, exec_timeout={args.exec_timeout}s, "
          f"concurrency={args.concurrency}, ref_dir={args.ref_dir or 'none'}")

    ref_cache = {} if args.ref_dir else None

    # Collect tasks that aren't already done (resume idempotency)
    pending_indices = []
    cached_records = {}
    for idx in range(args.start_from, len(rows)):
        cache_path = out_dir / "per_example" / f"{idx:04d}.json"
        if cache_path.exists():
            try:
                cached_records[idx] = json.loads(cache_path.read_text())
                continue
            except Exception:
                pass
        pending_indices.append(idx)

    print(f"{len(cached_records)} cached, {len(pending_indices)} to run")

    wall0 = time.time()
    done = 0
    total_to_run = len(pending_indices)

    def _process(idx):
        return idx, evaluate_row(idx, rows[idx], args, ref_invariants_cache=ref_cache)

    # Parallelize HTTP fan-out per row. Execution is still serial inside
    # evaluate_row (the subprocess calls run sequentially), so concurrency
    # really only buys HTTP overlap.
    if args.concurrency > 1 and total_to_run > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            for idx, rec in ex.map(_process, pending_indices):
                done += 1
                _emit(idx, rec, out_dir, done, total_to_run)
    else:
        for idx in pending_indices:
            _, rec = _process(idx)
            done += 1
            _emit(idx, rec, out_dir, done, total_to_run)

    # Reload everything for the summary
    all_records = []
    for idx in range(len(rows)):
        p = out_dir / "per_example" / f"{idx:04d}.json"
        if p.exists():
            all_records.append(json.loads(p.read_text()))

    summary = {
        "n_total": len(all_records),
        "wall_seconds": round(time.time() - wall0, 1),
        "base_agg": aggregate(all_records, "base"),
        "ft_agg":   aggregate(all_records, "ft"),
        "config": {
            "base_url": args.base_url, "base_model": args.base_model,
            "ft_url":   args.ft_url,   "ft_model":   args.ft_model,
            "eval_set": args.eval_set, "max_tokens": args.max_tokens,
            "temperature": args.temperature, "exec_timeout": args.exec_timeout,
            "tol": args.tol, "ref_dir": args.ref_dir,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print_summary(all_records, args)
    print(f"\nWrote {len(all_records)} per-example records and summary.json to {out_dir}")


def _emit(idx: int, rec: dict, out_dir: Path, done: int, total: int):
    (out_dir / "per_example" / f"{idx:04d}.json").write_text(json.dumps(rec, indent=2))
    b = rec.get("base") or {}
    f = rec.get("ft") or {}

    def _sig(s):
        if not s or s.get("skipped"):
            return "----"
        if s.get("query_error"):
            return "QERR"
        return "".join("Y" if s.get(k) else "." for k in
                       ("L1_syntax_ok", "L2_api_ok", "L3_exec_ok", "L4_step_ok"))

    b_q = (b.get("query_seconds") if b else None) or 0
    f_q = (f.get("query_seconds") if f else None) or 0
    print(f"  [{idx:04d}] {done}/{total}  base={_sig(b)} ({b_q:.0f}s)  "
          f"ft={_sig(f)} ({f_q:.0f}s)  cat={rec.get('category', '?')}")


if __name__ == "__main__":
    main()
