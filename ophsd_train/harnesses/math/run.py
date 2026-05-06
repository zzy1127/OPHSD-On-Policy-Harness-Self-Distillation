#!/usr/bin/env python3
"""Standalone Plan-Solve harness runner: zero-shot vs. harness on a math parquet.

This is a *standalone* evaluation tool — useful for sanity-checking the harness
prompts against any vLLM-served checkpoint without spinning up the OPHSD
trainer.  Results are appended JSONL files so the runner is resumable.

Usage examples
--------------
Local vLLM (recommended for reproducibility):

    export HARNESS_USE_LOCAL_MODEL=1
    export HARNESS_LOCAL_MODEL_PATH=/path/to/Qwen3-8B
    python -m harnesses.math.run \\
        --data data/test/math/test.parquet \\
        --mode both --max-test 100 --workers 64

Remote OpenAI-style API:

    export HARNESS_API_KEY=...
    export HARNESS_API_BASE=https://api.openai.com/v1
    export HARNESS_CHAT_MODEL=gpt-4o
    python -m harnesses.math.run --data my_test.parquet --mode harness
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .evaluate import evaluate_predictions, extract_boxed, format_results_table
from .plan_solve import PlanSolveHarness, ZeroShotSolver
from .preprocess import load_test_data

logger = logging.getLogger("harnesses.math.run")


# ────────────────────────────────────────────────────────────────────
# CLI-only configuration (resolved from environment)
# ────────────────────────────────────────────────────────────────────

USE_LOCAL_MODEL = os.getenv("HARNESS_USE_LOCAL_MODEL", "0") == "1"
LOCAL_MODEL_PORT = int(os.getenv("HARNESS_LOCAL_MODEL_PORT", "8000"))
LOCAL_MODEL_PATH = os.getenv("HARNESS_LOCAL_MODEL_PATH", "")
LOCAL_MODEL_NAME = os.path.basename(LOCAL_MODEL_PATH) if LOCAL_MODEL_PATH else ""

API_KEY = os.getenv("HARNESS_API_KEY", "EMPTY" if USE_LOCAL_MODEL else "")
API_BASE_URL = os.getenv(
    "HARNESS_API_BASE",
    f"http://localhost:{LOCAL_MODEL_PORT}/v1" if USE_LOCAL_MODEL else "https://api.openai.com/v1",
)
CHAT_MODEL = os.getenv("HARNESS_CHAT_MODEL", LOCAL_MODEL_NAME or "gpt-4o-mini")
API_WORKERS = int(os.getenv("HARNESS_WORKERS", "32"))
RESULTS_DIR = os.getenv("HARNESS_RESULTS_DIR", "outputs/math_harness")


# ────────────────────────────────────────────────────────────────────
# I/O helpers
# ────────────────────────────────────────────────────────────────────

def _get_client():
    from openai import OpenAI

    if not USE_LOCAL_MODEL and not API_KEY:
        logger.error("No API key. Set HARNESS_API_KEY env var.")
        sys.exit(1)
    if USE_LOCAL_MODEL:
        logger.info("Using local vLLM at %s  model=%s", API_BASE_URL, CHAT_MODEL)
    return OpenAI(api_key=API_KEY, base_url=API_BASE_URL)


def _load_done(path: str) -> dict[int, dict]:
    done: dict[int, dict] = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    done[obj["idx"]] = obj
                except Exception:
                    pass
    return done


def _append(path: str, record: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _results_path(name: str) -> str:
    return os.path.join(RESULTS_DIR, f"{name}.jsonl")


# ────────────────────────────────────────────────────────────────────
# Zero-shot baseline (stateless, fully parallel)
# ────────────────────────────────────────────────────────────────────

def run_zero_shot(test_data: list[dict]) -> list[str]:
    client = _get_client()
    solver = ZeroShotSolver(client, CHAT_MODEL)
    out_path = _results_path("zero_shot")
    done = _load_done(out_path)

    predictions = [""] * len(test_data)
    for i, rec in done.items():
        if i < len(predictions):
            predictions[i] = rec["prediction"]

    todo = [(i, s) for i, s in enumerate(test_data) if i not in done]
    logger.info(
        "Zero-shot: %d total, %d done, %d remaining (workers=%d)",
        len(test_data), len(done), len(todo), API_WORKERS,
    )

    write_lock = threading.Lock()
    counter = {"n": 0}

    def _process(args):
        idx, sample = args
        t0 = time.time()
        pred = solver.solve(sample["question"])
        ans = extract_boxed(pred) or ""
        with write_lock:
            predictions[idx] = pred
            _append(out_path, {"idx": idx, "prediction": pred, "extracted": ans})
            counter["n"] += 1
            logger.info(
                "[zero_shot %d/%d] idx=%d  ans=%s  %.1fs",
                counter["n"], len(todo), idx, ans[:60], time.time() - t0,
            )
        return idx, pred

    with ThreadPoolExecutor(max_workers=API_WORKERS) as pool:
        futures = {pool.submit(_process, item): item[0] for item in todo}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:
                logger.error("idx=%d failed: %s", futures[fut], exc)
    return predictions


# ────────────────────────────────────────────────────────────────────
# Plan-Solve harness  (parallel — stateless, no shared retrieval bank)
# ────────────────────────────────────────────────────────────────────

def run_harness(test_data: list[dict]) -> list[str]:
    """Run the Plan-Solve harness fully in parallel."""
    client = _get_client()
    harness = PlanSolveHarness(client, CHAT_MODEL)
    out_path = _results_path("harness")
    trace_path = _results_path("harness_trace")
    done = _load_done(out_path)

    predictions = [""] * len(test_data)
    for i, rec in done.items():
        if i < len(predictions):
            predictions[i] = rec["prediction"]

    todo = [(i, s) for i, s in enumerate(test_data) if i not in done]
    logger.info(
        "Harness: %d total, %d done, %d remaining (workers=%d)",
        len(test_data), len(done), len(todo), API_WORKERS,
    )

    write_lock = threading.Lock()
    counter = {"n": 0}

    def _process(args):
        idx, sample = args
        t0 = time.time()
        result = harness.predict(sample["question"])
        pred = result["final_response"]
        ans = extract_boxed(pred) or ""

        with write_lock:
            predictions[idx] = pred
            _append(out_path, {
                "idx": idx,
                "prediction": pred,
                "extracted": ans,
                "mode": result["mode"],
            })
            _append(trace_path, {
                "idx": idx,
                "mode": result["mode"],
                "trace": result["trace"],
            })
            counter["n"] += 1
            logger.info(
                "[harness %d/%d] idx=%d  mode=%s  ans=%s  %.1fs",
                counter["n"], len(todo), idx,
                result["mode"], ans[:60], time.time() - t0,
            )
        return idx, pred

    with ThreadPoolExecutor(max_workers=API_WORKERS) as pool:
        futures = {pool.submit(_process, item): item[0] for item in todo}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:
                logger.error("idx=%d failed: %s", futures[fut], exc)
    return predictions


# ────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["zero_shot", "harness", "both"], default="both")
    parser.add_argument("--data", type=str, required=True, help="Path to test parquet file")
    parser.add_argument("--max-test", type=int, default=0,
                        help="Use only the first N test samples (0 = all)")
    parser.add_argument("--benchmarks", type=str, default="",
                        help="Comma-separated data_source filter (e.g. 'aime24,MATH-500')")
    parser.add_argument("--per-benchmark-max", type=int, default=0,
                        help="Max samples per benchmark after filtering (0 = no cap)")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle test data before sampling (ensures benchmark diversity)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR,
                        help=f"Override results output directory (default: {RESULTS_DIR})")
    parser.add_argument("--workers", type=int, default=API_WORKERS,
                        help=f"Parallel workers (default: {API_WORKERS})")
    parser.add_argument("--clean", action="store_true",
                        help="Remove all previous results and restart from scratch")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    global RESULTS_DIR, API_WORKERS  # noqa: PLW0603 — CLI-side overrides
    RESULTS_DIR = args.results_dir
    API_WORKERS = args.workers

    if args.clean:
        for name in ("zero_shot", "harness", "harness_trace"):
            p = _results_path(name)
            if os.path.exists(p):
                os.remove(p)
                logger.info("Removed %s", p)
        summary = os.path.join(RESULTS_DIR, "summary.json")
        if os.path.exists(summary):
            os.remove(summary)

    logger.info("Loading test data from %s …", args.data)
    benchmark_filter = [x.strip() for x in args.benchmarks.split(",")] if args.benchmarks else None
    test_data = load_test_data(
        args.data,
        max_samples=args.max_test,
        shuffle=args.shuffle,
        seed=args.seed,
        benchmark_filter=benchmark_filter,
        per_benchmark_max=args.per_benchmark_max,
    )
    logger.info("Test samples: %d", len(test_data))

    results: dict[str, dict] = {}

    if args.mode in ("zero_shot", "both"):
        preds = run_zero_shot(test_data)
        ev = evaluate_predictions(preds, test_data)
        results["zero_shot"] = ev
        logger.info("Zero-shot  acc=%.4f  abstention=%.2f%%",
                    ev["accuracy"], ev["abstention_rate"] * 100)

    if args.mode in ("harness", "both"):
        preds = run_harness(test_data)
        ev = evaluate_predictions(preds, test_data)
        results["harness"] = ev
        logger.info("Harness    acc=%.4f  abstention=%.2f%%",
                    ev["accuracy"], ev["abstention_rate"] * 100)

    if results:
        print("\n" + format_results_table(results))
        os.makedirs(RESULTS_DIR, exist_ok=True)
        summary_path = os.path.join(RESULTS_DIR, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {k: {kk: vv for kk, vv in v.items() if kk != "per_sample"}
                 for k, v in results.items()},
                f, ensure_ascii=False, indent=2,
            )
        logger.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
