"""Math answer extraction and accuracy evaluation.

**Single source of truth for “is this answer correct?”** (pred vs gold string):
``answers_match`` in this file, together with ``_strip_string`` / ``_norm`` above it.
OPSD/GRPO and OPHSD training rewards use the same logic in their ``math_reward._answers_match``.

Supports per-benchmark breakdown when test_data contains a 'data_source' field.
"""

import logging
import re
import unicodedata
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


# ────────────────────────────────────────────────────────────────────
# Hendrycks strip_string normalisation (mirrors math_reward.py)
# ────────────────────────────────────────────────────────────────────

def _fix_fracs(s: str) -> str:
    parts = s.split("\\frac")
    out = parts[0]
    for p in parts[1:]:
        out += "\\frac"
        if p and p[0] == "{":
            out += p
        elif len(p) >= 2:
            a, b = p[0], p[1]
            rest = p[2:]
            if b != "{":
                out += "{" + a + "}{" + b + "}" + rest
            else:
                out += "{" + a + "}" + b + rest
        else:
            return s
    return out


def _fix_sqrt(s: str) -> str:
    if "\\sqrt" not in s:
        return s
    parts = s.split("\\sqrt")
    out = parts[0]
    for p in parts[1:]:
        if p and p[0] != "{":
            out += "\\sqrt{" + p[0] + "}" + p[1:]
        else:
            out += "\\sqrt" + p
    return out


def _fix_slash(s: str) -> str:
    if len(s.split("/")) != 2:
        return s
    a, b = s.split("/")
    try:
        ia, ib = int(a), int(b)
        assert s == f"{ia}/{ib}"
        return "\\frac{" + str(ia) + "}{" + str(ib) + "}"
    except Exception:
        return s


def _strip_string(s: str) -> str:
    """Normalise a math answer string (Hendrycks convention)."""
    s = s.replace("\n", "")
    s = s.replace("\\!", "")
    s = s.replace("\\\\", "\\")
    s = s.replace("tfrac", "frac")
    s = s.replace("dfrac", "frac")
    s = s.replace("\\left", "")
    s = s.replace("\\right", "")
    s = s.replace("^{\\circ}", "")
    s = s.replace("^\\circ", "")
    s = s.replace("\\$", "")
    # remove \text{ units (e.g. \text{True} -> True)
    if "\\text{" in s:
        s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = s.replace("\\\\%", "")
    s = s.replace("\\%", "")
    s = s.replace(" .", " 0.")
    s = s.replace("{.", "{0.")
    if not s:
        return s
    if s[0] == ".":
        s = "0" + s
    # strip "x = " prefix
    if len(s.split("=")) == 2 and len(s.split("=")[0]) <= 2:
        s = s.split("=")[1]
    s = _fix_sqrt(s)
    s = s.replace(" ", "")
    s = _fix_fracs(s)
    if s == "0.5":
        s = "\\frac{1}{2}"
    s = _fix_slash(s)
    return s


def _norm(s: str) -> str:
    """Unicode-normalise, lowercase, strip spaces/dollar signs."""
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"\s+", "", s).lower().strip().replace("$", "")


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def extract_boxed(text: str) -> Optional[str]:
    """Return the content of the last \\boxed{...}, handling nested braces."""
    if not text:
        return None
    text = strip_think(text)
    # Find the last occurrence of \boxed
    idx = text.rfind(r"\boxed")
    if idx < 0:
        return None
    # Walk forward to find matching closing brace
    i = idx
    depth = 0
    start = None
    while i < len(text):
        if text[i] == "{":
            depth += 1
            if depth == 1:
                start = i + 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
        i += 1
    return None   # unbalanced


def answers_match(pred: str, gold: str) -> bool:
    """Check answer equivalence, consistent with ophsd_train math_reward.py.

    Order of checks (same as training reward):
    1. _strip_string normalisation (handles dfrac/tfrac, \\text{}, \\left/\\right, etc.)
    2. Plain unicode-normalised lowercase match (handles Yes/No/True/False/letters)
    3. math_verify LaTeX-aware symbolic equivalence (handles \\frac, ^, etc.)
    """
    if not pred or not gold:
        return False

    # 1. Hendrycks strip_string normalisation
    try:
        if _strip_string(pred) == _strip_string(gold):
            return True
    except Exception:
        pass

    # 2. Plain normalised string (catches Yes/No/True/False after Unicode normalisation)
    if _norm(pred) == _norm(gold):
        return True

    # 3. math_verify symbolic check
    try:
        from math_verify import parse, verify as mv_verify
        p_wrapped = pred if "$" in pred else f"${pred}$"
        g_wrapped = gold if "$" in gold else f"${gold}$"
        p_parsed = parse(p_wrapped, fallback_mode="no_fallback", parsing_timeout=None)
        g_parsed = parse(g_wrapped, fallback_mode="no_fallback", parsing_timeout=None)
        return mv_verify(g_parsed, p_parsed, timeout_seconds=None)
    except Exception:
        pass

    return False


# ────────────────────────────────────────────────────────────────────
# Batch evaluation  (overall + per-benchmark breakdown)
# ────────────────────────────────────────────────────────────────────

def _bench_stats(correct: int, abstentions: int, total: int) -> dict:
    return {
        "accuracy":        correct / total if total else 0.0,
        "correct":         correct,
        "total":           total,
        "abstention_rate": abstentions / total if total else 0.0,
    }


def evaluate_predictions(predictions: list[str], test_data: list[dict]) -> dict:
    """Compute accuracy overall and broken down by data_source / benchmark.

    Returns dict with keys:
        accuracy          float    overall
        correct           int
        total             int
        abstention_rate   float
        per_benchmark     dict[source_name -> {accuracy, correct, total, abstention_rate}]
        per_sample        list[dict]  — per-sample detail
    """
    assert len(predictions) == len(test_data), (
        f"predictions length {len(predictions)} != test_data length {len(test_data)}"
    )

    overall_correct    = 0
    overall_abstentions = 0

    # per-benchmark counters
    bench_correct:     dict[str, int] = defaultdict(int)
    bench_abstentions: dict[str, int] = defaultdict(int)
    bench_total:       dict[str, int] = defaultdict(int)

    per_sample = []

    for pred_raw, sample in zip(predictions, test_data):
        gold   = sample["ground_truth"]
        source = sample.get("data_source", "unknown")

        extracted  = extract_boxed(pred_raw) if pred_raw else None
        abstained  = extracted is None
        is_correct = False if abstained else answers_match(extracted, gold)

        if abstained:
            overall_abstentions     += 1
            bench_abstentions[source] += 1
        if is_correct:
            overall_correct     += 1
            bench_correct[source] += 1
        bench_total[source] += 1

        per_sample.append({
            "question_id":  sample.get("question_id", ""),
            "data_source":  source,
            "ground_truth": gold,
            "extracted":    extracted,
            "correct":      is_correct,
        })

    total = len(test_data)

    per_benchmark = {
        src: _bench_stats(
            bench_correct[src],
            bench_abstentions[src],
            bench_total[src],
        )
        for src in bench_total
    }

    return {
        **_bench_stats(overall_correct, overall_abstentions, total),
        "per_benchmark": per_benchmark,
        "per_sample":    per_sample,
    }


def format_results_table(results: dict[str, dict]) -> str:
    """Format a method → eval_dict mapping into a printable table.

    results = {"zero_shot": evaluate_predictions(...), "harness": evaluate_predictions(...)}
    """
    methods  = list(results.items())
    # Collect all benchmark names across all methods
    benches  = sorted({b for _, ev in methods for b in ev.get("per_benchmark", {})})

    lines = []
    sep   = "=" * 74

    lines.append(sep)
    lines.append(f"{'Method':<16} {'Overall':>9} {'Abstain':>8}   " +
                 "  ".join(f"{b[:14]:>14}" for b in benches))
    lines.append("-" * 74)

    for name, ev in methods:
        overall = f"{ev['accuracy']*100:>8.2f}%"
        abstain = f"{ev['abstention_rate']*100:>7.1f}%"
        bench_cols = []
        for b in benches:
            bev = ev.get("per_benchmark", {}).get(b)
            if bev:
                bench_cols.append(f"{bev['accuracy']*100:>13.2f}%")
            else:
                bench_cols.append(f"{'N/A':>14}")
        lines.append(f"{name:<16} {overall} {abstain}   " + "  ".join(bench_cols))

    lines.append(sep)

    if len(methods) == 2:
        n1, ev1 = methods[0]
        n2, ev2 = methods[1]
        delta = (ev2["accuracy"] - ev1["accuracy"]) * 100
        lines.append(f"\nDelta Accuracy ({n2} - {n1}) = {delta:+.2f}%")
        for b in benches:
            bev1 = ev1.get("per_benchmark", {}).get(b, {})
            bev2 = ev2.get("per_benchmark", {}).get(b, {})
            if bev1 and bev2:
                bd = (bev2["accuracy"] - bev1["accuracy"]) * 100
                lines.append(f"  [{b}]  {bd:+.2f}%")

    return "\n".join(lines)
