"""USPTO-50K reaction classification reward function for verl.

Compatible with verl's custom reward function interface:
    compute_score(solution_str, ground_truth, **kwargs) -> dict

ground_truth is the string class number (e.g. "3").
solution_str is the raw model output text.
"""

import re


_THINK_RE = re.compile(r"<think>.*?</think>", re.S)

VALID_CLASSES = {str(i) for i in range(1, 11)}


def extract_class(solution_str: str) -> str:
    """Extract predicted class number from model output.

    Tries ``[class]N<eoa>`` format first, then falls back to bare digit scan.
    Always strips <think> blocks before parsing.
    """
    clean = _THINK_RE.sub("", solution_str).strip()

    # Primary: structured format
    m = re.search(r"\[class\]\s*(\d+)\s*(?:<eoa>|$)", clean, re.S)
    if m:
        return m.group(1).strip()

    # Fallback: first valid class number found in output
    m = re.search(r"\b(10|[1-9])\b", clean)
    if m:
        return m.group(1)

    return ""


def compute_score(solution_str: str, ground_truth: str, **kwargs) -> dict:
    """Compute accuracy-based reward for USPTO reaction classification.

    Args:
        solution_str: Raw model output.
        ground_truth: String class number, e.g. "3".

    Returns:
        dict with keys: score (float 0/1), acc (float 0/1), pred (str).
    """
    pred = extract_class(solution_str)
    gt = str(ground_truth).strip()

    correct = float(pred == gt and pred in VALID_CLASSES)

    return {
        "score": correct,
        "acc": correct,
        "pred": pred,
    }
