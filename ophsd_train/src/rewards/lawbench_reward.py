"""LawBench 3-3 (accusation prediction) reward function for verl.

Compatible with verl's custom reward function interface:
    compute_score(solution_str, ground_truth, **kwargs) -> dict

ground_truth is the semicolon-separated label string (e.g. "盗窃;诈骗").
solution_str is the raw model output text.
"""

import re

from harnesses.lawbench.evaluate import OPTION_LIST, compute_f1, extract_predictions


def compute_score(solution_str: str, ground_truth: str, **kwargs) -> dict:
    """Compute F1-based reward for accusation prediction.

    Args:
        solution_str: Raw model output (may contain [罪名]...<eoa> or free text).
        ground_truth: Semicolon-separated ground truth labels (e.g. "盗窃;诈骗").

    Returns:
        dict with keys: score (float), acc (float 0/1), pred (str).
    """
    pred_set = set(extract_predictions(solution_str))
    gt_set = {a.strip() for a in ground_truth.split(";") if a.strip()}

    f1 = compute_f1(pred_set, gt_set)

    return {
        "score": f1,
        "acc": float(f1 > 0),
        "pred": ";".join(sorted(pred_set)),
    }
