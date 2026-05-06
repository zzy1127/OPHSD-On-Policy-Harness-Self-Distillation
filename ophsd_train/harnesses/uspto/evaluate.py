"""Evaluation utilities for USPTO-50K reaction-type classification.

Metric: top-1 accuracy of the predicted class string against ground truth.
"""

from __future__ import annotations

from .draft_verify import extract_class


def compute_acc(pred: str | None, gt: str) -> float:
    """1.0 if ``pred`` parses to the same class number as ``gt``, else 0.0."""
    if pred is None:
        return 0.0
    return 1.0 if str(pred).strip() == str(gt).strip() else 0.0


def evaluate_predictions(
    predictions: list[str],
    ground_truths: list[dict],
) -> dict:
    """Evaluate raw model outputs against ground truth ``label_str``.

    Returns ``{accuracy, abstention_rate, total, per_sample}``.
    """
    scores: list[float] = []
    abstentions = 0
    per_sample: list[dict] = []

    for pred_text, gt in zip(predictions, ground_truths):
        pred_class = extract_class(pred_text)
        gt_class = str(gt.get("label_str", gt.get("accusations", [""])[0]))
        if pred_class is None:
            abstentions += 1
        score = compute_acc(pred_class, gt_class)
        scores.append(score)
        per_sample.append({"pred": pred_class, "gt": gt_class, "correct": score == 1.0})

    return {
        "accuracy": sum(scores) / max(len(scores), 1),
        "abstention_rate": abstentions / max(len(predictions), 1),
        "total": len(predictions),
        "per_sample": per_sample,
    }
