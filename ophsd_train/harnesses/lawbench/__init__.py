"""LawBench (CAIL) Draft-Verify harness package.

Public API:
    DraftVerificationHarness  — two-phase (draft + verify) accusation classifier.
    OPTION_LIST               — canonical 200+ accusation labels.
    extract_predictions       — parse model output → list[label].
    compute_f1                — set-level F1 of predictions vs ground truth.
    load_train_data           — JSON loader for sampled train data.
    load_test_data            — JSON loader for the LawBench 3-3 test split.
"""

from .draft_verify import DraftVerificationHarness
from .evaluate import OPTION_LIST, compute_f1, evaluate_predictions, extract_predictions
from .preprocess import load_test_data, load_train_data

__all__ = [
    "DraftVerificationHarness",
    "OPTION_LIST",
    "compute_f1",
    "evaluate_predictions",
    "extract_predictions",
    "load_test_data",
    "load_train_data",
]
