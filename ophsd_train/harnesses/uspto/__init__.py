"""USPTO-50K Draft-Verify harness package.

Public API:
    USPTODraftVerificationHarness — two-phase reaction-type classifier.
    CLASS_NAMES                   — int → class name map (1..10).
    INSTRUCTION                   — canonical user-facing prompt prefix.
    extract_class                 — parse model output → ``"3"`` / ``None``.
    compute_acc                   — accuracy of predictions vs ground truth.
    load_train_data               — JSON loader for sampled train data.
    load_test_data                — parquet/JSON loader for the val split.
"""

from .config import CLASS_NAMES, INSTRUCTION
from .draft_verify import USPTODraftVerificationHarness, extract_class
from .evaluate import compute_acc, evaluate_predictions
from .preprocess import load_test_data, load_train_data

__all__ = [
    "USPTODraftVerificationHarness",
    "CLASS_NAMES",
    "INSTRUCTION",
    "extract_class",
    "compute_acc",
    "evaluate_predictions",
    "load_test_data",
    "load_train_data",
]
