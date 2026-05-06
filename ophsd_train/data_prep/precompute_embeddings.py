"""Pre-compute training-set embeddings for the Draft-Verify harnesses.

Cosine-similarity retrieval inside ``DraftVerificationHarness`` uses
sentence-transformer embeddings of the ``fact`` field (lawbench) or
``rxn_smiles`` field (uspto).  Computing them once up front lets the trainer
seed the memory bank instantly instead of paying the embedding cost on every
restart.

Usage::

    # LawBench
    python -m data_prep.precompute_embeddings \\
        --task lawbench \\
        --train-json data/cail10k/data_train_10k.json \\
        --output     data/cail10k/train_embeddings.npy

    # USPTO
    python -m data_prep.precompute_embeddings \\
        --task uspto \\
        --train-json data/uspto10k/data_train_10k.json \\
        --output     data/uspto10k/train_embeddings.npy

The output ``.npy`` is a ``(N, D)`` float32 matrix aligned with the JSON's
row order, ready to plug into ``HARNESS_TRAIN_EMBEDDINGS`` in the launcher
scripts.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np

# Make ``harnesses`` importable when this file is run directly from the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from harnesses._memory_bank import Embedder  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _extract_text(task: str, item: dict) -> str:
    """Pick the field used for retrieval, per task."""
    if task == "lawbench":
        return item.get("fact") or item.get("question") or ""
    if task == "uspto":
        return item.get("rxn_smiles") or item.get("question") or ""
    raise ValueError(f"Unknown task: {task!r}; expected 'lawbench' or 'uspto'.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["lawbench", "uspto"], required=True)
    parser.add_argument("--train-json", required=True,
                        help="Path to data_train_10k.json")
    parser.add_argument("--output", required=True,
                        help="Output .npy path for the (N, D) embedding matrix")
    parser.add_argument("--embedding-model",
                        default="BAAI/bge-small-zh-v1.5",
                        help="sentence-transformers model id (default: bge-small-zh-v1.5)")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    logger.info("Loading %s ...", args.train_json)
    with open(args.train_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("  %d records", len(data))

    texts = [_extract_text(args.task, item) for item in data]

    logger.info("Loading embedder %s ...", args.embedding_model)
    embedder = Embedder(model_name=args.embedding_model)
    if not texts:
        logger.warning("No text to embed; nothing written.")
        return

    logger.info("Encoding %d texts ...", len(texts))
    embeddings = embedder.encode_queries(texts).astype(np.float32)
    logger.info("Embeddings shape=%s dtype=%s", embeddings.shape, embeddings.dtype)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.save(args.output, embeddings)
    logger.info("Saved -> %s", args.output)


if __name__ == "__main__":
    main()
