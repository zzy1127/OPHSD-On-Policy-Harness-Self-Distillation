"""Convert the DeepMath training JSON to verl parquet.

Input  : ``data/deepmath10k/data_train_10k.json`` — a JSON array where each
         entry has ``{question, answer, solution}``.  The ``question`` text
         already includes the standard ``\\boxed{}`` instruction tail.
Output : ``<out-dir>/train.parquet`` in verl's RL-dataset format.

Validation parquets are *not* generated here — supply your own DeepMath /
AIME / MATH-500 / OlympiadBench / HMMT validation parquet at training time
(via the ``VAL_FILE`` env var in ``scripts/train_ophsd_math.sh``).

Usage::

    python -m data_prep.prepare_deepmath_data \\
        --train-json data/deepmath10k/data_train_10k.json \\
        --out-dir    data/deepmath10k
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are an expert mathematician. Solve the problem step by step "
    "and put your final answer within \\boxed{}."
)


def _build_verl_row(idx: int, item: dict, data_source: str = "deepmath_10k") -> dict:
    question = str(item["question"]).strip()
    answer = str(item.get("answer", "")).strip()
    solution = str(item.get("solution", ""))

    return {
        "data_source": data_source,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": answer,
            "reference_solution": solution,
        },
        "extra_info": {
            "index": idx,
            "question": question,
            "answer": answer,
        },
        "solution": solution,
    }


def prepare_train(train_json_path: str, output_path: str) -> None:
    logger.info("Loading training data from %s ...", train_json_path)
    with open(train_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("  %d samples loaded", len(data))

    rows = [_build_verl_row(i, item) for i, item in enumerate(data)]
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Saved train parquet (%d rows) -> %s", len(df), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-json",
        default=os.path.join(os.path.dirname(__file__), os.pardir, os.pardir,
                             "data", "deepmath10k", "data_train_10k.json"),
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), os.pardir, os.pardir,
                             "data", "deepmath10k"),
    )
    args = parser.parse_args()

    prepare_train(args.train_json, os.path.join(args.out_dir, "train.parquet"))


if __name__ == "__main__":
    main()
