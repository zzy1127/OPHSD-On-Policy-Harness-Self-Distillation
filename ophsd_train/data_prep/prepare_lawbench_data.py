"""Convert the LawBench (CAIL 3-3 accusation) training JSON to verl parquet.

Input  : ``data/cail10k/data_train_10k.json``  (a JSON array; each element
         already has ``{instruction, question, fact, answer, accusations,
         label_str}``).
Output : ``<out-dir>/train.parquet``  in verl's RL-dataset format with the
         following columns:
            data_source     : str
            prompt          : list[{role, content}]
            ability         : str
            reward_model    : {style, ground_truth}
            extra_info      : dict (preserves the original fields)

Validation parquets are *not* generated here — the open-source bundle only
ships training data; users supply their own LawBench validation file.

Usage::

    python -m data_prep.prepare_lawbench_data \\
        --train-json data/cail10k/data_train_10k.json \\
        --out-dir    data/cail10k
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _build_verl_row(idx: int, item: dict, data_source: str = "lawbench_3_3") -> dict:
    instruction = item["instruction"]
    question = item["question"]
    prompt_content = f"{instruction}\n\n{question}"

    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": prompt_content}],
        "ability": "law",
        "reward_model": {
            "style": "rule",
            "ground_truth": item["label_str"],
        },
        "extra_info": {
            "index": idx,
            "instruction": instruction,
            "question": question,
            "fact": item["fact"],
            "accusations": item["accusations"],
            "label_str": item["label_str"],
        },
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
                             "data", "cail10k", "data_train_10k.json"),
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), os.pardir, os.pardir,
                             "data", "cail10k"),
    )
    args = parser.parse_args()

    prepare_train(args.train_json, os.path.join(args.out_dir, "train.parquet"))


if __name__ == "__main__":
    main()
