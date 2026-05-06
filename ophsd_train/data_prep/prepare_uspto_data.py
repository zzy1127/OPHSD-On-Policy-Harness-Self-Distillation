"""Convert the USPTO-50K training JSON to verl parquet.

Input  : ``data/uspto10k/data_train_10k.json`` — a JSON array where each entry
         carries ``{instruction, question, rxn_smiles, prod_smiles, id, class,
         class_name, label_str}``.
Output : ``<out-dir>/train.parquet`` with the columns expected by verl
         (``data_source``, ``prompt``, ``ability``, ``reward_model``, ``extra_info``).

Validation parquets are *not* generated here — supply your own when training.

Usage::

    python -m data_prep.prepare_uspto_data \\
        --train-json data/uspto10k/data_train_10k.json \\
        --out-dir    data/uspto10k
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


CLASS_NAMES = {
    1:  "heteroatom alkylation and arylation",
    2:  "acylation and related processes",
    3:  "C-C bond formation",
    4:  "heterocycle formation",
    5:  "protections",
    6:  "deprotections",
    7:  "reductions",
    8:  "oxidations",
    9:  "functional group interconversion (FGI)",
    10: "functional group addition (FGA)",
}


def _build_verl_row(idx: int, item: dict, data_source: str = "uspto50k") -> dict:
    rxn_smiles = item["rxn_smiles"]
    cls = int(item["class"])
    label_str = str(cls)
    prompt_content = f"{item['instruction']}\n\nReaction SMILES: {rxn_smiles}"

    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": prompt_content}],
        "ability": "chemistry",
        "reward_model": {
            "style": "rule",
            "ground_truth": label_str,
        },
        "extra_info": {
            "index": idx,
            "id": item.get("id", ""),
            "rxn_smiles": rxn_smiles,
            "prod_smiles": item.get("prod_smiles", ""),
            "class": cls,
            "class_name": item.get("class_name", CLASS_NAMES.get(cls, "")),
            "label_str": label_str,
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
                             "data", "uspto10k", "data_train_10k.json"),
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), os.pardir, os.pardir,
                             "data", "uspto10k"),
    )
    args = parser.parse_args()

    prepare_train(args.train_json, os.path.join(args.out_dir, "train.parquet"))


if __name__ == "__main__":
    main()
