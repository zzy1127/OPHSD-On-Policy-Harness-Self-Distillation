"""USPTO-50K data loading.

Both ``data_train_10k.json`` (sampled by ``data_prep/prepare_uspto_data.py``)
and the val parquet are normalised into the unified harness schema:
``{instruction, question, fact, accusations, label_str, ...}``.

The ``fact`` field is the reaction SMILES (used for retrieval), and
``accusations = [label_str]`` keeps the same list shape as the LawBench
harness so that ``MemoryBank`` can be reused without changes.
"""

from __future__ import annotations

import json


def load_train_data(path: str) -> list[dict]:
    """Load USPTO training JSON (output of ``prepare_uspto_data.py``).

    Each element has ``{instruction, question, rxn_smiles, prod_smiles, id,
    class, class_name, label_str}``.  We add ``fact`` and ``accusations`` so
    the entry is ready to drop into the shared :class:`MemoryBank`.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    result: list[dict] = []
    for item in data:
        entry = dict(item)
        entry.setdefault("fact", item.get("rxn_smiles", item.get("question", "")))
        entry.setdefault("accusations", [str(item["label_str"])])
        result.append(entry)
    return result


def load_test_data(path: str) -> list[dict]:
    """Load USPTO val data as a list of harness-compatible dicts.

    Accepts a ``.parquet`` file (produced by ``prepare_uspto_data.py``) or a
    JSON array in the same format as the train JSON.
    """
    if path.endswith(".parquet"):
        import pandas as pd

        df = pd.read_parquet(path)
        result: list[dict] = []
        for _, row in df.iterrows():
            extra = row.get("extra_info", {}) or {}
            rxn_smiles = extra.get("rxn_smiles", "")
            label_str = str(
                extra.get("label_str", row.get("reward_model", {}).get("ground_truth", ""))
            )
            prompt = row.get("prompt", [])
            instruction = prompt[0]["content"] if prompt else ""
            question = f"Reaction SMILES: {rxn_smiles}"
            result.append({
                "instruction": instruction,
                "question": question,
                "fact": rxn_smiles,
                "accusations": [label_str],
                "label_str": label_str,
            })
        return result
    return load_train_data(path)
