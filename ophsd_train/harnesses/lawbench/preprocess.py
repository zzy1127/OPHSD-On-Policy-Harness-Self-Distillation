"""LawBench data loading.

The LawBench *3-3* (accusation prediction) task is shipped as JSON.  Each
training record carries the unified harness schema ``{instruction, question,
fact, answer, accusations, label_str}``; the official test JSON is converted
on the fly by ``load_test_data``.
"""

from __future__ import annotations

import json


def load_test_data(path: str) -> list[dict]:
    """Load LawBench 3-3 test JSON.

    Returns a list of dicts with keys
    ``{instruction, question, fact, answer, accusations, label_str}``.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    result: list[dict] = []
    for item in raw:
        question = item["question"].replace("\r\n", "\n").strip()
        fact = question[len("事实:"):].strip() if question.startswith("事实:") else question

        answer = item["answer"]
        label_str = answer.split("罪名:")[1].strip() if "罪名:" in answer else answer.strip()

        result.append(
            {
                "instruction": item["instruction"],
                "question": question,
                "fact": fact,
                "answer": answer,
                "accusations": [a.strip() for a in label_str.split(";")],
                "label_str": label_str,
            }
        )
    return result


def load_train_data(path: str) -> list[dict]:
    """Load the sampled training JSON.

    File is a JSON array; each element already follows the unified schema
    ``{instruction, question, fact, answer, accusations, label_str}``.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
