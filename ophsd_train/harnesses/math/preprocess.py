"""Load and normalise math datasets (parquet format)."""

import ast
import json
import logging
from collections import Counter, defaultdict
from typing import Optional

logger = logging.getLogger(__name__)


def _parse_prompt(prompt_raw) -> str:
    """Extract the user-facing question text from the prompt field.

    The parquet stores prompt as a list of dicts [{role, content}, ...].
    It may arrive as a Python list (pyarrow already parsed) or as a string
    representation that needs eval/json.loads.
    """
    if isinstance(prompt_raw, list):
        msgs = prompt_raw
    elif isinstance(prompt_raw, str):
        try:
            msgs = json.loads(prompt_raw)
        except json.JSONDecodeError:
            try:
                msgs = ast.literal_eval(prompt_raw)
            except Exception:
                return prompt_raw.strip()
    else:
        return str(prompt_raw).strip()

    for msg in msgs:
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg["content"].strip()
    # fallback: concatenate all content
    return " ".join(m.get("content", "") for m in msgs if isinstance(m, dict)).strip()


def _parse_reward_model(rm_raw) -> dict:
    if isinstance(rm_raw, dict):
        return rm_raw
    if isinstance(rm_raw, str):
        try:
            return json.loads(rm_raw)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(rm_raw)
            except Exception:
                pass
    return {}


def load_test_data(
    path: str,
    max_samples: int = 0,
    shuffle: bool = False,
    seed: int = 42,
    benchmark_filter: Optional[list[str]] = None,
    per_benchmark_max: int = 0,
) -> list[dict]:
    """Load test parquet into a list of normalised dicts.

    Each dict has:
        question      : str   — the problem text
        ground_truth  : str   — expected answer (from reward_model.ground_truth)
        question_id   : str
        data_source   : str
        level         : str | None
    """
    import pyarrow.parquet as pq
    import random

    table = pq.read_table(path)
    all_n = len(table)

    # Load all records first, then shuffle, then slice —
    # so --max-test N samples spread across all benchmarks.
    records = []
    for i in range(all_n):
        row = {col: table[col][i].as_py() for col in table.schema.names}
        rm = _parse_reward_model(row.get("reward_model", {}))
        records.append({
            "question":     _parse_prompt(row.get("prompt", "")),
            "ground_truth": str(rm.get("ground_truth", "")).strip(),
            "question_id":  str(row.get("question_id", i)),
            "data_source":  str(row.get("data_source", "")),
            "level":        str(row.get("level", "")) if row.get("level") not in (None, "None") else None,
        })

    if benchmark_filter:
        allowed = {x.strip() for x in benchmark_filter if x.strip()}
        records = [r for r in records if r.get("data_source", "") in allowed]
        logger.info("Filtered by benchmarks=%s -> %d samples", sorted(allowed), len(records))

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(records)

    if per_benchmark_max > 0:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for rec in records:
            grouped[rec.get("data_source", "unknown")].append(rec)
        limited = []
        for source, items in grouped.items():
            limited.extend(items[:per_benchmark_max])
        records = limited
        logger.info(
            "Applied per_benchmark_max=%d -> %d samples",
            per_benchmark_max,
            len(records),
        )

    if max_samples > 0:
        records = records[:max_samples]

    dist = Counter(r.get("data_source", "unknown") for r in records)
    logger.info("Loaded %d test samples from %s (shuffle=%s)", len(records), path, shuffle)
    if dist:
        logger.info("Benchmark distribution: %s", dict(sorted(dist.items())))
    return records


def load_train_data(path: str, max_samples: int = 0) -> list[dict]:
    """Load training parquet.  Used for potential few-shot seeding (not needed for Type II).

    Each dict has the same keys as load_test_data plus:
        solution      : str   — reference solution text
    """
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    n = len(table)
    if max_samples > 0:
        n = min(n, max_samples)

    records = []
    for i in range(n):
        row = {col: table[col][i].as_py() for col in table.schema.names}
        rm = _parse_reward_model(row.get("reward_model", {}))
        records.append({
            "question":     _parse_prompt(row.get("prompt", "")),
            "ground_truth": str(rm.get("ground_truth", "")).strip(),
            "solution":     str(row.get("solution", "")),
            "question_id":  str(row.get("extra_info", {}).get("index", i) if isinstance(row.get("extra_info"), dict) else i),
            "data_source":  str(row.get("data_source", "")),
        })

    logger.info("Loaded %d train samples from %s", len(records), path)
    return records
