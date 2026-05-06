"""Draft-Verify harness for the LawBench 3-3 accusation task.

Two-phase pipeline:
    1. Draft   — retrieve ``draft_k`` similar precedents and ask the model for an
                 initial accusation list.
    2. Verify  — retrieve ``confirm_k`` precedents that agree with the draft and
                 ``challenge_k`` similar precedents that disagree, then ask the
                 model to validate or revise its draft.

The harness is *online*: ``MemoryBank`` grows incrementally during evaluation /
training, so retrieval improves as more labelled examples accumulate.

Below the ``COLD_START_THRESHOLD`` (default 10), the harness short-circuits to a
plain zero-shot call so that early predictions are not poisoned by an empty
memory bank.
"""

from __future__ import annotations

import logging
import re
from typing import Callable, Optional

import numpy as np

from .._api import call_api, strip_similarity, strip_think
from .._memory_bank import MemoryBank
from .config import COLD_START_THRESHOLD, DRAFT_K, VERIFY_CHALLENGE_K, VERIFY_CONFIRM_K
from .evaluate import OPTION_LIST

logger = logging.getLogger(__name__)


def _extract_labels(response: str) -> list[str]:
    """Parse model output → list of accusation labels.

    Tries the structured ``[罪名]...<eoa>`` block first.  Falls back to scanning
    the cleaned text for known options.  ``<think>`` traces are stripped first
    so reasoning content does not pollute label extraction.
    """
    clean = strip_think(response)
    m = re.search(r"\[罪名\](.*?)(?:<eoa>|$)", clean, re.S)
    if m:
        raw = m.group(1).strip()
        labels = [l.strip() for l in raw.split(";") if l.strip()]
        if labels:
            return labels
    return [opt for opt in OPTION_LIST if opt in clean]


class DraftVerificationHarness:
    """LawBench Draft-Verify harness.

    Phase 1 (Draft)  : retrieve ``draft_k`` similar examples → LLM call → draft labels.
    Phase 2 (Verify) : retrieve confirmers / challengers → LLM call → final labels.
    """

    def __init__(
        self,
        client,
        model: str,
        memory_bank: MemoryBank,
        embed_fn: Callable[[str], np.ndarray],
        draft_k: int = DRAFT_K,
        confirm_k: int = VERIFY_CONFIRM_K,
        challenge_k: int = VERIFY_CHALLENGE_K,
        cold_start_threshold: int = COLD_START_THRESHOLD,
    ):
        self.client = client
        self.model = model
        self.memory = memory_bank
        self.embed_fn = embed_fn
        self.draft_k = draft_k
        self.confirm_k = confirm_k
        self.challenge_k = challenge_k
        self.cold_start = cold_start_threshold

    def predict(
        self,
        instruction: str,
        question: str,
        fact: str,
        precomputed_emb: "np.ndarray | None" = None,
    ) -> dict:
        """Run the full Draft-Verify pipeline on one sample.

        Returns a dict with ``draft_response``, ``final_response``, ``mode`` and a
        ``trace`` sub-dict carrying per-phase retrieval results / messages.
        """
        if len(self.memory.examples) < self.cold_start:
            msgs = [{"role": "user", "content": f"{instruction}\n\n{question}"}]
            resp = call_api(self.client, self.model, msgs)
            return {
                "draft_response": resp,
                "final_response": resp,
                "mode": "cold_start",
                "trace": {"cold_start": {"messages": msgs, "response": resp}},
            }

        query_emb = precomputed_emb if precomputed_emb is not None else self.embed_fn(fact)

        similar = self.memory.retrieve(query_emb, k=self.draft_k)
        draft_msgs = self._draft_messages(question, similar)
        draft_resp = call_api(self.client, self.model, draft_msgs)
        draft_labels = _extract_labels(draft_resp)

        if not draft_labels:
            return {
                "draft_response": draft_resp,
                "final_response": draft_resp,
                "draft_labels": [],
                "mode": "draft_only",
                "query_emb": query_emb,
                "trace": {
                    "draft": {
                        "retrieved": strip_similarity(similar),
                        "messages": draft_msgs,
                        "response": draft_resp,
                        "extracted_labels": [],
                    },
                },
            }

        confirmers = self.memory.retrieve(
            query_emb, k=self.confirm_k, filter_labels=draft_labels, same_label=True,
        )
        challengers = self.memory.retrieve(
            query_emb, k=self.challenge_k, filter_labels=draft_labels, same_label=False,
        )
        verify_msgs = self._verify_messages(question, draft_labels, confirmers, challengers)
        final_resp = call_api(self.client, self.model, verify_msgs)

        return {
            "draft_response": draft_resp,
            "final_response": final_resp,
            "draft_labels": draft_labels,
            "mode": "full",
            "query_emb": query_emb,
            "trace": {
                "draft": {
                    "retrieved": strip_similarity(similar),
                    "messages": draft_msgs,
                    "response": draft_resp,
                    "extracted_labels": draft_labels,
                },
                "verify": {
                    "confirmers": strip_similarity(confirmers),
                    "challengers": strip_similarity(challengers),
                    "messages": verify_msgs,
                    "response": final_resp,
                },
            },
        }

    def _draft_messages(self, question: str, examples: list[dict]) -> list[dict]:
        ex_text = ""
        for ex in examples:
            ex_text += f"· 若事实包含[{ex['fact']}]，则定性为[{ex['label_str']}]\n"

        user = (
            "以下是类似案件的裁判经验供你参考：\n\n"
            f"{ex_text}\n"
            f"现在请你判断以下案件的罪名：\n{question}\n\n"
            "请将答案写在[罪名]和<eoa>之间。"
            "例如[罪名]盗窃;诈骗<eoa>。请你严格按照这个格式回答。"
        )
        return [
            {"role": "system", "content": "你是一位资深法官，擅长根据案件事实判断罪名。"},
            {"role": "user", "content": user},
        ]

    def _verify_messages(
        self,
        question: str,
        draft_labels: list[str],
        confirmers: list[dict],
        challengers: list[dict],
    ) -> list[dict]:
        draft_str = ";".join(draft_labels)

        confirm_text = ""
        if confirmers:
            confirm_text = "支持该罪名的类似判例：\n"
            for ex in confirmers:
                confirm_text += f"· 事实：{ex['fact']}　→　罪名：{ex['label_str']}\n"
            confirm_text += "\n"

        challenge_text = ""
        if challengers:
            challenge_text = "情况相似但罪名不同的判例：\n"
            for ex in challengers:
                challenge_text += f"· 事实：{ex['fact']}　→　罪名：{ex['label_str']}\n"
            challenge_text += "\n"

        user = (
            f"你之前对以下案件做出了初步判断：\n{question}\n\n"
            f"初步判断罪名：{draft_str}\n\n"
            f"{confirm_text}"
            f"{challenge_text}"
            "请综合考虑上述正反两方面的证据，对初步判断进行验证。\n"
            "如果你认为初步判断正确，请维持原判；如果有误，请修正。\n\n"
            "请将最终答案写在[罪名]和<eoa>之间。"
            "例如[罪名]盗窃;诈骗<eoa>。请你严格按照这个格式回答。"
        )
        return [
            {"role": "system", "content": "你是一位资深法官，需要对罪名的初步判断进行审核验证。"},
            {"role": "user", "content": user},
        ]
