"""Draft-Verify harness for USPTO-50K reaction-type classification.

English-prompt counterpart to ``harnesses.lawbench.draft_verify``.  Labels are
single integers (1..10) instead of multi-label accusation sets, and the prompt
templates are written in English; otherwise the two harnesses share the same
structure (cold-start zero-shot, draft, verify) and the same backing
:class:`MemoryBank` storage.
"""

from __future__ import annotations

import logging
import re
from typing import Callable, Optional

import numpy as np

from .._api import call_api, strip_similarity, strip_think
from .._memory_bank import MemoryBank
from .config import (
    CLASS_NAMES,
    COLD_START_THRESHOLD,
    DRAFT_K,
    VERIFY_CHALLENGE_K,
    VERIFY_CONFIRM_K,
)

logger = logging.getLogger(__name__)


def extract_class(response: str) -> Optional[str]:
    """Parse model output → class number string (``"3"``) or ``None``.

    Tries the structured ``[class]N<eoa>`` block first, then falls back to a
    bare digit 1-10 scan in the cleaned text.
    """
    clean = strip_think(response)
    m = re.search(r"\[class\]\s*(\d+)\s*(?:<eoa>|$)", clean, re.S)
    if m:
        return m.group(1).strip()
    m = re.search(r"\b(10|[1-9])\b", clean)
    if m:
        return m.group(1)
    return None


class USPTODraftVerificationHarness:
    """Two-phase Draft-Verify harness for USPTO-50K.

    Memory-bank entries use:
      - ``fact``        → ``rxn_smiles``  (used for cosine-similarity retrieval)
      - ``label_str``   → string class number, e.g. ``"3"``
      - ``accusations`` → ``[label_str]``  (single-element list for MemoryBank
        compatibility with the LawBench code path)
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

        ``fact`` is the reaction SMILES and is used for embedding / retrieval.
        Returns ``{draft_response, final_response, mode, trace, ...}``.
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
        draft_class = extract_class(draft_resp)

        if not draft_class:
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
                        "extracted_class": None,
                    },
                },
            }

        confirmers = self.memory.retrieve(
            query_emb, k=self.confirm_k, filter_labels=[draft_class], same_label=True,
        )
        challengers = self.memory.retrieve(
            query_emb, k=self.challenge_k, filter_labels=[draft_class], same_label=False,
        )
        verify_msgs = self._verify_messages(question, draft_class, confirmers, challengers)
        final_resp = call_api(self.client, self.model, verify_msgs)

        return {
            "draft_response": draft_resp,
            "final_response": final_resp,
            "draft_labels": [draft_class],
            "mode": "full",
            "query_emb": query_emb,
            "trace": {
                "draft": {
                    "retrieved": strip_similarity(similar),
                    "messages": draft_msgs,
                    "response": draft_resp,
                    "extracted_class": draft_class,
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
            cls = ex.get("label_str", "?")
            cls_name = CLASS_NAMES.get(int(cls), cls) if str(cls).isdigit() else cls
            rxn = ex.get("fact", ex.get("rxn_smiles", ""))
            ex_text += f"· Reaction: [{rxn}]  →  class [{cls}] ({cls_name})\n"

        user = (
            "Here are similar reactions from past examples for reference:\n\n"
            f"{ex_text}\n"
            f"Now classify the following reaction:\n{question}\n\n"
            "Output only the class number inside [class] and <eoa> tags. "
            "For example: [class]3<eoa>"
        )
        return [
            {
                "role": "system",
                "content": "You are an expert organic chemist specializing in reaction classification.",
            },
            {"role": "user", "content": user},
        ]

    def _verify_messages(
        self,
        question: str,
        draft_class: str,
        confirmers: list[dict],
        challengers: list[dict],
    ) -> list[dict]:
        draft_name = CLASS_NAMES.get(int(draft_class), draft_class) if str(draft_class).isdigit() else draft_class

        confirm_text = ""
        if confirmers:
            confirm_text = f"Reactions also classified as class {draft_class} ({draft_name}):\n"
            for ex in confirmers:
                rxn = ex.get("fact", ex.get("rxn_smiles", ""))
                confirm_text += f"· {rxn}  →  class {ex['label_str']}\n"
            confirm_text += "\n"

        challenge_text = ""
        if challengers:
            challenge_text = "Similar reactions but with different classes:\n"
            for ex in challengers:
                rxn = ex.get("fact", ex.get("rxn_smiles", ""))
                cls = ex.get("label_str", "?")
                cls_name = CLASS_NAMES.get(int(cls), cls) if str(cls).isdigit() else cls
                challenge_text += f"· {rxn}  →  class {cls} ({cls_name})\n"
            challenge_text += "\n"

        user = (
            f"You previously classified the following reaction as class {draft_class} ({draft_name}):\n"
            f"{question}\n\n"
            f"{confirm_text}"
            f"{challenge_text}"
            "Please review your draft classification considering the supporting and contrasting examples above. "
            "Maintain your answer if correct, or revise if needed.\n\n"
            "Output only the final class number inside [class] and <eoa> tags. "
            "For example: [class]3<eoa>"
        )
        return [
            {
                "role": "system",
                "content": "You are an expert organic chemist verifying a reaction classification.",
            },
            {"role": "user", "content": user},
        ]
