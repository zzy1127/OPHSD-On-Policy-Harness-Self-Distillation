"""Plan-Solve harness — Type II Internal Reasoning Harness for math.

Two roles, served by the *same* model with different system prompts:

  Call 1  Plan Agent   : decompose the problem into a clear, numbered plan.
  Call 2  Solve Agent  : execute the plan step-by-step and produce ``\\boxed{}``.

OPHSD teacher context  : the solve messages (with the plan + optional GT hint).
Eval answer            : ``extract_boxed(solve_response)``.

When ``use_gt=False`` the harness ignores any caller-provided answer / solution
hints, so a single switch fully neutralises GT leakage during validation.

Multi-instance dispatch
-----------------------
The harness can fan out across multiple vLLM clients via ``build_clients``;
``call_api(client=None, ...)`` will then round-robin through the registered
pool.  Single-instance setups still work — the pool just contains one entry.
"""

from __future__ import annotations

import itertools
import logging
import re
import threading
import time
from typing import Optional

from .config import (
    API_MAX_RETRIES,
    API_TEMPERATURE,
    API_TIMEOUT,
    PLAN_MAX_TOKENS,
    SOLVE_MAX_TOKENS,
)

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)


# ─────────────────────────────────────────────────────────────────────
# Round-robin client pool
# ─────────────────────────────────────────────────────────────────────

_clients: list = []
_models: list[str] = []
_rr_lock = threading.Lock()
_rr_counter = itertools.count()


def _pick_client():
    """Round-robin select ``(client, model)`` from the registered pool."""
    if not _clients:
        raise RuntimeError("call build_clients() before using harness")
    idx = next(_rr_counter) % len(_clients)
    return _clients[idx], _models[idx]


def build_clients(clients: list, models: list[str]) -> None:
    """Register the client pool used when ``call_api`` is given ``client=None``."""
    global _clients, _models
    _clients = clients
    _models = models
    logger.info("Registered %d vLLM client(s) for round-robin", len(clients))


def call_api(
    client,
    model: str,
    messages: list[dict],
    temperature: float = API_TEMPERATURE,
    max_tokens: int = SOLVE_MAX_TOKENS,
    top_p: float = 1.0,
    enable_thinking: bool = True,
) -> str:
    """Chat completion with retries and a token-limit fallback.

    If ``client`` is ``None``, auto-selects from the round-robin pool registered
    via :func:`build_clients`.  When vLLM rejects with HTTP 400 + a token-limit
    string, ``max_tokens`` is halved and the call retried.
    """
    if client is None:
        client, model = _pick_client()

    extra_body = {} if enable_thinking else {"chat_template_kwargs": {"enable_thinking": False}}

    for attempt in range(API_MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                timeout=API_TIMEOUT,
                extra_body=extra_body if extra_body else None,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            err = str(exc)
            if ("max_tokens" in err or "input_tokens" in err) and "400" in err and max_tokens > 512:
                max_tokens = max(512, max_tokens // 2)
                logger.warning("Token limit exceeded; reducing max_tokens → %d", max_tokens)
                continue
            logger.warning("API attempt %d/%d: %s", attempt + 1, API_MAX_RETRIES, exc)
            if attempt < API_MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                logger.error("API failed after %d retries", API_MAX_RETRIES)
                return ""
    return ""


def strip_think(text: str) -> str:
    """Remove closed ``<think>...</think>`` blocks (Qwen3 / R1-style).

    Unclosed blocks (caused by truncation) are intentionally left in place so
    the partial reasoning still flows into the solve prompt as context.
    """
    return _THINK_RE.sub("", text).strip()


def unwrap_think(text: str) -> str:
    """Replace ``<think>...</think>`` with its inner content (tags removed)."""
    return _THINK_RE.sub(lambda m: m.group(1).strip(), text).strip()


_BOXED_INSTR_RE = re.compile(
    r"\s*(Please reason step[- ]by[- ]step[,.].*|"
    r"[Pp]ut your final answer within\s*\\boxed\{?\}?.*)",
    re.S,
)


def strip_boxed_instruction(question: str) -> str:
    """Drop the trailing ``\\boxed{}`` instruction that every val question carries.

    Without this, the plan agent sees the directive and would emit a complete
    solution (with ``\\boxed{answer}``) instead of an outline of steps.
    """
    return _BOXED_INSTR_RE.sub("", question).rstrip()


def extract_boxed(text: str) -> Optional[str]:
    """Return the *innermost* content of the last ``\\boxed{...}`` in ``text``."""
    results: list[str] = []
    i = 0
    while i < len(text):
        pos = text.find(r"\boxed{", i)
        if pos == -1:
            break
        depth = 0
        start = pos + len(r"\boxed{")
        j = start
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                if depth == 0:
                    results.append(text[start:j].strip())
                    break
                depth -= 1
            j += 1
        i = pos + 1
    return results[-1] if results else None


# ─────────────────────────────────────────────────────────────────────
# Zero-shot baseline
# ─────────────────────────────────────────────────────────────────────

class ZeroShotSolver:
    """Single-call baseline used by ``run.py`` to compare against the harness."""

    def __init__(self, client=None, model: str = ""):
        self.client = client
        self.model = model

    def solve(self, question: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert mathematician. Solve the problem step by step "
                    "and put your final answer within \\boxed{}."
                ),
            },
            {"role": "user", "content": question},
        ]
        return call_api(self.client, self.model, messages, max_tokens=SOLVE_MAX_TOKENS)


# ─────────────────────────────────────────────────────────────────────
# Plan-Solve harness
# ─────────────────────────────────────────────────────────────────────

class PlanSolveHarness:
    """Two-phase Plan → Solve harness.

    Phase 1 (Plan)  : analyse the reference solution if available, then distil
                      it into a concise step-by-step plan.
    Phase 2 (Solve) : follow the plan and produce the full derivation, optionally
                      cross-checked against a verification-only GT hint.

    OPHSD teacher context : the solve messages (with plan + optional GT hint).
    Result keys           :
        plan_response   — Plan agent raw output.
        solve_response  — Solve agent raw output (the teacher response for OPHSD).
        final_response  — alias of ``solve_response``.
        mode            — ``"plan_solve"`` or ``"solve_only"`` when the plan
                          was empty (e.g. due to thinking-token truncation).
        trace           — per-phase ``{messages, response}`` for debugging.
    """

    n_solutions: int = 1   # required by trainer's _run_harness_inference signature

    def __init__(self, client=None, model: str = "", use_gt: bool = True):
        """Args:
            client / model : vLLM OpenAI client + model name (or ``None`` to
                use the round-robin pool from :func:`build_clients`).
            use_gt         : If False, the harness ignores any answer / solution
                provided by the caller and runs a pure no-GT plan→solve loop.
                This lets the trainer always forward the dataset GT and have a
                single switch (``harness.math.use_gt``) decide whether the
                teacher gets to peek at it.
        """
        self.client = client
        self.model = model
        self.use_gt = use_gt

    def predict_batch(
        self,
        questions: list[str],
        ground_truths: Optional[list[str]] = None,
        solutions: Optional[list[str]] = None,
        max_workers: int = 8,
    ) -> list[Optional[dict]]:
        """Parallel batch prediction.

        When ``self.use_gt`` is ``False``, ``ground_truths`` and ``solutions``
        are ignored regardless of what callers pass in.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results: list[Optional[dict]] = [None] * len(questions)

        def _one(i: int):
            if self.use_gt:
                ans = ground_truths[i] if ground_truths else ""
                sol = solutions[i] if solutions else ""
            else:
                ans, sol = "", ""
            return i, self.predict(questions[i], answer=ans, solution=sol)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_one, i) for i in range(len(questions))]
            for f in as_completed(futures):
                try:
                    idx, res = f.result()
                    results[idx] = res
                except Exception as exc:
                    logger.warning("PlanSolveHarness predict_batch item failed: %s", exc)
        return results

    def predict(self, question: str, answer: str = "", solution: str = "") -> dict:
        if not self.use_gt:
            answer, solution = "", ""

        plan_msgs = self._plan_messages(question, solution=solution)
        plan_resp = call_api(self.client, self.model, plan_msgs, max_tokens=PLAN_MAX_TOKENS)
        plan_clean = strip_think(plan_resp)

        if not plan_clean.strip():
            solve_msgs = self._solve_messages_no_plan(question)
            solve_resp = call_api(self.client, self.model, solve_msgs, max_tokens=SOLVE_MAX_TOKENS)
            return {
                "plan_response": plan_resp,
                "solve_response": solve_resp,
                "final_response": solve_resp,
                "mode": "solve_only",
                "trace": {
                    "plan": {"messages": plan_msgs, "response": plan_resp},
                    "solve": {"messages": solve_msgs, "response": solve_resp},
                },
            }

        solve_msgs = self._solve_messages(question, plan_clean, answer=answer)
        solve_resp = call_api(self.client, self.model, solve_msgs, max_tokens=SOLVE_MAX_TOKENS)

        return {
            "plan_response": plan_resp,
            "solve_response": solve_resp,
            "final_response": solve_resp,
            "mode": "plan_solve",
            "trace": {
                "plan": {"messages": plan_msgs, "response": plan_resp},
                "solve": {"messages": solve_msgs, "response": solve_resp},
            },
        }

    def _plan_messages(self, question: str, solution: str = "") -> list[dict]:
        sol_hint = (
            f"\n\nReference solution (use this to understand the approach, "
            f"then outline the key steps as a plan):\n{solution}"
            if solution else ""
        )
        user_content = (
            f"Problem:\n{question}{sol_hint}\n\n"
            + ("Analyze the reference solution and summarize it as a concise "
               "step-by-step plan." if solution else "Provide a solution plan.")
        )
        return [
            {
                "role": "system",
                "content": (
                    "You are a strategic math tutor. Your job is NOT to solve the problem, "
                    "but to outline a clear and concise solution plan: identify the key "
                    "concepts needed, list the main steps in order, and flag any potential "
                    "pitfalls. Keep the plan concise but include all the key steps, and "
                    "avoid redundant rechecking."
                ),
            },
            {"role": "user", "content": user_content},
        ]

    def _solve_messages(self, question: str, plan: str, answer: str = "") -> list[dict]:
        gt_hint = f"\n\nReference answer (for verification only): {answer}" if answer else ""
        return [
            {
                "role": "system",
                "content": (
                    "You are an expert mathematician. You have been given a solution plan. "
                    "Solve the problem following this plan. Show all working and end with "
                    "\\boxed{}."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Problem:\n{question}\n\n"
                    f"Solution plan:\n{plan}{gt_hint}\n\n"
                    "Now solve the problem following this plan. "
                    "Show all working and end with \\boxed{}."
                ),
            },
        ]

    def _solve_messages_no_plan(self, question: str) -> list[dict]:
        return [
            {
                "role": "system",
                "content": (
                    "You are an expert mathematician. Solve the problem step by step "
                    "and put your final answer within \\boxed{}."
                ),
            },
            {"role": "user", "content": question},
        ]
