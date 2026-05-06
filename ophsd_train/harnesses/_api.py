"""Shared vLLM chat-completion helpers used by every task harness.

All harnesses talk to a *colocated* vLLM server (the same one that serves the
on-policy student rollout during OPHSD training).  The HTTP path is therefore
local-only and fast; the helpers here just wrap retry / context-overflow logic
so that individual harnesses do not have to reinvent it.
"""

from __future__ import annotations

import logging
import re
import time

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT: int = 300
DEFAULT_MAX_RETRIES: int = 3
DEFAULT_TEMPERATURE: float = 0.1
DEFAULT_MAX_TOKENS: int = 8192   # Qwen3 thinking traces are long, give a bit of room


def call_api(
    client,
    model: str,
    messages: list[dict],
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    **extra_kwargs,
) -> str:
    """Call ``client.chat.completions.create`` with retries.

    Special-case: when vLLM rejects with HTTP 400 + ``input_tokens`` (request
    exceeds ``max_model_len``), halve ``max_tokens`` and try again immediately.
    Returns the decoded text or ``""`` on persistent failure.
    """
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                **extra_kwargs,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            err_str = str(exc)
            if "input_tokens" in err_str and "400" in err_str and max_tokens > 512:
                new_max = max(512, max_tokens // 2)
                logger.warning(
                    "Context length exceeded (max_tokens=%d → %d); retrying",
                    max_tokens, new_max,
                )
                max_tokens = new_max
                continue
            logger.warning("API attempt %d/%d failed: %s", attempt + 1, max_retries, exc)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                logger.error("API call failed after %d retries", max_retries)
                return ""
    return ""


_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def strip_think(text: str) -> str:
    """Remove ``<think>...</think>`` reasoning blocks (Qwen3 / R1-style)."""
    return _THINK_RE.sub("", text).strip()


def strip_similarity(examples: list[dict]) -> list[dict]:
    """Cast similarity scores to plain ``float`` so result dicts are JSON-safe."""
    return [{**ex, "similarity": float(ex.get("similarity", 0.0))} for ex in examples]
