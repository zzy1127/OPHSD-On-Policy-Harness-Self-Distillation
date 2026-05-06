"""Configuration constants for the math Plan-Solve harness.

These are *module-level mutables*: the trainer overrides them at startup based
on the run's Hydra config (``harness.math.plan_max_tokens``, etc.) before the
first harness call.  Keeping them as bare module attributes makes the override
straightforward (``import harnesses.math.config as cfg; cfg.SOLVE_MAX_TOKENS = ...``).
"""

import os

API_TIMEOUT: int = int(os.getenv("HARNESS_API_TIMEOUT", "1800"))
API_MAX_RETRIES: int = int(os.getenv("HARNESS_API_MAX_RETRIES", "1"))
API_TEMPERATURE: float = float(os.getenv("HARNESS_API_TEMPERATURE", "0.6"))

# Plan agent: thinking is enabled and stripped after the call.  4096 covers
# the typical Qwen3 plan trace; longer plans get truncated cheaply.
PLAN_MAX_TOKENS: int = int(os.getenv("HARNESS_PLAN_MAX_TOKENS", "4096"))

# Solve agent: full step-by-step solution that ends with \boxed{}.
SOLVE_MAX_TOKENS: int = int(os.getenv("HARNESS_SOLVE_MAX_TOKENS", "8192"))
