"""Reasoning harnesses for OPHSD, organised by task.

Public sub-packages:
  - ``harnesses.lawbench``  → Draft-Verify harness for the LawBench accusation task.
  - ``harnesses.uspto``     → Draft-Verify harness for USPTO-50K reaction classification.
  - ``harnesses.math``      → Plan-Solve harness for DeepMath problems.

Shared infrastructure:
  - ``harnesses._api``         → vLLM chat API helpers (``call_api``, ``strip_think``).
  - ``harnesses._memory_bank`` → ``Embedder`` + online ``MemoryBank``
                                 used by both Draft-Verify variants.
"""
