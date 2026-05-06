"""LawBench Draft-Verify default hyper-parameters.

Tunable values that the trainer can override via CLI / Hydra (see
``ophsd_trainer.yaml``).  Defaults below mirror the paper experiments.
"""

DRAFT_K: int = 5              # number of similar examples shown to the draft prompt
VERIFY_CONFIRM_K: int = 5     # examples sharing the draft label, shown to the verifier
VERIFY_CHALLENGE_K: int = 5   # examples NOT sharing the draft label, shown to the verifier
COLD_START_THRESHOLD: int = 10  # below this many memory-bank entries → fall back to zero-shot
