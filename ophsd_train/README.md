# `ophsd_train/` — implementation notes

This directory holds the OPHSD trainer, the per-task harnesses, and the
launcher scripts.

## Source layout

```
src/
├── ophsd/
│   ├── main_ophsd.py        # Hydra entry point
│   ├── ophsd_trainer.py     # OPHSDTrainer (subclass of verl RayPPOTrainer)
│   ├── ophsd_worker.py      # OPSDWorker (FSDP teacher/student forward + KL loss)
│   ├── batch_builder.py     # builds (teacher, student) batches from harness output
│   ├── losses.py            # KL / JSD / forward-KL implementations + entropy sampler
│   └── config/ophsd_trainer.yaml
└── rewards/
    ├── lawbench_reward.py   # set-level F1 over accusation labels
    ├── uspto_reward.py      # exact-match accuracy over reaction class
    └── math_reward.py       # math_verify-based answer equivalence
```

## Harness layout

```
harnesses/
├── _api.py                # call_api + strip_think (shared)
├── _memory_bank.py        # Embedder + online MemoryBank (shared)
├── lawbench/
│   ├── draft_verify.py    # DraftVerificationHarness (LawBench Chinese prompts)
│   ├── preprocess.py      # train / test JSON loaders
│   ├── evaluate.py        # OPTION_LIST + extract_predictions + compute_f1
│   └── config.py          # default DRAFT_K / VERIFY_CONFIRM_K / ...
├── uspto/
│   ├── draft_verify.py    # USPTODraftVerificationHarness (English prompts)
│   ├── preprocess.py
│   ├── evaluate.py        # extract_class + compute_acc + evaluate_predictions
│   └── config.py          # CLASS_NAMES + INSTRUCTION + DRAFT_K / ...
└── math/
    ├── plan_solve.py      # PlanSolveHarness + ZeroShotSolver + helpers
    ├── preprocess.py      # parquet loaders for math eval
    ├── evaluate.py        # answers_match (Hendrycks + math_verify)
    ├── run.py             # standalone CLI: vLLM zero-shot vs. harness on a parquet
    └── config.py          # API_TIMEOUT / PLAN_MAX_TOKENS / SOLVE_MAX_TOKENS
```

## Hydra config

`src/ophsd/config/ophsd_trainer.yaml` carries the trainer config.  The
harness section is **task-organised**:

```yaml
harness:
  # Shared vLLM connection + concurrency knobs
  port: 8000
  workers: 100
  ...

  # Picked up only when benchmark=math
  math:
    val_data_path: ""
    use_gt: true
    plan_max_tokens: 4096
    solve_max_tokens: 8192

  # Picked up only when benchmark=lawbench
  lawbench:
    train_data_path: ""
    embeddings_path: ""
    val_data_path: ""
    val_embeddings_path: ""
    embedding_model: "BAAI/bge-small-zh-v1.5"
    draft_k: 5
    confirm_k: 5
    challenge_k: 5
    cold_start_threshold: 10

  # Picked up only when benchmark=uspto  (same structure as lawbench)
  uspto: { ... }
```

The trainer reads `harness.{benchmark}.*` via the `_task_cfg` helper, so each
launcher script only sets the keys for its own task.

## Standalone harness evaluation (math)

`harnesses/math/run.py` lets you sanity-check the harness against any
vLLM-served checkpoint without touching the training loop:

```bash
export HARNESS_USE_LOCAL_MODEL=1
export HARNESS_LOCAL_MODEL_PATH=/path/to/Qwen3-8B
python -m harnesses.math.run \
    --data /path/to/test.parquet \
    --mode both \
    --workers 64
```

Results (zero-shot vs. harness) are appended to JSONL files under
`HARNESS_RESULTS_DIR` (default `outputs/math_harness`) and are resumable.

## Notes on the OPSD/OPHSD distillation step

* The student p_S forward is run by the actor module; the teacher forward is
  run by the **frozen** ref module in the same FSDP worker (`ophsd_worker.py`,
  `update_opsd`).
* The ref module is initialised from the same checkpoint as the actor and is
  never updated during training — this matches the OPSD paper's choice of a
  fixed-initial-policy teacher.
* The harness orchestrates calls to the *current* actor weights (live vLLM
  server), so the harness's reasoning quality co-evolves with the student
  even though the final teacher logits are computed by the frozen ref.
