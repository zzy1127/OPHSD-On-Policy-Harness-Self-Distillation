# OPHSD: On-Policy Harness Self-Distillation

Official implementation of OPHSD: A framework that internalizes inference-time harnesses into the base model via on-policy self-distillation.

The base RL trainer is built on
[OPD](https://github.com/HJSang/OPSD_OnPolicyDistillation).

**Git Large File Storage:** `data/deepmath10k/data_train_10k.json` is tracked with [Git LFS](https://git-lfs.github.com/) because it exceeds GitHub’s 100&nbsp;MB file limit as plain Git. Install `git-lfs`, run `git lfs install` once, then clone as usual (`git clone …`) — LFS will fetch the real JSON automatically.

## Repository layout

```
OPHSD_opensource/
├── data/                                 # Training data (≤10 k samples each)
│   ├── deepmath10k/data_train_10k.json   # DeepMath, schema {question, answer, solution}
│   ├── cail10k/data_train_10k.json       # LawBench / CAIL 3-3, schema with `accusations`
│   └── uspto10k/data_train_10k.json      # USPTO-50K reaction classification
└── ophsd_train/
    ├── src/
    │   ├── ophsd/                        # OPHSD trainer + worker (Hydra config)
    │   └── rewards/                      # task-specific reward functions
    ├── harnesses/                        # one self-contained sub-package per task
    │   ├── lawbench/                     #   draft-verify (Chinese accusations)
    │   ├── uspto/                        #   draft-verify (English reactions)
    │   ├── math/                         #   plan-solve (mathematical reasoning)
    │   ├── _api.py                       # shared chat-completion helper
    │   └── _memory_bank.py               # online memory bank + Embedder
    ├── data_prep/                        # JSON → verl parquet converters
    ├── scripts/                          # one launcher per benchmark
    │   ├── train_ophsd_math.sh
    │   ├── train_ophsd_lawbench.sh
    │   └── train_ophsd_uspto.sh
    └── README.md                         # implementation notes
```

## Quick start

### 1. Install

```bash
pip install -r requirements.txt   # see ophsd_train/README.md for the full list
pip install -e <path-to-verl>     # required, OPHSD subclasses verl PPO trainer
```

### 2. Convert the training data to verl parquet

```bash
cd ophsd_train

python -m data_prep.prepare_deepmath_data
python -m data_prep.prepare_lawbench_data
python -m data_prep.prepare_uspto_data
```

Each command writes a `train.parquet` next to its source JSON.

### 3. (Lawbench / USPTO only) pre-compute training embeddings

The Draft-Verify harness retrieves nearest-neighbour examples from an online
`MemoryBank`.  Pre-computing embeddings for the 10 k training records lets
the trainer skip the cold-start cost on every restart:

```bash
python -m data_prep.precompute_embeddings \
    --task lawbench \
    --train-json data/cail10k/data_train_10k.json \
    --output     data/cail10k/train_embeddings.npy
```

### 4. Train

```bash
# Math (Plan-Solve harness) — see scripts/train_ophsd_math.sh for all knobs.
MODEL_PATH=/path/to/Qwen3-8B \
TRAIN_FILE=data/deepmath10k/train.parquet \
VAL_FILE=/path/to/your/math_val.parquet \
bash ophsd_train/scripts/train_ophsd_math.sh

# LawBench (Draft-Verify harness)
MODEL_PATH=/path/to/Qwen3-8B \
TRAIN_FILE=data/cail10k/train.parquet \
VAL_FILE=/path/to/lawbench_val.parquet \
HARNESS_TRAIN_DATA=data/cail10k/data_train_10k.json \
HARNESS_VAL_DATA=/path/to/lawbench_val.json \
HARNESS_TRAIN_EMBEDDINGS=data/cail10k/train_embeddings.npy \
bash ophsd_train/scripts/train_ophsd_lawbench.sh

# USPTO-50K (Draft-Verify harness, English)
MODEL_PATH=/path/to/Qwen3-8B \
TRAIN_FILE=data/uspto10k/train.parquet \
VAL_FILE=/path/to/uspto_val.parquet \
HARNESS_TRAIN_DATA=data/uspto10k/data_train_10k.json \
HARNESS_VAL_DATA=/path/to/uspto_val.parquet \
HARNESS_TRAIN_EMBEDDINGS=data/uspto10k/train_embeddings.npy \
bash ophsd_train/scripts/train_ophsd_uspto.sh
```

## Harness organisation

Each task has its own self-contained sub-package under `harnesses/`:

| Task        | Harness                              | Module                                            |
|-------------|--------------------------------------|---------------------------------------------------|
| LawBench    | `DraftVerificationHarness`           | `harnesses.lawbench.draft_verify`                 |
| USPTO-50K   | `USPTODraftVerificationHarness`      | `harnesses.uspto.draft_verify`                    |
| DeepMath    | `PlanSolveHarness`                   | `harnesses.math.plan_solve`                       |

Shared infrastructure (`call_api`, `MemoryBank`, `Embedder`) lives at the
`harnesses/` package root.  Hydra config is split the same way: shared knobs
(vLLM connection, concurrency) sit at the top of `harness:`, while per-task
knobs live under `harness.{lawbench,uspto,math}.*` and are picked up
automatically based on the run's `benchmark` value.
