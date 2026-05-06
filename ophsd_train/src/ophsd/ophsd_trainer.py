"""
OPHSD trainer: On-Policy Harness Self-Distillation.

Extends the OPSD trainer by replacing ground-truth teacher context with
full harness pipeline (draft + verify) outputs.  The harness uses the same
vLLM server that serves the student rollout, so harness inference runs
BEFORE the engine sleeps.

Training loop per step:
  1. Student on-policy rollout via vLLM
  2. Harness teacher inference via vLLM (draft + verify)
  3. vLLM sleeps
  4. Build OPHSD batch (harness output as teacher context)
  5. FSDP forward teacher + student → reverse KL → backward → step
  6. Update memory bank with batch samples (true labels)
  7. vLLM wakes up with updated actor weights
"""

import json
import logging
import os
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from openai import OpenAI
from torch.utils.data import Dataset
from tqdm import tqdm

from verl.protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role, compute_response_mask
from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils.metric import reduce_metrics

from .batch_builder import build_ophsd_batch_from_verl_batch

# Shared harness infrastructure
from harnesses._memory_bank import MemoryBank, Embedder

# Per-task harness modules (one harness per benchmark)
from harnesses.lawbench import (
    DraftVerificationHarness,
    load_test_data as load_lawbench_test_data,
    load_train_data as load_lawbench_train_data,
)
from harnesses.uspto import (
    USPTODraftVerificationHarness,
    load_test_data as load_uspto_test_data,
    load_train_data as load_uspto_train_data,
)
from harnesses.math import PlanSolveHarness


def _load_math_data(path: str) -> list[dict]:
    """Load math harness data from JSON file.

    Supports two formats:
      - List of {"question": ..., "answer": ...}  (harness JSON from prepare_math_data.py)
      - Parquet file (converted on-the-fly)

    Each returned dict has keys: question, answer, solution (may be empty string).
    """
    import json as _json
    if not path or not os.path.exists(path):
        return []
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            items = _json.load(f)
        return [{"question": it["question"], "answer": it.get("answer", ""),
                 "solution": it.get("solution", "")} for it in items]
    if path.endswith(".parquet"):
        try:
            import pyarrow.parquet as pq
            df = pq.read_table(path).to_pandas()
        except ImportError:
            import pandas as pd
            df = pd.read_parquet(path)
        # verl format: question in extra_info["question"] or prompt list, answer in reward_model["ground_truth"]
        if "extra_info" in df.columns and "reward_model" in df.columns:
            rows = []
            for _, r in df.iterrows():
                ei = r["extra_info"] if isinstance(r["extra_info"], dict) else {}
                rm = r["reward_model"] if isinstance(r["reward_model"], dict) else {}
                q = ei.get("question", "")
                # fallback: extract user content from prompt list (deepmath format)
                if not q:
                    prompt = r.get("prompt") if hasattr(r, "get") else None
                    if prompt is None and "prompt" in df.columns:
                        prompt = r["prompt"]
                    if prompt is not None:
                        try:
                            prompt_iter = list(prompt)
                        except TypeError:
                            prompt_iter = []
                        for msg in prompt_iter:
                            if isinstance(msg, dict) and msg.get("role") == "user":
                                q = str(msg.get("content", "")).strip()
                                break
                a = rm.get("ground_truth", "")
                src = str(r.get("data_source", "math") or "math")
                # prefer reference_solution in reward_model, then top-level solution column
                sol = str(rm.get("reference_solution", "") or "")
                if not sol and "solution" in df.columns:
                    sol = str(r["solution"]) if r["solution"] else ""
                if q:
                    rows.append({"question": str(q), "answer": str(a),
                                 "solution": sol, "data_source": src})
            return rows
        q_col  = next((c for c in ["problem", "question"] if c in df.columns), None)
        gt_col = next((c for c in ["answer", "ground_truth"] if c in df.columns), None)
        if q_col is None or gt_col is None:
            return []
        sol_col = "solution" if "solution" in df.columns else None
        return [{"question": str(r[q_col]), "answer": str(r[gt_col]),
                 "solution": str(r[sol_col]) if sol_col else ""}
                for _, r in df.iterrows()]
    return []

py_logger = logging.getLogger(__name__)


def _task_cfg(harness_cfg, benchmark: str):
    """Return the task-specific sub-section of ``harness_cfg``.

    The harness config is organised by task (``harness.math``, ``harness.lawbench``,
    ``harness.uspto``); shared knobs (``port``, ``model_name``, ``workers``, ...)
    live at the top level.  This helper returns the task sub-section, falling
    back to the top-level ``harness_cfg`` when the sub-section is missing so
    that older flat configs keep loading.
    """
    sub = harness_cfg.get(benchmark, None) if hasattr(harness_cfg, "get") else None
    return sub if sub is not None else harness_cfg


class OPHSDTrainer(RayPPOTrainer):
    """OPHSD trainer: harness-augmented teacher for on-policy self-distillation."""

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, type],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
    ):
        if Role.ActorRollout not in role_worker_mapping and Role.ActorRolloutRef in role_worker_mapping:
            role_worker_mapping = dict(role_worker_mapping)
            role_worker_mapping[Role.ActorRollout] = role_worker_mapping[Role.ActorRolloutRef]

        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            processor=processor,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
        )

        # ── OPSD config ──
        opsd_cfg = config.get("opsd", {})
        self.loss_type = opsd_cfg.get("loss_type", "reverse_kl")
        self.beta = opsd_cfg.get("beta", 0.5)
        self.chunk_size = opsd_cfg.get("chunk_size", 512)
        self.opsd_max_length = opsd_cfg.get("max_length", 16384)
        self.sample_ratio = opsd_cfg.get("sample_ratio", 1.0)
        self.entropy_alpha = opsd_cfg.get("entropy_alpha", 2.0)
        self.jsd_gamma = opsd_cfg.get("jsd_gamma", 0.0)
        self.jsd_topk = opsd_cfg.get("jsd_topk", 0)
        self.test_freq = config.trainer.test_freq
        self.reward_beta = opsd_cfg.get("reward_beta", None)
        if self.reward_beta is not None and self.reward_beta <= 0:
            self.reward_beta = None

        apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        if OmegaConf.is_config(apply_chat_template_kwargs):
            apply_chat_template_kwargs = OmegaConf.to_container(apply_chat_template_kwargs, resolve=True)
        self.apply_chat_template_kwargs = dict(apply_chat_template_kwargs or {})

        # ── Benchmark selection ──
        self.benchmark = config.get("benchmark", "lawbench")
        if self.benchmark not in ("lawbench", "uspto", "math"):
            raise ValueError(f"Unsupported benchmark: {self.benchmark!r}. Choose 'lawbench', 'uspto', or 'math'.")
        py_logger.info("OPHSD benchmark: %s", self.benchmark)

        # ── Reward function ──
        self.reward_fn = get_custom_reward_fn(config)
        if self.reward_fn is None:
            if self.benchmark == "uspto":
                from rewards.uspto_reward import compute_score as _uspto_reward
                self.reward_fn = _uspto_reward
            elif self.benchmark == "math":
                from rewards.math_reward import compute_score as _math_reward
                self.reward_fn = _math_reward
            else:
                from harnesses.lawbench.evaluate import extract_predictions, compute_f1
                def _lawbench_reward(solution_str, ground_truth, **kw):
                    pred_set = set(extract_predictions(solution_str))
                    gt_set = {a.strip() for a in ground_truth.split(";") if a.strip()}
                    f1 = compute_f1(pred_set, gt_set)
                    return {"score": f1, "acc": float(f1 > 0), "pred": ";".join(sorted(pred_set))}
                self.reward_fn = _lawbench_reward

        # ── Harness components ──
        harness_cfg = config.get("harness", {})
        vllm_port = harness_cfg.get("port", 8000)
        model_name = harness_cfg.get("model_name", "Qwen3-8B")

        # Client URL resolved lazily after vLLM servers start (see _ensure_harness_ready)
        self._harness_base_url: Optional[str] = None
        self._harness_vllm_port = vllm_port
        self.harness_client = None  # initialised on first use

        self.memory_bank = MemoryBank()
        self._bank_added_indices: set[int] = set()
        _default_local_dir = config.trainer.get("default_local_dir", "checkpoints")
        self._bank_ckpt_path = os.path.join(_default_local_dir, "memory_bank.pkl")
        # PID-stamped filename: each trainer process (Ray actor) writes to its own file,
        # so multi-instance setups never produce duplicate lines.
        self._metrics_log_path = os.path.join(_default_local_dir, f"metrics_pid{os.getpid()}.jsonl")

        self._harness_model_name = model_name
        self._harness_cfg = harness_cfg
        self.harness_workers = harness_cfg.get("workers", harness_cfg.get("batch_size", 20))
        self.harness_val_workers = harness_cfg.get("val_workers", self.harness_workers)
        # Number of independent harness runs per val question (for pass@k, matches val_kwargs.n)
        self.harness_val_n = harness_cfg.get("val_n", self.config.actor_rollout_ref.rollout.val_kwargs.get("n", 1))

        task_cfg = _task_cfg(harness_cfg, self.benchmark)

        if self.benchmark == "math":
            # Math harness is stateless — no memory bank, no embeddings.
            self.harness_clf: Optional[PlanSolveHarness] = None  # built lazily
            self._embedder = None
            self._embed_fn = None
            self.memory_bank = None
            self._bank_added_indices = set()
            self.train_data_items = []
            self.train_embeddings = None
            self.val_embeddings = None

            # Val data is needed for harness pass@k evaluation.
            val_data_path = task_cfg.get("val_data_path", "")
            self.val_data_items = (
                _load_math_data(val_data_path)
                if val_data_path and os.path.exists(val_data_path) else []
            )
        else:
            embedding_model_path = task_cfg.get("embedding_model", "BAAI/bge-small-zh-v1.5")
            self._embedder = Embedder(model_name=embedding_model_path)
            self._embed_fn = self._embedder.encode_query
            # ``harness_clf`` is built lazily once vLLM server addresses are known.
            self.harness_clf: Optional[DraftVerificationHarness] = None

            # Pre-computed training data and embeddings (memory-bank seed).
            train_data_path = task_cfg.get("train_data_path", "")
            embeddings_path = task_cfg.get("embeddings_path", "")
            if self.benchmark == "uspto":
                self.train_data_items = load_uspto_train_data(train_data_path) if train_data_path else []
            else:
                self.train_data_items = load_lawbench_train_data(train_data_path) if train_data_path else []
            self.train_embeddings = (
                np.load(embeddings_path) if embeddings_path and os.path.exists(embeddings_path) else None
            )

            # Pre-computed val data and embeddings.
            val_data_path = task_cfg.get("val_data_path", "")
            val_embeddings_path = task_cfg.get("val_embeddings_path", "")
            if self.benchmark == "uspto":
                self.val_data_items = (
                    load_uspto_test_data(val_data_path)
                    if val_data_path and os.path.exists(val_data_path) else []
                )
            else:
                self.val_data_items = (
                    load_lawbench_test_data(val_data_path)
                    if val_data_path and os.path.exists(val_data_path) else []
                )
            self.val_embeddings = (
                np.load(val_embeddings_path)
                if val_embeddings_path and os.path.exists(val_embeddings_path) else None
            )

        py_logger.info(
            "OPHSD init: memory_bank=%s, train_data=%d, val_data=%d, harness_workers=%d",
            len(self.memory_bank) if self.memory_bank is not None else "N/A",
            len(self.train_data_items),
            len(self.val_data_items),
            self.harness_workers,
        )

    # ── Harness lazy init ──

    def _ensure_harness_ready(self):
        """Initialise harness client + classifier using the actual vLLM server URL.

        Called before the first harness inference.  verl's AgentLoopManager
        exposes ``server_addresses`` only after ``init_workers()`` completes,
        so we resolve the URL lazily here.
        """
        if self.harness_clf is not None:
            return

        base_url = None
        try:
            addrs = self.async_rollout_manager.server_addresses
            py_logger.info("vLLM server_addresses (%d): %s", len(addrs) if addrs else 0, addrs)
            if addrs:
                addr = addrs[0]
                if not addr.startswith("http"):
                    addr = f"http://{addr}"
                base_url = f"{addr}/v1"
                py_logger.info("Resolved vLLM server address: %s", base_url)
        except Exception as exc:
            py_logger.warning("Could not resolve vLLM server address: %s", exc)

        if base_url is None:
            base_url = f"http://localhost:{self._harness_vllm_port}/v1"
            py_logger.warning("Falling back to hardcoded harness URL: %s", base_url)

        self._harness_base_url = base_url
        self.harness_client = OpenAI(api_key="EMPTY", base_url=base_url)

        model_name = self._harness_model_name
        try:
            registered = [m.id for m in self.harness_client.models.list().data]
            if registered:
                if model_name not in registered:
                    model_name = registered[0]
                    py_logger.info("vLLM registered model '%s', using that instead of '%s'",
                                   model_name, self._harness_model_name)
        except Exception as exc:
            py_logger.warning("Could not list vLLM models, using config name '%s': %s", model_name, exc)

        harness_cfg = self._harness_cfg
        task_cfg = _task_cfg(harness_cfg, self.benchmark)

        if self.benchmark == "math":
            # Apply token budgets from config into the math harness module.
            import harnesses.math.config as _mh_cfg
            import harnesses.math.plan_solve as _ps

            solve_toks = task_cfg.get("solve_max_tokens", _mh_cfg.SOLVE_MAX_TOKENS)
            plan_toks = task_cfg.get("plan_max_tokens", _mh_cfg.PLAN_MAX_TOKENS)
            _mh_cfg.SOLVE_MAX_TOKENS = solve_toks
            _mh_cfg.PLAN_MAX_TOKENS = plan_toks
            py_logger.info(
                "Math harness token budgets: plan=%d solve=%d", plan_toks, solve_toks,
            )

            # Shared API timeout lives at the top level of harness_cfg.
            api_timeout = harness_cfg.get("api_timeout", harness_cfg.get("timeout"))
            if api_timeout is not None:
                api_timeout = int(api_timeout)
                _mh_cfg.API_TIMEOUT = api_timeout
                # ``plan_solve`` binds ``API_TIMEOUT`` at import time, so patch both modules.
                _ps.API_TIMEOUT = api_timeout
                py_logger.info("Math harness API_TIMEOUT=%ds", api_timeout)

            use_gt = bool(task_cfg.get("use_gt", True))
            self.harness_clf = PlanSolveHarness(
                client=self.harness_client,
                model=model_name,
                use_gt=use_gt,
            )
            py_logger.info("PlanSolveHarness ready (use_gt=%s)", use_gt)
        else:
            harness_cls = (
                USPTODraftVerificationHarness if self.benchmark == "uspto"
                else DraftVerificationHarness
            )
            self.harness_clf = harness_cls(
                client=self.harness_client,
                model=model_name,
                memory_bank=self.memory_bank,
                embed_fn=self._embed_fn,
                draft_k=task_cfg.get("draft_k", 5),
                confirm_k=task_cfg.get("confirm_k", 5),
                challenge_k=task_cfg.get("challenge_k", 5),
                cold_start_threshold=task_cfg.get("cold_start_threshold", 10),
            )
        py_logger.info(
            "Harness ready: url=%s model=%s benchmark=%s",
            base_url, model_name, self.benchmark,
        )

    # ── Metrics file logging ──

    def _log_metrics_to_file(self, metrics: dict, step: int, line_type: str = "train"):
        """Append one line to the per-run JSONL metrics file.

        Each line has a "type" field:
          "train"        — training-step metrics (one line per step)
          "val_student"  — bare-model validation metrics  (one line per val step)
          "val_harness"  — harness-augmented validation metrics (one line per val step)

        All float values are serialised with 6 decimal places.
        """
        os.makedirs(os.path.dirname(self._metrics_log_path), exist_ok=True)
        parts = [f'"step": {step}', f'"type": "{line_type}"']
        for k, v in metrics.items():
            k_json = json.dumps(str(k))
            try:
                parts.append(f"{k_json}: {float(v):.6f}")
            except (TypeError, ValueError):
                parts.append(f"{k_json}: {json.dumps(v, ensure_ascii=False)}")
        with open(self._metrics_log_path, "a", encoding="utf-8") as fh:
            fh.write("{" + ", ".join(parts) + "}\n")

    # ── Decode helpers (same as OPSD) ──

    def _decode_response_texts(self, batch: DataProto) -> list[str]:
        if "response_mask" not in batch.batch:
            batch.batch["response_mask"] = compute_response_mask(batch)
        decoded = []
        for response_ids, response_mask in zip(
            batch.batch["responses"], batch.batch["response_mask"], strict=True,
        ):
            decoded.append(self.tokenizer.decode(response_ids[response_mask.bool()], skip_special_tokens=True))
        return decoded

    def _pad_opsd_batch_for_dispatch(self, opsd_batch: DataProto) -> DataProto:
        n_dp = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        n_samples = opsd_batch.batch["student_input_ids"].shape[0]
        if n_samples == 0 or n_samples % n_dp == 0:
            return opsd_batch
        pad_to = ((n_samples // n_dp) + 1) * n_dp
        pad_count = pad_to - n_samples
        padded = {}
        for key in list(opsd_batch.batch.keys()):
            tensor = opsd_batch.batch[key]
            if key == "valid_row_mask" or key == "sample_weights":
                padded_rows = torch.zeros(pad_count, dtype=tensor.dtype, device=tensor.device)
            else:
                padded_rows = tensor[-1:].expand(pad_count, *tensor.shape[1:]).clone()
            padded[key] = torch.cat([tensor, padded_rows])
        return DataProto.from_single_dict(padded)

    # ── Harness inference ──

    def _run_harness_inference(self, data_items: list, embeddings) -> list[Optional[dict]]:
        """Run the full harness pipeline (draft + verify) in parallel.

        Args:
            data_items:  list of dicts with keys instruction / question / fact.
            embeddings:  numpy array of precomputed query embeddings (same length),
                         or None to let the harness compute them on the fly.

        Must be called while vLLM is still serving (before sleep_replicas).
        """
        self._ensure_harness_ready()
        n = len(data_items)
        results: list[Optional[dict]] = [None] * n

        if self.benchmark == "math":
            # Batch mode: all phases flattened into a single pool per phase.
            # max_workers covers Q×N Phase-1 calls + up to Q Phase-3/4 calls.
            # Always forward the dataset GT/solution; the harness itself
            # decides whether to look at them via its own ``use_gt`` flag.
            max_workers = self.harness_workers * self.harness_clf.n_solutions
            questions    = [s["question"]        for s in data_items]
            ground_truths = [s.get("answer", "")   for s in data_items]
            solutions     = [s.get("solution", "") for s in data_items]
            try:
                batch_out = self.harness_clf.predict_batch(
                    questions, ground_truths, solutions=solutions, max_workers=max_workers
                )
                for i, res in enumerate(batch_out):
                    results[i] = res
            except Exception as exc:
                py_logger.warning("Math harness batch inference failed: %s", exc)
            return results

        def _predict_one(i: int) -> tuple[int, Optional[dict]]:
            sample = data_items[i]
            try:
                emb = embeddings[i].copy() if (embeddings is not None and i < len(embeddings)) else None
                result = self.harness_clf.predict(
                    sample["instruction"],
                    sample["question"],
                    sample["fact"],
                    precomputed_emb=emb,
                )
                result.pop("query_emb", None)
                return i, result
            except Exception as exc:
                py_logger.warning("Harness inference failed idx=%d: %s", i, exc)
                return i, None

        with ThreadPoolExecutor(max_workers=self.harness_workers) as pool:
            futures = [pool.submit(_predict_one, i) for i in range(n)]
            for f in as_completed(futures):
                try:
                    idx, result = f.result()
                    results[idx] = result
                except Exception as exc:
                    py_logger.error("Harness future failed: %s", exc)

        n_ok = sum(1 for r in results if r is not None)
        n_cold = sum(1 for r in results if r is not None and r.get("mode") == "cold_start")
        n_full = sum(1 for r in results if r is not None and r.get("mode") == "full")
        n_plan = sum(1 for r in results if r is not None and r.get("mode") == "plan_solve")
        bank_size = len(self.memory_bank) if self.memory_bank is not None else 0
        py_logger.info(
            "Harness inference: %d/%d OK (cold_start=%d, full=%d, plan_solve=%d), memory_bank=%d",
            n_ok, n, n_cold, n_full, n_plan, bank_size,
        )
        return results

    # ── Memory bank helpers ──

    def _make_bank_entry(self, sample: dict) -> dict:
        """Build the dict stored in the memory bank for one sample.

        LawBench: uses ``fact`` / ``accusations`` / ``label_str``.
        USPTO:    uses ``fact`` (= rxn_smiles) / ``accusations`` (=[label_str]) / ``label_str``.

        Both share the same MemoryBank schema; ``accusations`` is used for
        label-based retrieval filtering in both DraftVerificationHarness variants.
        """
        fact = sample.get("fact", sample.get("rxn_smiles", ""))
        label_str = sample.get("label_str", "")
        accusations = sample.get("accusations", [label_str] if label_str else [])
        return {
            "fact": fact,
            "accusations": accusations,
            "label_str": label_str,
        }

    # ── Memory bank update ──

    def _update_memory_bank(self, batch: DataProto):
        """Add batch samples to the memory bank with their true labels and embeddings.

        No-op for math benchmark (no memory bank needed for Plan-Solve harness).
        """
        if self.memory_bank is None:
            return  # math benchmark: no memory bank

        extra_infos = batch.non_tensor_batch.get("extra_info", [])
        added = 0
        skipped_dup = 0

        for extra in extra_infos:
            if extra is None:
                continue
            sample_idx = extra.get("index", -1)
            if sample_idx < 0 or sample_idx >= len(self.train_data_items):
                continue

            if sample_idx in self._bank_added_indices:
                skipped_dup += 1
                continue

            sample = self.train_data_items[sample_idx]
            emb = None
            if self.train_embeddings is not None and sample_idx < len(self.train_embeddings):
                emb = self.train_embeddings[sample_idx].copy()
            else:
                emb = np.zeros(512, dtype=np.float32)

            self.memory_bank.add(self._make_bank_entry(sample), emb)
            self._bank_added_indices.add(sample_idx)
            added += 1

        py_logger.info(
            "Memory bank updated: +%d samples (skipped %d duplicates), total=%d",
            added, skipped_dup, len(self.memory_bank),
        )

    # ── Online harness validation ──

    def _run_online_harness_val(self) -> dict[int, dict]:
        """Run harness on val set with a fresh memory bank built from scratch.

        Mirrors the standalone harness protocol: samples are processed
        sequentially and each sample's ground truth is added to the bank
        *after* its prediction, so later samples benefit from earlier ones.

        Returns {val_data_items_index: harness_result_dict}.
        """
        self._ensure_harness_ready()

        if self.benchmark == "math":
            # Math: no memory bank; run PlanSolveHarness n_runs times per question
            return self._run_math_harness_val(self.harness_clf, n_runs=self.harness_val_n)

        val_bank = MemoryBank()
        harness_cfg = self._harness_cfg
        task_cfg = _task_cfg(harness_cfg, self.benchmark)
        harness_cls = (
            USPTODraftVerificationHarness if self.benchmark == "uspto"
            else DraftVerificationHarness
        )
        val_harness = harness_cls(
            client=self.harness_client,
            model=self.harness_clf.model,
            memory_bank=val_bank,
            embed_fn=self._embed_fn,
            draft_k=task_cfg.get("draft_k", 5),
            confirm_k=task_cfg.get("confirm_k", 5),
            challenge_k=task_cfg.get("challenge_k", 5),
            cold_start_threshold=task_cfg.get("cold_start_threshold", 10),
        )

        batch_size = self.harness_val_workers
        results: dict[int, dict] = {}
        n = len(self.val_data_items)
        n_batches = (n + batch_size - 1) // batch_size

        py_logger.info("Online harness val started: %d samples, %d workers, %d batches",
                       n, batch_size, n_batches)

        for start in range(0, n, batch_size):
            batch_indices = list(range(start, min(start + batch_size, n)))
            batch_num = start // batch_size + 1
            py_logger.info("  harness val batch %d/%d  (samples %d-%d)",
                           batch_num, n_batches, start, batch_indices[-1])

            def _predict_one(idx: int) -> tuple[int, Optional[dict]]:
                sample = self.val_data_items[idx]
                emb = self.val_embeddings[idx].copy() if (
                    self.val_embeddings is not None and idx < len(self.val_embeddings)
                ) else None
                try:
                    res = val_harness.predict(
                        sample["instruction"], sample["question"], sample["fact"],
                        precomputed_emb=emb,
                    )
                    res.pop("query_emb", None)
                    return idx, res
                except Exception as exc:
                    py_logger.warning("Val online harness failed idx=%d: %s", idx, exc)
                    return idx, None

            with ThreadPoolExecutor(max_workers=batch_size) as pool:
                batch_results = list(pool.map(_predict_one, batch_indices))

            for idx, res in batch_results:
                if res is not None:
                    results[idx] = res
                sample = self.val_data_items[idx]
                emb = self.val_embeddings[idx].copy() if (
                    self.val_embeddings is not None and idx < len(self.val_embeddings)
                ) else self._embed_fn(sample["fact"])
                val_bank.add(
                    self._make_bank_entry(sample),
                    emb,
                )

        py_logger.info("Online harness val done: %d/%d succeeded, bank=%d", len(results), n, len(val_bank))
        return results

    def _run_math_harness_val(self, harness_instance, n_runs: int = 1) -> dict[int, dict]:
        """Run :class:`PlanSolveHarness` on val items via ``predict_batch``.

        Each question is run ``n_runs`` times independently (for pass@k); all
        runs across all questions are submitted in a single batched dispatch.
        """
        n = len(self.val_data_items)
        if n == 0:
            return {}

        questions     = [s["question"]        for s in self.val_data_items]
        # Val harness is always GT-free regardless of the train-time
        # ``harness.use_gt`` setting: no reference solution in plan phase,
        # no answer hint in solve phase — measures pure plan-solve
        # generalisation. We pass empty strings explicitly so this stays
        # GT-free even if the harness instance was built with use_gt=True.
        ground_truths = [""] * n
        solutions     = [""] * n
        max_workers   = self.harness_val_workers * harness_instance.n_solutions

        py_logger.info(
            "Math harness val started: %d samples × %d runs, max_workers=%d",
            n, n_runs, max_workers,
        )

        run_buckets: dict[int, list[Optional[dict]]] = {i: [None] * n_runs for i in range(n)}

        for run in range(n_runs):
            try:
                batch_out = harness_instance.predict_batch(
                    questions, ground_truths, solutions=solutions, max_workers=max_workers
                )
                for idx, res in enumerate(batch_out):
                    run_buckets[idx][run] = res
            except Exception as exc:
                py_logger.warning("Math harness val run=%d failed: %s", run, exc)

        results: dict[int, dict] = {}
        for idx, runs in run_buckets.items():
            valid = [r for r in runs if r is not None]
            if valid:
                primary = valid[0]
                primary["all_responses"] = [
                    (r.get("final_response", "") if r is not None else "") for r in runs
                ]
                results[idx] = primary

        py_logger.info("Math harness val done: %d/%d questions have ≥1 result", len(results), n)
        return results

    def _dump_harness_val(self, harness_results: dict[int, dict], dump_path: str) -> None:
        """Dump math harness val responses as ``harness_{step}.jsonl``.

        Each line mirrors the student ``{step}.jsonl`` format so that
        ``plot_dapo_math_acc.py`` (or any offline scorer) can re-score both
        with the same grader:
          {"input": <question>, "output": <final_response>, "gts": <gt>, "step": N}

        All rollouts stored in ``all_responses`` are written as separate rows so
        pass@k can be computed offline the same way as for student rollouts.
        """
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"harness_{self.global_steps}.jsonl")
        lines = []
        for idx, res in sorted(harness_results.items()):
            if idx >= len(self.val_data_items):
                continue
            sample = self.val_data_items[idx]
            question = sample.get("question", "")
            gt = sample.get("answer", "")
            data_source = sample.get("data_source", "math")
            all_responses = res.get("all_responses", [res.get("final_response", "")])
            for resp in all_responses:
                if not resp:
                    continue
                lines.append(json.dumps({
                    "input":       question,
                    "output":      resp,
                    "gts":         gt,
                    "step":        self.global_steps,
                    "data_source": data_source,
                }, ensure_ascii=False))
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        py_logger.info("Dumped harness val responses → %s (%d rows)", filename, len(lines))

    def _run_warm_harness_val(self) -> dict[int, dict]:
        """Run harness on val set using the training memory bank (warm/retrieval-only).

        For math benchmark, this is the same as online val (no memory bank).
        Returns {val_data_items_index: harness_result_dict}.
        """
        self._ensure_harness_ready()

        if self.benchmark == "math":
            return self._run_math_harness_val(self.harness_clf, n_runs=self.harness_val_n)

        harness_cfg = self._harness_cfg
        task_cfg = _task_cfg(harness_cfg, self.benchmark)
        harness_cls = (
            USPTODraftVerificationHarness if self.benchmark == "uspto"
            else DraftVerificationHarness
        )
        warm_harness = harness_cls(
            client=self.harness_client,
            model=self.harness_clf.model,
            memory_bank=self.memory_bank,
            embed_fn=self._embed_fn,
            draft_k=task_cfg.get("draft_k", 5),
            confirm_k=task_cfg.get("confirm_k", 5),
            challenge_k=task_cfg.get("challenge_k", 5),
            cold_start_threshold=task_cfg.get("cold_start_threshold", 10),
        )

        n = len(self.val_data_items)
        results: dict[int, dict] = {}
        batch_size = self.harness_val_workers
        bank_size = len(self.memory_bank) if self.memory_bank is not None else 0

        py_logger.info("Warm harness val started: %d samples, bank=%d", n, bank_size)

        def _predict_one(idx: int) -> tuple[int, Optional[dict]]:
            sample = self.val_data_items[idx]
            emb = self.val_embeddings[idx].copy() if (
                self.val_embeddings is not None and idx < len(self.val_embeddings)
            ) else None
            try:
                res = warm_harness.predict(
                    sample["instruction"], sample["question"], sample["fact"],
                    precomputed_emb=emb,
                )
                res.pop("query_emb", None)
                return idx, res
            except Exception as exc:
                py_logger.warning("Warm harness val failed idx=%d: %s", idx, exc)
                return idx, None

        with ThreadPoolExecutor(max_workers=batch_size) as pool:
            for idx, res in pool.map(_predict_one, range(n)):
                if res is not None:
                    results[idx] = res

        py_logger.info("Warm harness val done: %d/%d succeeded", len(results), n)
        return results

    # ── Validation ──

    def _validate(self, merged: bool = False):
        """Validate the bare student, cold-start harness, and warm harness.

        Produces three parallel sets of metrics:
          val-student/...       — bare model output scored directly
          val-harness/...       — cold-start harness (fresh memory bank from scratch)
          val-warm-harness/...  — warm harness (training memory bank, retrieval only)
        """
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_uids = []

        # Harness val accumulators
        harness_val_accs: list[float] = []
        harness_val_f1s: list[float] = []
        harness_val_accs_by_src: dict[str, list[float]] = {}
        harness_val_f1s_by_src: dict[str, list[float]] = {}
        warm_val_accs: list[float] = []
        warm_val_f1s: list[float] = []

        # ── Online harness val (vLLM alive, fresh memory bank) ──
        val_harness_results: dict[int, dict] = {}
        warm_harness_results: dict[int, dict] = {}
        if self.val_data_items:
            py_logger.info("=== Validation started (step %d) ===", self.global_steps)
            try:
                val_harness_results = self._run_online_harness_val()
            except Exception as exc:
                py_logger.warning("Online harness val failed, skipping: %s", exc)
            if self.memory_bank is not None and len(self.memory_bank) > 0:
                try:
                    warm_harness_results = self._run_warm_harness_val()
                except Exception as exc:
                    py_logger.warning("Warm harness val failed, skipping: %s", exc)
        else:
            py_logger.warning(
                "Val harness skipped: val_data_items not loaded. "
                "Set harness.val_data_path in config."
            )

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            val_n = self.config.actor_rollout_ref.rollout.val_kwargs.get("n", 1)
            test_batch = test_batch.repeat(repeat_times=val_n, interleave=True)

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }

            size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            test_output_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            self.checkpoint_manager.sleep_replicas()
            test_output = unpad_dataproto(test_output_padded, pad_size=pad_size)

            test_batch = test_batch.union(test_output)
            test_batch.batch["response_mask"] = compute_response_mask(test_batch)
            responses = self._decode_response_texts(test_batch)

            extra_infos_raw = test_batch.non_tensor_batch.get("extra_info", [])
            extra_infos = extra_infos_raw if (extra_infos_raw is not None and len(extra_infos_raw) > 0) else [None] * len(responses)

            for i, (response, gt) in enumerate(zip(responses, ground_truths, strict=True)):
                if gt is not None:
                    result = self.reward_fn(solution_str=response, ground_truth=gt)
                    if isinstance(result, dict):
                        acc = float(result.get("acc", result.get("score", 0.0)))
                        f1 = float(result.get("score", acc))
                        pred = result.get("pred", "")
                    else:
                        acc = float(result)
                        f1 = acc
                        pred = ""
                else:
                    acc = 0.0
                    f1 = 0.0
                    pred = ""
                sample_scores.append(acc)
                reward_extra_infos_dict["reward"].append(acc)
                reward_extra_infos_dict["acc"].append(acc)
                reward_extra_infos_dict["f1"].append(f1)
                reward_extra_infos_dict["pred"].append(pred)

                # Look up online harness result by val_data_items index
                extra = extra_infos[i] if i < len(extra_infos) else None
                val_idx = extra.get("index", -1) if extra is not None else -1
                h_out = val_harness_results.get(val_idx)
                if h_out is not None and gt is not None:
                    all_h_responses = h_out.get("all_responses", [h_out.get("final_response", "")])
                    # pass@k: 1 if any response is correct
                    h_accs = []
                    for h_resp in all_h_responses:
                        if not h_resp:
                            h_accs.append(0.0)
                            continue
                        h_result = self.reward_fn(solution_str=h_resp, ground_truth=gt)
                        if isinstance(h_result, dict):
                            h_accs.append(float(h_result.get("acc", h_result.get("score", 0.0))))
                        else:
                            h_accs.append(float(h_result))
                    h_pass = float(any(a > 0 for a in h_accs))   # pass@k
                    h_mean = float(np.mean(h_accs)) if h_accs else 0.0  # mean@k
                    harness_val_accs.append(h_pass)
                    harness_val_f1s.append(h_mean)
                    # per-benchmark breakdown (math only — val_data_items has data_source)
                    if val_idx >= 0 and val_idx < len(self.val_data_items):
                        src = self.val_data_items[val_idx].get("data_source", "unknown")
                        harness_val_accs_by_src.setdefault(src, []).append(h_pass)
                        harness_val_f1s_by_src.setdefault(src, []).append(h_mean)

                # Look up warm harness result
                w_out = warm_harness_results.get(val_idx)
                if w_out is not None and gt is not None:
                    all_w_responses = w_out.get("all_responses", [w_out.get("final_response", "")])
                    w_accs = []
                    for w_resp in all_w_responses:
                        if not w_resp:
                            w_accs.append(0.0)
                            continue
                        w_result = self.reward_fn(solution_str=w_resp, ground_truth=gt)
                        if isinstance(w_result, dict):
                            w_accs.append(float(w_result.get("acc", w_result.get("score", 0.0))))
                        else:
                            w_accs.append(float(w_result))
                    warm_val_accs.append(float(any(a > 0 for a in w_accs)))
                    warm_val_f1s.append(float(np.mean(w_accs)) if w_accs else 0.0)

            sample_uids.extend(test_batch.non_tensor_batch["uid"])
            input_ids = test_batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_outputs.extend(responses)
            data_source_lst.append(
                test_batch.non_tensor_batch.get("data_source", ["unknown"] * len(test_batch))
            )

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs, outputs=sample_outputs,
                gts=sample_gts, scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )
            # Dump harness val responses alongside student rollouts so they can
            # be re-scored offline with the same grader for a fair comparison.
            if self.benchmark == "math" and val_harness_results:
                self._dump_harness_val(val_harness_results, dump_path=val_data_dir)

        data_sources = np.concatenate(data_source_lst, axis=0) if data_source_lst else np.array([])

        # Student metrics dict
        student_metrics = self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, [])

        # Benchmark-specific metric namespace
        val_ns = f"{self.benchmark}_val"

        # Harness metrics dict (separate, written as its own line)
        harness_metrics: dict = {}
        if harness_val_accs:
            k = self.harness_val_n
            harness_metrics[f"val-harness/{val_ns}/acc/pass@{k}"] = float(np.mean(harness_val_accs))
            harness_metrics[f"val-harness/{val_ns}/acc/mean@{k}"] = float(np.mean(harness_val_f1s))
            # per-benchmark breakdown
            for src, accs in harness_val_accs_by_src.items():
                src_safe = src.replace("/", "_")
                harness_metrics[f"val-harness/{src_safe}/acc/pass@{k}"] = float(np.mean(accs))
                harness_metrics[f"val-harness/{src_safe}/acc/mean@{k}"] = float(np.mean(harness_val_f1s_by_src.get(src, [0])))

        warm_metrics: dict = {}
        if warm_val_accs:
            k = self.harness_val_n
            warm_metrics[f"val-warm-harness/{val_ns}/acc/pass@{k}"] = float(np.mean(warm_val_accs))
            warm_metrics[f"val-warm-harness/{val_ns}/acc/mean@{k}"] = float(np.mean(warm_val_f1s))

        k = self.harness_val_n
        py_logger.info(
            "Val student: acc=%.4f f1=%.4f  |  harness pass@%d=%.4f mean@%d=%.4f",
            float(np.mean(reward_extra_infos_dict["acc"])) if reward_extra_infos_dict["acc"] else 0,
            float(np.mean(reward_extra_infos_dict["f1"])) if reward_extra_infos_dict["f1"] else 0,
            k, harness_metrics.get(f"val-harness/{val_ns}/acc/pass@{k}", 0),
            k, harness_metrics.get(f"val-harness/{val_ns}/acc/mean@{k}", 0),
        )

        self.checkpoint_manager.update_weights()
        return student_metrics, harness_metrics, warm_metrics

    # ── Main training loop ──

    def fit(self):
        """Main OPHSD training loop: harness-augmented distillation."""
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()
        self.checkpoint_manager.update_weights()
        progress = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="OPHSD Training")

        if self.config.trainer.get("val_before_train", True):
            student_m, harness_m, warm_m = self._validate()
            if student_m:
                logger.log(data={**student_m, **(harness_m or {}), **(warm_m or {})}, step=self.global_steps)
                self._log_metrics_to_file(student_m, step=self.global_steps, line_type="val_student")
                if harness_m:
                    self._log_metrics_to_file(harness_m, step=self.global_steps, line_type="val_harness")
                if warm_m:
                    self._log_metrics_to_file(warm_m, step=self.global_steps, line_type="val_warm_harness")
            if self.config.trainer.get("val_only", False):
                progress.close()
                return

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if self.global_steps >= self.total_training_steps:
                    progress.close()
                    py_logger.info("OPHSD training complete at step %d", self.global_steps)
                    return

                self.global_steps += 1
                step_t0 = time.time()
                metrics = {}

                batch = DataProto.from_single_dict(batch_dict)
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                saved_raw_prompt = batch.non_tensor_batch.get("raw_prompt")
                saved_extra_info = batch.non_tensor_batch.get("extra_info")

                gen_batch = self._get_gen_batch(batch)
                gen_batch.meta_info = {
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "recompute_log_prob": False,
                    "do_sample": True,
                    "global_steps": self.global_steps,
                }

                # 1. Student rollout (vLLM still running)
                gen_t0 = time.time()
                gen_output = self.async_rollout_manager.generate_sequences(gen_batch)
                metrics["timing/generate_s"] = time.time() - gen_t0

                # Restore non_tensor_batch fields popped by _get_gen_batch
                batch = batch.union(gen_output)
                if saved_raw_prompt is not None and "raw_prompt" not in batch.non_tensor_batch:
                    batch.non_tensor_batch["raw_prompt"] = saved_raw_prompt
                if saved_extra_info is not None and "extra_info" not in batch.non_tensor_batch:
                    batch.non_tensor_batch["extra_info"] = saved_extra_info

                # 2. Harness teacher inference (vLLM still running)
                harness_t0 = time.time()
                batch_len = len(batch.batch["responses"])
                extra_infos = batch.non_tensor_batch.get("extra_info", [None] * batch_len)
                harness_outputs: list[Optional[dict]] = [None] * batch_len

                if self.benchmark == "math":
                    # Math: no memory bank — run PlanSolveHarness directly per question
                    math_items_batch, math_positions = [], []
                    for pos, extra in enumerate(extra_infos):
                        if extra is not None and extra.get("question"):
                            math_items_batch.append({
                                "question": extra["question"],
                                "answer":   extra.get("answer", ""),
                            })
                            math_positions.append(pos)
                    py_logger.info(
                        "Math harness dispatch: %d/%d have question in extra_info",
                        len(math_items_batch), batch_len,
                    )
                    if math_items_batch:
                        sub = self._run_harness_inference(math_items_batch, None)
                        for pos, result in zip(math_positions, sub):
                            harness_outputs[pos] = result
                else:
                    train_items_batch, train_embs_batch = [], []
                    train_positions = []
                    n_extra_none = 0
                    n_extra_no_idx = 0
                    for pos, extra in enumerate(extra_infos):
                        if extra is None:
                            n_extra_none += 1
                            continue
                        idx = extra.get("index", -1)
                        if 0 <= idx < len(self.train_data_items):
                            train_items_batch.append(self.train_data_items[idx])
                            emb = self.train_embeddings[idx].copy() if (
                                self.train_embeddings is not None and idx < len(self.train_embeddings)
                            ) else None
                            train_embs_batch.append(emb)
                            train_positions.append(pos)
                        else:
                            n_extra_no_idx += 1
                    py_logger.info(
                        "Harness dispatch: %d/%d have valid extra_info index "
                        "(extra_none=%d, bad_idx=%d, train_data_items=%d)",
                        len(train_items_batch), batch_len,
                        n_extra_none, n_extra_no_idx, len(self.train_data_items),
                    )
                    if train_items_batch:
                        sub = self._run_harness_inference(
                            train_items_batch,
                            np.array(train_embs_batch) if (train_embs_batch and train_embs_batch[0] is not None) else None,
                        )
                        for pos, result in zip(train_positions, sub):
                            harness_outputs[pos] = result
                metrics["timing/harness_s"] = time.time() - harness_t0

                # ── Harness output statistics (math benchmark) ────────────
                # PlanSolveHarness emits mode ∈ {"plan_solve", "solve_only"}.
                if self.benchmark == "math":
                    n_harness_total = sum(1 for r in harness_outputs if r is not None)
                    n_plan_solve    = sum(1 for r in harness_outputs if r is not None and r.get("mode") == "plan_solve")
                    n_solve_only    = sum(1 for r in harness_outputs if r is not None and r.get("mode") == "solve_only")
                    metrics["harness/n_returned"]    = n_harness_total
                    metrics["harness/n_plan_solve"]  = n_plan_solve
                    metrics["harness/n_solve_only"]  = n_solve_only
                    metrics["harness/plan_solve_rate"] = n_plan_solve / max(1, n_harness_total)
                    py_logger.info(
                        "Harness stats: total=%d plan_solve=%d(%.0f%%) solve_only=%d",
                        n_harness_total, n_plan_solve,
                        100 * n_plan_solve / max(1, n_harness_total),
                        n_solve_only,
                    )

                # 3. vLLM sleeps
                sleep_t0 = time.time()
                self.checkpoint_manager.sleep_replicas()
                metrics["timing/sleep_s"] = time.time() - sleep_t0

                batch.batch["response_mask"] = compute_response_mask(batch)

                # Compute reward for logging / optional reward weighting
                responses = self._decode_response_texts(batch)
                ground_truths = [
                    item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch
                ]
                batch_size = len(responses)

                rewards = []
                f1_scores = []
                for response, ground_truth in zip(responses, ground_truths, strict=True):
                    if ground_truth is not None:
                        result = self.reward_fn(solution_str=response, ground_truth=ground_truth)
                        if isinstance(result, dict):
                            acc = float(result.get("acc", result.get("score", 0.0)))
                            f1 = float(result.get("score", acc))
                        else:
                            acc = float(result)
                            f1 = acc
                        rewards.append(1.0 if acc > 0 else 0.0)
                        f1_scores.append(f1)
                    else:
                        rewards.append(0.0)
                        f1_scores.append(0.0)
                n_correct = sum(rewards)
                metrics["ophsd/accuracy"] = n_correct / max(1, batch_size)
                metrics["ophsd/f1"] = float(np.mean(f1_scores)) if f1_scores else 0.0
                metrics["ophsd/batch_size"] = batch_size
                metrics["ophsd/memory_bank_size"] = len(self.memory_bank) if self.memory_bank is not None else 0

                sample_weights = None
                if self.reward_beta is not None:
                    reward_tensor = torch.tensor(rewards, dtype=torch.float32)
                    sample_weights = torch.softmax(reward_tensor / self.reward_beta, dim=0) * len(rewards)
                    metrics["ophsd/reward_weight_max"] = sample_weights.max().item()
                    metrics["ophsd/reward_weight_min"] = sample_weights.min().item()

                # 4. Build OPHSD batch (harness output as teacher context)
                build_t0 = time.time()
                ophsd_batch = build_ophsd_batch_from_verl_batch(
                    batch=batch,
                    tokenizer=self.tokenizer,
                    max_length=self.opsd_max_length,
                    harness_outputs=harness_outputs,
                    apply_chat_template_kwargs=self.apply_chat_template_kwargs,
                    sample_weights=sample_weights,
                )
                metrics["timing/build_batch_s"] = time.time() - build_t0

                # 5. KL divergence + backward + step
                fsdp_t0 = time.time()
                if ophsd_batch is not None:
                    ophsd_batch = self._pad_opsd_batch_for_dispatch(ophsd_batch)
                    ophsd_batch.meta_info["opsd_loss_type"] = self.loss_type
                    ophsd_batch.meta_info["opsd_beta"] = self.beta
                    ophsd_batch.meta_info["opsd_chunk_size"] = self.chunk_size
                    ophsd_batch.meta_info["opsd_sample_ratio"] = self.sample_ratio
                    ophsd_batch.meta_info["opsd_entropy_alpha"] = self.entropy_alpha
                    ophsd_batch.meta_info["opsd_jsd_gamma"] = self.jsd_gamma
                    ophsd_batch.meta_info["opsd_jsd_topk"] = self.jsd_topk
                    if self.reward_beta is not None:
                        ophsd_batch.meta_info["opsd_reward_beta"] = self.reward_beta

                    ophsd_output = self.actor_rollout_wg.update_opsd(ophsd_batch)
                    ophsd_metrics = reduce_metrics(ophsd_output.meta_info["metrics"])
                    metrics.update(ophsd_metrics)
                else:
                    metrics["ophsd/skipped"] = 1.0
                metrics["timing/fsdp_update_s"] = time.time() - fsdp_t0

                # 6. Update memory bank
                self._update_memory_bank(batch)

                # Save memory bank checkpoint periodically (skip for math benchmark)
                if self.memory_bank is not None and self.global_steps % 50 == 0:
                    os.makedirs(os.path.dirname(self._bank_ckpt_path), exist_ok=True)
                    self.memory_bank.save(self._bank_ckpt_path)
                    py_logger.info("Memory bank checkpoint saved (%d examples)", len(self.memory_bank))

                # 7. vLLM wakes up with updated weights
                wake_t0 = time.time()
                self.checkpoint_manager.update_weights()
                metrics["timing/update_weights_s"] = time.time() - wake_t0

                is_last = self.global_steps >= self.total_training_steps
                is_val = self.test_freq > 0 and self.global_steps % self.test_freq == 0

                student_m: dict = {}
                harness_m: dict = {}
                warm_m: dict = {}
                if is_val or is_last:
                    student_m, harness_m, warm_m = self._validate()

                save_freq = self.config.trainer.save_freq
                if save_freq > 0 and (is_last or self.global_steps % save_freq == 0):
                    self._save_checkpoint()

                metrics["training/global_step"] = self.global_steps
                metrics["training/epoch"] = epoch
                metrics["timing/step_s"] = time.time() - step_t0

                # Log to tracker (student val merged in for dashboards)
                logger.log(data={**metrics, **student_m, **harness_m, **warm_m}, step=self.global_steps)

                # Write metrics file: one train line per step; extra lines on val steps
                self._log_metrics_to_file(metrics, step=self.global_steps, line_type="train")
                if student_m:
                    self._log_metrics_to_file(student_m, step=self.global_steps, line_type="val_student")
                if harness_m:
                    self._log_metrics_to_file(harness_m, step=self.global_steps, line_type="val_harness")
                if warm_m:
                    self._log_metrics_to_file(warm_m, step=self.global_steps, line_type="val_warm_harness")

                progress.update(1)

                if is_last:
                    progress.close()
                    py_logger.info("OPHSD training complete at step %d", self.global_steps)
                    return

        progress.close()
        py_logger.info("OPHSD training complete!")
