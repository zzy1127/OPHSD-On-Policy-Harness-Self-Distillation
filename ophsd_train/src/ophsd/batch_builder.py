"""
Build paired teacher/student tokenized batches for OPHSD training.

Key difference from OPSD:
  - Teacher context = the actual harness prompt that was used when the model
    produced its final answer (full retrieval + draft + verify messages),
    NOT a post-hoc summary of the answer.
  - Both teacher and student compute token probabilities over the same
    student rollout response.  KL(student || teacher) teaches the student to
    match the teacher's distribution when it has full retrieval context.
"""

import logging
from typing import Optional

import torch
from transformers import PreTrainedTokenizer

from verl.protocol import DataProto

logger = logging.getLogger(__name__)

# Only log the first N teacher contexts to avoid flooding logs.
_TEACHER_LOG_BUDGET = 3
_teacher_log_count = 0


def _log_teacher_context(mode: str, messages: list[dict]):
    """Print teacher prompt to stdout (visible in Ray worker logs).

    Only fires for the first _TEACHER_LOG_BUDGET samples to avoid flooding.
    Full content is printed without truncation so the harness type can be verified.
    """
    global _teacher_log_count
    if _teacher_log_count >= _TEACHER_LOG_BUDGET:
        return
    _teacher_log_count += 1

    sep = "=" * 60
    lines = [
        "",
        sep,
        f"[OPHSD Teacher Context #{_teacher_log_count}]  mode={mode}  ({len(messages)} messages)",
        sep,
    ]
    for i, msg in enumerate(messages):
        role = msg.get("role", "?")
        content = msg.get("content", "")
        lines.append(f"--- [{i}] {role} ({len(content)} chars) ---")
        lines.append(content)   # full content, no truncation
    lines.append(sep)
    print("\n".join(lines), flush=True)


def _build_sequence_from_token_ids(
    prompt_ids: list[int],
    response_ids: torch.Tensor,
    max_length: int,
    pad_token_id: int,
) -> Optional[dict]:
    """Build a padded sequence: [prompt | response | padding]."""
    response_ids_list = response_ids.tolist()
    if not response_ids_list:
        return None

    if len(response_ids_list) >= max_length:
        response_ids_list = response_ids_list[: max_length - 1]

    max_prompt_len = max_length - len(response_ids_list)
    prompt_ids = prompt_ids[-max_prompt_len:]
    full_ids = prompt_ids + response_ids_list
    loss_mask = [0] * len(prompt_ids) + [1] * len(response_ids_list)

    seq_len = len(full_ids)
    if seq_len < 2:
        return None

    pad_len = max_length - seq_len
    return {
        "input_ids": torch.tensor(full_ids + [pad_token_id] * pad_len, dtype=torch.long),
        "attention_mask": torch.tensor([1] * seq_len + [0] * pad_len, dtype=torch.long),
        "position_ids": torch.tensor(list(range(seq_len)) + [0] * pad_len, dtype=torch.long),
        "loss_mask": torch.tensor(loss_mask + [0] * pad_len, dtype=torch.float32),
    }


def _get_response_mask(batch: DataProto) -> torch.Tensor:
    if "response_mask" in batch.batch:
        return batch.batch["response_mask"]

    if "attention_mask" not in batch.batch or "responses" not in batch.batch:
        raise KeyError("OPHSD batch construction requires response_mask or attention_mask + responses")

    response_length = batch.batch["responses"].shape[1]
    return batch.batch["attention_mask"][:, -response_length:]


def _build_harness_reference_messages(
    student_messages: list[dict],
    reference_solution: str,
    mode: str,
) -> list[dict]:
    """Wrap a harness-generated reference solution into a teacher prompt.

    Format:
        <original problem>

        Here is a reference solution:
        {plan + solve output from harness}

        After understanding the reference solution, please try to solve this
        problem using your own approach below.

    This keeps the teacher doing active reasoning (not just copying the harness
    output verbatim), which prevents KL from collapsing to a degenerate
    "repeat-the-reference" distribution.
    """
    student_content = ""
    for msg in reversed(student_messages):
        if msg.get("role") == "user":
            student_content = msg["content"]
            break

    teacher_content = (
        f"{student_content}\n\n"
        f"Here is a reference solution:\n{reference_solution}\n\n"
        f"After understanding the reference solution, please try to solve this "
        f"problem using your own approach below."
    )

    teacher_messages = []
    user_replaced = False
    for msg in student_messages:
        if msg.get("role") == "user" and not user_replaced:
            teacher_messages.append({"role": "user", "content": teacher_content})
            user_replaced = True
        else:
            teacher_messages.append(dict(msg))

    if not user_replaced:
        teacher_messages.append({"role": "user", "content": teacher_content})

    return teacher_messages


def _get_teacher_messages_from_harness(
    harness_out: dict,
    student_messages: list[dict],
) -> list[dict]:
    """Extract the teacher prompt from the harness trace.

    Supports two harness types:

    Type I (LawBench/USPTO — DraftVerificationHarness):
      Uses the harness draft/verify output as reference, wrapped in the
      "Here is a reference solution → solve it yourself" prompt format.
      Priority: verify → draft → cold_start → fallback

    Type II (Math — PlanSolveHarness):
      Concatenates plan + solve responses as the reference solution, wrapped
      in the same prompt format.  Teacher sees the full harness reasoning
      but must still produce its own derivation.
      Priority: solve (with plan prefix) → plan-only fallback → fallback
    """
    trace = harness_out.get("trace", {})
    mode  = harness_out.get("mode", "cold_start")

    # ── Math Plan-Solve harness ──
    # Teacher sees the solve_messages directly: ``[system, user(problem+plan)]``
    # with an explicit instruction to execute the plan.  The teacher must still
    # produce its own full solution — it has not seen the solve_response yet.
    # This mirrors the Draft-Verify path: teacher gets a structured *request*
    # (with helpful context) but must generate the answer itself, keeping the
    # KL target meaningful.
    if mode in ("plan_solve", "solve_only"):
        if "solve" in trace and trace["solve"].get("messages"):
            msgs = trace["solve"]["messages"]
            _log_teacher_context(mode, msgs)
            return msgs
        # fallback: plan messages only
        if "plan" in trace and trace["plan"].get("messages"):
            msgs = trace["plan"]["messages"]
            _log_teacher_context(f"{mode}_plan_fallback", msgs)
            return msgs

    # ── Type I: LawBench/USPTO DraftVerificationHarness ──
    # Use the harness messages directly — they already contain the full
    # retrieval context (similar cases, draft label, verify prompt) in the
    # correct language (Chinese for LawBench, English for USPTO).
    if mode == "full" and "verify" in trace:
        msgs = trace["verify"]["messages"]
        _log_teacher_context(mode, msgs)
        return msgs
    if mode in ("full", "draft_only") and "draft" in trace:
        msgs = trace["draft"]["messages"]
        _log_teacher_context(mode, msgs)
        return msgs
    if "cold_start" in trace:
        msgs = trace["cold_start"]["messages"]
        _log_teacher_context(mode, msgs)
        return msgs

    # Fallback: use student prompt unchanged (no harness context available)
    logger.warning("Harness trace missing messages for mode=%s; falling back to student prompt", mode)
    return list(student_messages)


def build_ophsd_batch_from_verl_batch(
    batch: DataProto,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 16384,
    harness_outputs: Optional[list[dict]] = None,
    apply_chat_template_kwargs: Optional[dict] = None,
    sample_weights: Optional[torch.Tensor] = None,
    max_response_kl_tokens: Optional[int] = None,
) -> Optional[DataProto]:
    """Build paired teacher/student sequences for KL distillation.

    For each sample:
      - Student sequence: bare prompt + student rollout response
      - Teacher sequence: full harness trace prompt (retrieved examples +
        draft + verify messages) + same student rollout response

    Both sequences share the same response tokens; the KL loss
    KL(P_student || P_teacher) trains the student to match the teacher's
    distribution, which benefits from the full retrieval context.

    Samples without a valid harness output are skipped.
    """
    if len(batch) == 0:
        return None
    if "raw_prompt" not in batch.non_tensor_batch:
        raise KeyError("OPHSD requires data.return_raw_chat=True so raw_prompt is available")
    if "responses" not in batch.batch:
        raise KeyError("OPHSD requires rollout responses in the batch")

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    response_mask = _get_response_mask(batch)
    chat_kwargs = dict(apply_chat_template_kwargs or {})
    student_seqs = []
    teacher_seqs = []
    kept_indices = []
    skip_empty_input = 0
    skip_no_harness = 0
    skip_student_seq = 0
    skip_teacher_seq = 0

    raw_prompts = batch.non_tensor_batch["raw_prompt"]
    responses = batch.batch["responses"]

    for sample_idx, (raw_prompt, response_ids, sample_response_mask) in enumerate(zip(
        raw_prompts, responses, response_mask, strict=True
    )):
        student_messages = list(raw_prompt)
        valid_response_ids = response_ids[sample_response_mask.bool()]

        if len(student_messages) == 0 or valid_response_ids.numel() == 0:
            skip_empty_input += 1
            continue

        # Get harness output for this sample
        harness_out = None
        if harness_outputs is not None and sample_idx < len(harness_outputs):
            harness_out = harness_outputs[sample_idx]

        if harness_out is None or not harness_out.get("final_response"):
            skip_no_harness += 1
            continue

        # Optionally truncate response for KL loss to the first N tokens,
        # concentrating gradient where prompt context matters most.
        kl_response_ids = valid_response_ids
        if max_response_kl_tokens is not None and valid_response_ids.numel() > max_response_kl_tokens:
            kl_response_ids = valid_response_ids[:max_response_kl_tokens]

        # Student sequence: bare prompt + (truncated) student rollout response
        student_prompt_ids = tokenizer.apply_chat_template(
            student_messages, add_generation_prompt=True, tokenize=True, **chat_kwargs,
        )
        s_seq = _build_sequence_from_token_ids(student_prompt_ids, kl_response_ids, max_length, pad_token_id)
        if s_seq is None:
            skip_student_seq += 1
            continue

        # Teacher sequence: full harness trace prompt + same (truncated) student response
        teacher_messages = _get_teacher_messages_from_harness(harness_out, student_messages)
        teacher_prompt_ids = tokenizer.apply_chat_template(
            teacher_messages, add_generation_prompt=True, tokenize=True, **chat_kwargs,
        )
        t_seq = _build_sequence_from_token_ids(teacher_prompt_ids, kl_response_ids, max_length, pad_token_id)
        if t_seq is None:
            skip_teacher_seq += 1
            continue

        student_seqs.append(s_seq)
        teacher_seqs.append(t_seq)
        kept_indices.append(sample_idx)

    total = len(raw_prompts)
    skipped = skip_empty_input + skip_no_harness + skip_student_seq + skip_teacher_seq
    if skipped:
        logger.warning(
            "OPHSD batch: kept %d/%d — skipped: empty_input=%d, no_harness=%d, "
            "student_seq_fail=%d, teacher_seq_fail=%d  (max_length=%d)",
            len(student_seqs), total,
            skip_empty_input, skip_no_harness,
            skip_student_seq, skip_teacher_seq,
            max_length,
        )

    if not student_seqs:
        return None

    batch_dict = {}
    for prefix, seqs in [("teacher_", teacher_seqs), ("student_", student_seqs)]:
        for key in ["input_ids", "attention_mask", "position_ids", "loss_mask"]:
            batch_dict[f"{prefix}{key}"] = torch.stack([s[key] for s in seqs])
    batch_dict["valid_row_mask"] = torch.ones(len(student_seqs), dtype=torch.bool)

    if sample_weights is not None:
        batch_dict["sample_weights"] = sample_weights[kept_indices]

    return DataProto.from_single_dict(batch_dict)
