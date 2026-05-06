"""Math reward function for OPHSD training.

Equivalence logic is intentionally identical to
``harnesses/math/evaluate.py::answers_match`` so that the offline harness
evaluator and the online training reward agree on every scored sample.

Interface matches the other reward functions in this directory:
    compute_score(solution_str, ground_truth, **kw) -> dict
        score : float  — 1.0 if correct, 0.0 otherwise
        acc   : float  — same as score
        pred  : str    — extracted prediction string
"""

import re
import unicodedata


# ---------------------------------------------------------------------------
# Hendrycks strip_string normalisation (mirrors harnesses/math/evaluate.py)
# ---------------------------------------------------------------------------

def _fix_fracs(s: str) -> str:
    parts = s.split("\\frac")
    out = parts[0]
    for p in parts[1:]:
        out += "\\frac"
        if p and p[0] == "{":
            out += p
        elif len(p) >= 2:
            a, b = p[0], p[1]
            rest = p[2:]
            if b != "{":
                out += "{" + a + "}{" + b + "}" + rest
            else:
                out += "{" + a + "}" + b + rest
        else:
            return s
    return out


def _fix_sqrt(s: str) -> str:
    if "\\sqrt" not in s:
        return s
    parts = s.split("\\sqrt")
    out = parts[0]
    for p in parts[1:]:
        if p and p[0] != "{":
            out += "\\sqrt{" + p[0] + "}" + p[1:]
        else:
            out += "\\sqrt" + p
    return out


def _fix_slash(s: str) -> str:
    if len(s.split("/")) != 2:
        return s
    a, b = s.split("/")
    try:
        ia, ib = int(a), int(b)
        assert s == f"{ia}/{ib}"
        return "\\frac{" + str(ia) + "}{" + str(ib) + "}"
    except Exception:
        return s


def _strip_string(s: str) -> str:
    s = s.replace("\n", "")
    s = s.replace("\\!", "")
    s = s.replace("\\\\", "\\")
    s = s.replace("tfrac", "frac")
    s = s.replace("dfrac", "frac")
    s = s.replace("\\left", "")
    s = s.replace("\\right", "")
    s = s.replace("^{\\circ}", "")
    s = s.replace("^\\circ", "")
    s = s.replace("\\$", "")
    if "\\text{" in s:
        s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = s.replace("\\\\%", "")
    s = s.replace("\\%", "")
    s = s.replace(" .", " 0.")
    s = s.replace("{.", "{0.")
    if not s:
        return s
    if s[0] == ".":
        s = "0" + s
    if len(s.split("=")) == 2 and len(s.split("=")[0]) <= 2:
        s = s.split("=")[1]
    s = _fix_sqrt(s)
    s = s.replace(" ", "")
    s = _fix_fracs(s)
    if s == "0.5":
        s = "\\frac{1}{2}"
    s = _fix_slash(s)
    return s


def _last_boxed(text: str):
    """Return the last \\boxed{...} substring (including the \\boxed{ and })."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None
    if "\\boxed " in text:
        return "\\boxed " + text.split("\\boxed ")[-1].split("$")[0]
    i = idx
    depth = 0
    right = None
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                right = i
                break
        i += 1
    return None if right is None else text[idx : right + 1]


def _remove_boxed(s: str) -> str:
    if s.startswith("\\boxed "):
        return s[len("\\boxed ") :]
    left = "\\boxed{"
    if not s.startswith(left) or not s.endswith("}"):
        return s
    return s[len(left) : -1]


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"\s+", "", s).lower().strip().replace("$", "")


def _answers_match(pred: str, gold: str) -> bool:
    """Identical logic to ``harnesses/math/evaluate.py::answers_match``."""
    if not pred or not gold:
        return False

    try:
        if _strip_string(pred) == _strip_string(gold):
            return True
    except Exception:
        pass

    if _norm(pred) == _norm(gold):
        return True

    try:
        from math_verify import parse, verify as mv_verify

        p_wrapped = pred if "$" in pred else f"${pred}$"
        g_wrapped = gold if "$" in gold else f"${gold}$"
        p_parsed = parse(p_wrapped, fallback_mode="no_fallback", parsing_timeout=None)
        g_parsed = parse(g_wrapped, fallback_mode="no_fallback", parsing_timeout=None)
        return mv_verify(g_parsed, p_parsed, timeout_seconds=None)
    except Exception:
        return False


def _is_equiv(pred: str, gold: str) -> bool:
    if pred is None or gold is None:
        return False
    try:
        return _strip_string(pred) == _strip_string(gold)
    except Exception:
        return pred == gold


def compute_score(solution_str: str, ground_truth: str, **kwargs) -> dict:
    """Main reward entry point (full answers_match including math_verify)."""
    pred = ""
    score = 0.0
    try:
        boxed = _last_boxed(solution_str)
        if boxed is not None:
            pred = _remove_boxed(boxed)
            score = float(_answers_match(pred, ground_truth))
    except Exception:
        pass
    return {"score": score, "acc": score, "pred": pred}


def compute_score_fast(solution_str: str, ground_truth: str, **kwargs) -> dict:
    """Same contract as ``compute_score`` but **no math_verify** (strip + norm only).

    Used by ``OPSDTrainer`` on the math benchmark so train/val logging cannot
    block on sympy.  Final accuracy should be measured offline with the full grader.
    """
    pred = ""
    score = 0.0
    try:
        boxed = _last_boxed(solution_str)
        if boxed is not None:
            pred = _remove_boxed(boxed)
            if pred and ground_truth:
                try:
                    if _strip_string(pred) == _strip_string(ground_truth):
                        score = 1.0
                    elif _norm(pred) == _norm(ground_truth):
                        score = 1.0
                except Exception:
                    pass
    except Exception:
        pass
    return {"score": score, "acc": score, "pred": pred}
