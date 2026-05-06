"""USPTO-50K Draft-Verify defaults and shared label constants."""

DRAFT_K: int = 5
VERIFY_CONFIRM_K: int = 5
VERIFY_CHALLENGE_K: int = 5
COLD_START_THRESHOLD: int = 10

# 10-class reaction-type taxonomy used in USPTO-50K.
CLASS_NAMES: dict[int, str] = {
    1:  "heteroatom alkylation and arylation",
    2:  "acylation and related processes",
    3:  "C-C bond formation",
    4:  "heterocycle formation",
    5:  "protections",
    6:  "deprotections",
    7:  "reductions",
    8:  "oxidations",
    9:  "functional group interconversion (FGI)",
    10: "functional group addition (FGA)",
}

_OPTIONS_STR = "\n".join(f"  {k}. {v}" for k, v in CLASS_NAMES.items())

# Canonical user-facing instruction prefix shared by data prep + harness prompts.
INSTRUCTION: str = (
    "You are an expert organic chemist. Given the following reaction SMILES, "
    "classify the reaction into one of the 10 reaction types listed below. "
    "Output only the class number inside [class] and <eoa> tags. "
    "For example: [class]3<eoa>\n\n"
    "Reaction types:\n"
    + _OPTIONS_STR
)
