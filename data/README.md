# Training data

Three 10 k-sample training sets used in the paper.  Each `data_train_10k.json`
is a JSON array; columns are documented per file.

## DeepMath 10k — `deepmath10k/data_train_10k.json`

Mathematical-reasoning training data sampled from
[DeepMath](https://huggingface.co/datasets/zwhe99/DeepMath-103K).  Each row
has:

```json
{
    "question":  "Find the trace of a linear transformation T ... \\boxed{}.",
    "answer":    "0",
    "solution":  "Okay, so I need to find the trace ..."
}
```

The `question` text already includes the standard "put your final answer
within `\boxed{}`" tail, so the launcher script does not append any system
prompt.  The `solution` is the reference solution shown to the Plan Agent
when `harness.math.use_gt=true`.

## CAIL / LawBench 10k — `cail10k/data_train_10k.json`

Sampled training data for LawBench task 3-3 (accusation prediction over the
Chinese AI and Law (CAIL) corpus).  Each row has:

```json
{
    "instruction": "你是一位资深法官，...",
    "question":    "事实: ...",
    "fact":        "...",
    "answer":      "罪名: 盗窃;诈骗",
    "accusations": ["盗窃", "诈骗"],
    "label_str":   "盗窃;诈骗"
}
```

`fact` is used by the Draft-Verify harness as the retrieval key.

## USPTO-50K 10k — `uspto10k/data_train_10k.json`

Stratified-sample subset of USPTO-50K reaction classification.  Each row has:

```json
{
    "instruction": "You are an expert organic chemist. ...",
    "question":    "Reaction SMILES: ...",
    "rxn_smiles":  "...",
    "prod_smiles": "...",
    "id":          "patent_id",
    "class":       3,
    "class_name":  "C-C bond formation",
    "label_str":   "3"
}
```

`rxn_smiles` is used by the Draft-Verify harness as the retrieval key.

## Converting to verl parquet

The OPHSD trainer expects verl-format parquet files.  See
`ophsd_train/data_prep/prepare_*.py`:

```bash
cd ophsd_train
python -m data_prep.prepare_deepmath_data    # → data/deepmath10k/train.parquet
python -m data_prep.prepare_lawbench_data    # → data/cail10k/train.parquet
python -m data_prep.prepare_uspto_data       # → data/uspto10k/train.parquet
```

## Validation data

This open-source bundle ships only training data.  Use any held-out
LawBench / USPTO / AIME / MATH-500 / OlympiadBench / HMMT split as the
validation set; pass it via the `VAL_FILE` env var to the launcher script.
