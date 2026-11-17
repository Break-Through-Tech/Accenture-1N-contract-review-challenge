"""
Score a contract with the fine-tuned ModernBERT models (issue #31).

`predict_contract(raw_text)` cleans and chunks the text exactly the way training did
(cuad_cleaning.clean_text_with_map, then chunking_util.build_chunker + chunk_text_with_offsets), runs every
chunk through the trained model(s), and returns one row per (chunk, category) with a probability and a
yes/no prediction from the tuned per-category cutoffs. `contract_rollup` folds that into one row per category.

Default model = ensemble_top2: the plain mean of the sigmoid probabilities of modernbert_v2_biasinit and
modernbert_v1_6ep (error_analysis.ipynb cell 7), with the best-F1 cutoffs picked on the tune contracts in
results/error_analysis/selected_thresholds.csv. Those cutoffs were picked for the ensemble only, so any other
`runs=` has to come with its own `thresholds=` (picked on tune, never on val or test).

How to run
----------
The weights aren't in the repo. Each run is a Hugging Face save_pretrained folder (config.json +
model.safetensors) at <model root>/<run>/model/, written by trainer.save_model in modernbert_finetune.py.
On Google Drive the model root is cuad_finetune/runs/ (ask Albert for access; if the folder was shared with
you, add a shortcut to it in My Drive so Colab sees it under /content/drive/MyDrive). The model root is
looked up in this order: the `model_root` argument, the CUAD_MODEL_ROOT environment variable, then
/content/drive/MyDrive/cuad_finetune/runs if it exists.

Colab (GPU or CPU runtime), one cell per line group:

    from google.colab import drive
    drive.mount("/content/drive")

    !git clone --branch <branch-with-predict.py> https://github.com/Break-Through-Tech/Accenture-1N-contract-review-challenge.git /content/repo
    !pip install -q git+https://github.com/brandonstarxel/chunking_evaluation.git backoff
    %cd /content/repo/notebooks

    import os
    os.environ["CUAD_MODEL_ROOT"] = "/content/drive/MyDrive/cuad_finetune/runs"
    from predict import predict_contract, contract_rollup
    pred = predict_contract(open("/content/my_contract.txt", encoding="utf-8").read(), contract_id="my_contract")
    contract_rollup(pred)

Local (x64 Python 3.11-3.13 venv with requirements.txt installed), from notebooks/:

    python predict.py --text-file my_contract.txt --model-root C:/path/to/cuad_finetune/runs
    # or set CUAD_MODEL_ROOT once ($env:CUAD_MODEL_ROOT = "..." in PowerShell, export in bash) and drop --model-root

Precision: on an Ampere-or-newer GPU (A100, L4) the forward pass runs under bf16 autocast, the same as
trainer.predict when results/<run>/val_probs.parquet were written, so it reproduces those to ~1e-3. On CPU
(or a T4) it runs in fp32, which differs from the bf16-made probabilities by up to ~1e-2.
predict_check.py checks both.
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification

from chunking_util import MAX_LENGTH, MODEL_NAME, build_chunker, chunk_text_with_offsets, get_token_len_fn, load_tokenizer
from cuad_cleaning import clean_text_with_map

NOTEBOOK_DIR = Path(__file__).resolve().parent
REPO_DIR = NOTEBOOK_DIR.parent

# ensemble_top2: the two runs with the best tune macro AP (error_analysis.ipynb cell 7)
RUNS_ENSEMBLE = ("modernbert_v2_biasinit", "modernbert_v1_6ep")
THRESHOLDS_CSV = REPO_DIR / "results" / "error_analysis" / "selected_thresholds.csv"
THRESHOLD_COLUMN = "threshold_ensemble_top2"
# Known issue: Affiliate License-Licensor's cutoff in that file is 0.01. It had only 2 tune positives, and the
# low cutoff causes ~170 of the 684 validation false positives. Kept as is: the threshold-floor experiment (E1,
# results/improvement_experiments/) found no floor that helped on tune. To override it (or any other category),
# pass e.g. thresholds={"Affiliate License-Licensor": 0.5}; categories not in the mapping keep the CSV cutoff.

MODEL_ROOT_ENV = "CUAD_MODEL_ROOT"
COLAB_MODEL_ROOT = Path("/content/drive/MyDrive/cuad_finetune/runs")
N_CATEGORIES = 41
PREDICT_BATCH_SIZE = 8          # ~1 GB of activations at 768 tokens in fp32; fine on CPU and any Colab GPU
PRECISIONS = ("auto", "bf16", "fp32")


# --- text -> chunks (reuses the training cleaner + chunker, nothing re-implemented) ---
@lru_cache(maxsize=1)
def _chunker():
    # built once: loading the tokenizer is slow, and get_token_len_fn's per-piece cache carries over between contracts
    return build_chunker(get_token_len_fn(MODEL_NAME))


def chunk_contract(raw_text: str, contract_id: str = "contract") -> pd.DataFrame:
    """Clean + chunk one contract the way main_pipeline.ipynb did for training.

    chunk_start/chunk_end are offsets into the *cleaned* text (same as chunks_df); raw_start/raw_end point back
    into `raw_text` through the cleaner's char map, with raw_end exclusive. chunk_index counts every chunk the
    chunker produced, including any whose position couldn't be recovered; those are dropped here exactly as
    training dropped them (chunks_df[chunk_start >= 0]), so chunk ids match chunks_df and val_probs."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError("raw_text is empty -- nothing to score")
    clean_text, clean_to_raw = clean_text_with_map(raw_text)

    rows, n_dropped = [], 0
    for i, (chunk_text, (start, end)) in enumerate(chunk_text_with_offsets(_chunker(), clean_text)):
        if start < 0:
            n_dropped += 1
            continue
        rows.append({
            "contract_id": contract_id,
            "chunk_id": f"{contract_id}__chunk{i}",
            "chunk_index": i,
            "chunk_text": chunk_text,
            "chunk_start": start,
            "chunk_end": end,
            "raw_start": clean_to_raw[start],
            "raw_end": clean_to_raw[end - 1] + 1,   # chunk_end is exclusive, so map its last char and step past it
        })
    if n_dropped:
        warnings.warn(f"{contract_id}: {n_dropped} chunk(s) couldn't be located in the cleaned text and won't be "
                      f"scored (training dropped such chunks too)")
    if not rows:
        raise ValueError(f"{contract_id}: no chunks could be scored")
    return pd.DataFrame(rows)


# --- models: load once, reuse ---
def resolve_model_root(model_root=None) -> Path:
    if model_root is not None:
        root = Path(model_root)
    elif os.environ.get(MODEL_ROOT_ENV):
        root = Path(os.environ[MODEL_ROOT_ENV])
    elif COLAB_MODEL_ROOT.exists():
        root = COLAB_MODEL_ROOT
    else:
        raise FileNotFoundError(
            f"no model folder given: pass model_root=..., set {MODEL_ROOT_ENV}, or mount Google Drive in Colab "
            f"(expected {COLAB_MODEL_ROOT}). It should contain <run>/model/config.json for each of {RUNS_ENSEMBLE}"
        )
    if not root.is_dir():
        raise FileNotFoundError(f"model root {root} doesn't exist")
    return root


def run_model_dir(root: Path, run: str) -> Path:
    """<root>/<run>/model/ (the layout modernbert_finetune.py writes); <root>/<run>/ is accepted too."""
    for path in (root / run / "model", root / run):
        if (path / "config.json").exists():
            return path
    raise FileNotFoundError(f"no config.json in {root / run / 'model'} -- is {run} downloaded / is the root right?")


def resolve_device(device=None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_precision(precision: str, device: torch.device) -> str:
    """'auto' = bf16 autocast on Ampere-or-newer GPUs (what modernbert_finetune.pick_precision chose for the runs,
    so the saved probabilities are reproduced), fp32 everywhere else. Uses the compute capability, not
    torch.cuda.is_bf16_supported(), for the same reason pick_precision does (it can say True on a T4)."""
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    if precision == "fp32":
        return "fp32"
    bf16_ok = device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8
    if precision == "bf16" and not bf16_ok:
        raise ValueError("precision='bf16' needs an Ampere-or-newer GPU (A100, L4); use 'fp32' or 'auto'")
    return "bf16" if bf16_ok else "fp32"


@lru_cache(maxsize=8)
def _load_model(root: str, run: str, device: str):
    path = run_model_dir(Path(root), run)
    # sdpa = the attention the runs trained and predicted with (no flash-attn on Colab, nothing set in code);
    # pinned so a machine with flash-attn installed doesn't silently switch to unpadded attention
    model = AutoModelForSequenceClassification.from_pretrained(path, attn_implementation="sdpa")
    model = model.float().to(device).eval()    # fp32 weights; bf16 happens only through autocast, as in training
    categories = tuple(model.config.id2label[i] for i in range(model.config.num_labels))
    if len(categories) != N_CATEGORIES or any(c.startswith("LABEL_") for c in categories):
        raise ValueError(f"{path}: config.id2label doesn't name the {N_CATEGORIES} CUAD categories "
                         f"({len(categories)} labels, first: {categories[:2]})")
    return categories, model


def load_models(model_root=None, runs: Sequence[str] = RUNS_ENSEMBLE, device=None):
    """-> (categories in model output order, {run: model in eval mode}). Each model is loaded once per
    (root, run, device) and reused, so scoring v2 alone after the ensemble doesn't reload it."""
    runs = tuple(runs)
    if not runs:
        raise ValueError("runs is empty")
    root = str(resolve_model_root(model_root).resolve())
    device = str(resolve_device(device))
    categories, models = None, {}
    for run in runs:
        run_categories, models[run] = _load_model(root, run, device)
        if categories is not None and run_categories != categories:
            raise ValueError(f"{run} orders its categories differently from {runs[0]} -- can't average them")
        categories = run_categories
    return list(categories), models


# --- cutoffs ---
def load_thresholds(path: Path = THRESHOLDS_CSV, column: str = THRESHOLD_COLUMN) -> pd.Series:
    cutoffs = pd.read_csv(path).set_index("category")[column].astype(float)
    if len(cutoffs) != N_CATEGORIES or cutoffs.index.duplicated().any():
        raise ValueError(f"{path} [{column}] should have one cutoff for each of {N_CATEGORIES} categories")
    if not ((cutoffs > 0) & (cutoffs < 1)).all():
        raise ValueError(f"{path} [{column}] has cutoffs outside (0, 1)")
    return cutoffs


def resolve_thresholds(thresholds, categories: Sequence[str], runs: Sequence[str]) -> np.ndarray:
    """One cutoff per category, in model order.

    thresholds=None      -> selected_thresholds.csv, only allowed for the ensemble it was picked for
    a mapping            -> overrides those categories on top of the CSV (ensemble) or must cover all 41 (other runs)
    a number             -> the same cutoff for every category
    a 41-long sequence   -> already in model order"""
    categories = list(categories)
    is_ensemble = set(runs) == set(RUNS_ENSEMBLE)      # a mean doesn't depend on the order of the runs
    if thresholds is None or isinstance(thresholds, Mapping):
        if is_ensemble:
            base = load_thresholds()
            if set(base.index) != set(categories):
                raise ValueError(f"{THRESHOLDS_CSV} and the models name different categories")
        elif thresholds is None:
            raise ValueError(
                f"the cutoffs in {THRESHOLDS_CSV.name} were picked for ensemble_top2 {RUNS_ENSEMBLE} only; "
                f"pass thresholds= (picked on the tune contracts) to score with runs={tuple(runs)}"
            )
        else:
            base = pd.Series(np.nan, index=categories)
        overrides = dict(thresholds or {})
        unknown = set(overrides) - set(categories)
        if unknown:
            raise ValueError(f"thresholds names unknown categories: {sorted(unknown)}")
        base = base.reindex(categories).astype(float)
        for category, value in overrides.items():
            base.loc[category] = float(value)
        missing = base.index[base.isna()].tolist()
        if missing:
            raise ValueError(f"no cutoff for {len(missing)} categories, e.g. {missing[:3]}")
        cutoffs = base.to_numpy(dtype=float)
    elif isinstance(thresholds, (int, float)):
        cutoffs = np.full(len(categories), float(thresholds))
    else:
        cutoffs = np.asarray(thresholds, dtype=float)
        if cutoffs.shape != (len(categories),):
            raise ValueError(f"expected {len(categories)} cutoffs in model order, got shape {cutoffs.shape}")
    if not (np.isfinite(cutoffs).all() and (cutoffs >= 0).all() and (cutoffs <= 1).all()):
        raise ValueError("cutoffs must be in [0, 1]")
    return cutoffs


# --- chunks -> probabilities ---
def _autocast(precision: str):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if precision == "bf16" else nullcontext()


def score_chunks(chunk_texts: Sequence[str], *, model_root=None, runs: Sequence[str] = RUNS_ENSEMBLE, device=None,
                 precision: str = "auto", batch_size: int = PREDICT_BATCH_SIZE):
    """-> (categories, {run: float32 array (n_chunks, 41)}), rows in input order.

    Tokenized like training (modernbert_tokenizer_pipeline.tokenize_dataset: [CLS]/[SEP] on, no truncation,
    raise if over 768 tokens) and padded per batch with tokenizer.pad, which is what DataCollatorWithPadding does.
    probability = sigmoid of the fp32 logits, per category, as in modernbert_finetune.py."""
    device = resolve_device(device)
    precision = resolve_precision(precision, device)
    categories, models = load_models(model_root, runs, device)
    tokenizer = load_tokenizer(MODEL_NAME)

    texts = list(chunk_texts)
    encoded = tokenizer(texts, add_special_tokens=True, padding=False, truncation=False, return_attention_mask=True)
    lengths = [len(ids) for ids in encoded["input_ids"]]
    too_long = [i for i, n in enumerate(lengths) if n > MAX_LENGTH]
    if too_long:
        raise ValueError(f"{len(too_long)} chunk(s) exceed ModernBERT's {MAX_LENGTH}-token input "
                         f"(longest {max(lengths)}) -- the chunker config drifted from training")

    # longest first, so each batch pads to a similar length. Pad positions are masked out, so batching only
    # changes results at the float-noise level (predict_check.py compares batch sizes 8 and 1 to confirm)
    order = sorted(range(len(texts)), key=lambda i: -lengths[i])
    probs = {run: np.empty((len(texts), len(categories)), dtype=np.float32) for run in models}
    with torch.inference_mode():
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            batch = tokenizer.pad({"input_ids": [encoded["input_ids"][i] for i in idx],
                                   "attention_mask": [encoded["attention_mask"][i] for i in idx]},
                                  return_tensors="pt")
            batch = {k: v.to(device) for k, v in batch.items()}
            for run, model in models.items():
                with _autocast(precision):
                    logits = model(**batch).logits
                probs[run][idx] = torch.sigmoid(logits.float()).cpu().numpy()
    return list(categories), probs


# --- public API ---
def predict_contract(raw_text: str, *, contract_id: str = "contract", model_root=None,
                     runs: Sequence[str] = RUNS_ENSEMBLE, thresholds=None, device=None, precision: str = "auto",
                     batch_size: int = PREDICT_BATCH_SIZE) -> pd.DataFrame:
    """Raw contract text (CUAD `paragraphs[].context`, or any plain text) -> one row per (chunk, category):

        contract_id, chunk_id, chunk_index, chunk_start, chunk_end, raw_start, raw_end,
        category, prob, threshold, predicted

    chunk_start/chunk_end index the cleaned text, raw_start/raw_end index `raw_text` (end exclusive);
    predicted = prob >= threshold, the rule eval_metrics.multilabel_metrics uses. Chunk texts: chunk_contract()."""
    runs = tuple(runs)
    categories, _ = load_models(model_root, runs, device)
    cutoffs = resolve_thresholds(thresholds, categories, runs)    # fail on a cutoff problem before scoring
    chunks = chunk_contract(raw_text, contract_id)
    _, run_probs = score_chunks(chunks["chunk_text"], model_root=model_root, runs=runs, device=device,
                                precision=precision, batch_size=batch_size)
    # ensemble_top2 = plain mean of the runs' sigmoid probabilities (error_analysis.ipynb cell 7)
    probs = np.mean([run_probs[run] for run in runs], axis=0)

    n_chunks, n_cats = probs.shape
    out = pd.DataFrame({
        col: np.repeat(chunks[col].to_numpy(), n_cats)
        for col in ("contract_id", "chunk_id", "chunk_index", "chunk_start", "chunk_end", "raw_start", "raw_end")
    })
    out["category"] = np.tile(categories, n_chunks)
    out["prob"] = probs.ravel()
    out["threshold"] = np.tile(cutoffs, n_chunks)
    out["predicted"] = out["prob"] >= out["threshold"]
    return out


def contract_rollup(pred: pd.DataFrame) -> pd.DataFrame:
    """One row per (contract, category): the highest chunk probability, which chunk it came from, how many chunks
    are predicted yes, and contract_predicted = any chunk predicted yes. The team hasn't evaluated anything at
    contract level yet (all metrics so far are per chunk); this is the starting point for the November triage step."""
    keys = ["contract_id", "category"]
    top = pred.loc[pred.groupby(keys, sort=False)["prob"].idxmax(), [*keys, "chunk_id", "prob", "threshold"]]
    top = top.rename(columns={"chunk_id": "top_chunk_id", "prob": "max_prob"}).set_index(keys)
    top["n_chunks_predicted"] = pred.groupby(keys, sort=False)["predicted"].sum()
    top["contract_predicted"] = top["n_chunks_predicted"] > 0
    return top.reset_index()[[*keys, "contract_predicted", "max_prob", "threshold", "n_chunks_predicted", "top_chunk_id"]]


def main():
    parser = argparse.ArgumentParser(description="Score one contract text file with the fine-tuned ModernBERT model(s).")
    parser.add_argument("--text-file", type=Path, required=True, help="plain-text contract (UTF-8)")
    parser.add_argument("--contract-id", default=None, help="default: the file name without extension")
    parser.add_argument("--model-root", type=Path, default=None, help=f"folder holding <run>/model/; default ${MODEL_ROOT_ENV}")
    parser.add_argument("--runs", nargs="+", default=list(RUNS_ENSEMBLE))
    parser.add_argument("--threshold", type=float, default=None,
                        help="one cutoff for every category (required when --runs isn't the ensemble)")
    parser.add_argument("--precision", choices=PRECISIONS, default="auto")
    parser.add_argument("--device", default=None, help="cpu or cuda; default cuda if available")
    parser.add_argument("--batch-size", type=int, default=PREDICT_BATCH_SIZE)
    parser.add_argument("--out", type=Path, default=None, help="optional CSV for the per-chunk predictions")
    args = parser.parse_args()

    start = time.time()
    pred = predict_contract(args.text_file.read_text(encoding="utf-8"), contract_id=args.contract_id or args.text_file.stem,
                            model_root=args.model_root, runs=args.runs, thresholds=args.threshold, device=args.device,
                            precision=args.precision, batch_size=args.batch_size)
    rollup = contract_rollup(pred)
    print(f"{pred['chunk_id'].nunique()} chunks scored in {time.time() - start:.1f}s (incl. model load) "
          f"on {resolve_device(args.device)}")
    found = rollup[rollup["contract_predicted"]].sort_values("max_prob", ascending=False)
    print(f"\n{len(found)} of {len(rollup)} categories predicted present:")
    print(found.drop(columns="contract_id").to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if args.out:
        pred.to_csv(args.out, index=False)
        print(f"\nper-chunk predictions -> {args.out}")


if __name__ == "__main__":
    main()
