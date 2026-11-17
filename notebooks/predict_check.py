"""
Checks for predict.py (issue #31, "Done when": predict_contract reproduces the saved validation probabilities and
runs on a contract it hasn't seen). Exits with code 1 if any check fails.

  A. Chunks: for 3 validation contracts (fewest, median and most chunks in val_probs), the raw CUAD text run
     through predict.chunk_contract gives exactly the chunk ids in results/<run>/val_probs.parquet; with
     --chunks-df, also the same chunk_text/chunk_start/chunk_end as training's chunks_df.parquet. Every chunk's
     raw_start/raw_end slice of the raw text, cleaned again, equals the chunk text. Needs no model weights.
  B. Probabilities, joined on chunk_id, for each run against its own val_probs and for the ensemble against the
     mean of the two saved files. Max abs difference over all (chunk, category) pairs:
       B1 (GPU, bf16 autocast = how val_probs were made; the acceptance test): <= 1e-3, each run and ensemble.
       B2 (fp32, e.g. CPU):  <= 2e-2, plus the ensemble's yes/no at the selected cutoffs must agree with the saved
          ensemble's. Every disagreeing (chunk, category) pair is listed with both probabilities and the cutoff.
     --batch-sizes 8 1 repeats B with batch size 1 (no padding at all), to tell padding noise from a real bug.
  C. Smoke test on an unseen contract: --smoke-text (preferred: any non-CUAD contract) or --smoke-test-json N
     (reads only the text of test.json contract N; no labels are read, no metrics computed, nothing tuned).
     Checks shape, ids, probability range, predicted == prob >= threshold, raw offsets, and times it.

How to run
----------
Check A only, anywhere the chunking stack is installed (no weights needed), from notebooks/:

    python predict_check.py --skip-probs

Colab, GPU runtime (Runtime > Change runtime type > A100 or L4, so bf16 matches training), one cell per group:

    from google.colab import drive
    drive.mount("/content/drive")

    !git clone --branch <branch-with-predict.py> https://github.com/Break-Through-Tech/Accenture-1N-contract-review-challenge.git /content/repo
    !pip install -q git+https://github.com/brandonstarxel/chunking_evaluation.git backoff
    %cd /content/repo/notebooks

    import os
    os.environ["CUAD_MODEL_ROOT"] = "/content/drive/MyDrive/cuad_finetune/runs"
    !python predict_check.py --chunks-df /content/drive/MyDrive/cuad_finetune/data/chunks_df.parquet --batch-sizes 8 1 --smoke-text /content/my_contract.txt

Then switch to a CPU runtime, re-run the cells above, and run the fp32 check (B2):

    !python predict_check.py --precision fp32 --chunks-df /content/drive/MyDrive/cuad_finetune/data/chunks_df.parquet --smoke-text /content/my_contract.txt

Local x64 Python 3.11-3.13 venv (CPU, so B2), from notebooks/:

    python predict_check.py --model-root C:/path/to/cuad_finetune/runs --smoke-text my_contract.txt

If B fails, check in order: chunk ids (A), category order, tokenizer (the model folder's own tokenizer is compared
below), sigmoid vs softmax, precision / attention implementation, transformers version (runs used 5.20.1).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer

import predict
from chunking_util import MODEL_NAME, load_tokenizer
from cuad_cleaning import clean_text_with_map
from splits import load_splits

REPO_DIR = predict.REPO_DIR
TRAIN_JSON = REPO_DIR / "data" / "cuad" / "train_separate_questions.json"   # the file main_pipeline.ipynb chunked
TEST_JSON = REPO_DIR / "data" / "cuad" / "test.json"                       # smoke test only: text, never labels
RESULTS_DIR = REPO_DIR / "results"
TOL_BF16 = 1e-3
TOL_FP32 = 2e-2
N_CONTRACTS = 3
TRAINED_WITH = {"torch": "2.11.0+cu130", "transformers": "5.20.1"}   # results/<run>/run_config.json


class Report:
    def __init__(self):
        self.failures = []

    def check(self, ok: bool, label: str, detail: str = ""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
        if not ok:
            self.failures.append(label)
        return ok


def load_contract_texts(path: Path) -> dict[str, str]:
    """title -> raw contract text (paragraphs[0].context), the input main_pipeline.ipynb cleaned and chunked."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    texts = {}
    for doc in raw["data"]:
        assert len(doc["paragraphs"]) == 1, f"{doc['title']}: expected one paragraph (the whole contract)"
        texts[doc["title"]] = doc["paragraphs"][0]["context"]
    return texts


def load_val_probs(runs) -> dict[str, pd.DataFrame]:
    refs = {run: pd.read_parquet(RESULTS_DIR / run / "val_probs.parquet") for run in runs}
    first = next(iter(refs.values()))
    for run, df in refs.items():
        assert df["chunk_id"].equals(first["chunk_id"]), f"{run}: val_probs rows differ from {runs[0]}"
    return refs


def pick_contracts(val_ref: pd.DataFrame, n: int = N_CONTRACTS) -> list[str]:
    """Fewest, median and most chunks: covers a short contract and the longest one in validation."""
    sizes = val_ref.groupby("contract_id").size().sort_values(kind="stable")
    positions = sorted({0, len(sizes) // 2, len(sizes) - 1})[:n]
    return [sizes.index[p] for p in positions]


def check_raw_offsets(report: Report, raw_text: str, chunks: pd.DataFrame, label: str):
    """Approved decision 5: raw_text[raw_start:raw_end], cleaned again, has to give back the chunk text.
    (The chunker strips chunk edges, so .strip() on chunk_text only matters if that ever changes.)"""
    bad = [row.chunk_id for row in chunks.itertuples(index=False)
           if clean_text_with_map(raw_text[row.raw_start:row.raw_end])[0] != row.chunk_text.strip()]
    empty = int((chunks["raw_end"] <= chunks["raw_start"]).sum())
    report.check(not bad and not empty, f"{label}: raw offsets reproduce every chunk ({len(chunks)})",
                 f"{len(bad)} mismatched, e.g. {bad[:3]}; {empty} empty" if bad or empty else "")


# --- A: chunks ---
def check_chunks(report: Report, contracts, texts, val_ref, chunks_ref) -> dict[str, pd.DataFrame]:
    print("\n=== A. chunk boundaries ===")
    out = {}
    for cid in contracts:
        chunks = predict.chunk_contract(texts[cid], contract_id=cid)
        saved_ids = val_ref.loc[val_ref["contract_id"] == cid, "chunk_id"].tolist()
        ours = chunks["chunk_id"].tolist()
        report.check(ours == saved_ids, f"{cid[:60]}: chunk ids match val_probs ({len(saved_ids)})",
                     "" if ours == saved_ids else f"ours {len(ours)}, saved {len(saved_ids)}; "
                     f"first difference at {next((i for i, (a, b) in enumerate(zip(ours, saved_ids)) if a != b), min(len(ours), len(saved_ids)))}")
        if chunks_ref is not None:
            ref = chunks_ref[chunks_ref["contract_id"] == cid].set_index("chunk_id")
            ref = ref[ref["chunk_start"] >= 0]
            mine = chunks.set_index("chunk_id")
            same_ids = mine.index.equals(ref.index)
            cols = ["chunk_text", "chunk_start", "chunk_end"]
            diff = [] if not same_ids else [c for c in cols if not mine[c].astype(ref[c].dtype).equals(ref[c])]
            report.check(same_ids and not diff, f"{cid[:60]}: text/start/end match training's chunks_df",
                         "chunk ids differ" if not same_ids else (f"differs in {diff}" if diff else ""))
        check_raw_offsets(report, texts[cid], chunks, cid[:60])
        out[cid] = chunks
    return out


# --- B: probabilities ---
def compare(new: np.ndarray, saved: np.ndarray, chunk_ids, categories) -> dict:
    diff = np.abs(new.astype(np.float64) - saved.astype(np.float64))
    r, c = np.unravel_index(int(np.argmax(diff)), diff.shape)
    per_cat = pd.Series(diff.max(axis=0), index=categories).sort_values(ascending=False)
    return {"max": float(diff.max()), "mean": float(diff.mean()), "at": (chunk_ids[r], categories[c]),
            "saved": float(saved[r, c]), "new": float(new[r, c]), "per_cat": per_cat}


def check_tokenizer(report: Report, model_root, run: str, texts):
    """Cause #3 in the plan: the base tokenizer (used for chunking and inference) vs. the copy saved with the model."""
    model_dir = predict.run_model_dir(predict.resolve_model_root(model_root), run)
    if not any((model_dir / f).exists() for f in ("tokenizer.json", "tokenizer_config.json")):
        print(f"  (no tokenizer saved in {model_dir}; skipping the tokenizer comparison)")
        return
    base, saved = load_tokenizer(MODEL_NAME), AutoTokenizer.from_pretrained(model_dir)
    same = base(list(texts))["input_ids"] == saved(list(texts))["input_ids"]
    report.check(same, f"base tokenizer == tokenizer saved with {run} on the checked chunks")


def check_probs(report: Report, chunks_by_contract, val_refs, runs, model_root, device, precision, batch_size):
    chunks = pd.concat(chunks_by_contract.values(), ignore_index=True)
    ids = chunks["chunk_id"].tolist()
    start = time.time()
    categories, new = predict.score_chunks(chunks["chunk_text"], model_root=model_root, runs=runs, device=device,
                                           precision=precision, batch_size=batch_size)
    seconds = time.time() - start
    used = predict.resolve_precision(precision, predict.resolve_device(device))
    tol = TOL_BF16 if used == "bf16" else TOL_FP32
    name = "B1" if used == "bf16" else "B2"
    print(f"\n=== {name}. probabilities: {used} on {predict.resolve_device(device)}, batch size {batch_size}, "
          f"{len(ids)} chunks x {len(runs)} run(s) in {seconds:.1f}s; tolerance {tol:g} ===")

    saved = {}
    for run in runs:
        ref = val_refs[run]
        ref_cats = ref.columns[2:].tolist()
        report.check(ref_cats == categories, f"{run}: model category order == val_probs column order")
        saved[run] = ref.set_index("chunk_id").loc[ids, categories].to_numpy()

    results = {run: compare(new[run], saved[run], ids, categories) for run in runs}
    if set(runs) == set(predict.RUNS_ENSEMBLE):
        ens_new = np.mean([new[r] for r in runs], axis=0)
        ens_saved = np.mean([saved[r] for r in runs], axis=0)
        results["ensemble_top2"] = compare(ens_new, ens_saved, ids, categories)
    else:
        ens_new = ens_saved = None

    for label, res in results.items():
        report.check(res["max"] <= tol, f"{label}: max |diff| {res['max']:.2e} <= {tol:g}",
                     f"mean {res['mean']:.1e}; worst at {res['at'][0]} / {res['at'][1]} "
                     f"(saved {res['saved']:.4f}, new {res['new']:.4f})")
    worst = results[next(reversed(results))]["per_cat"].head(5)
    print("  largest per-category max |diff| (" + next(reversed(results)) + "): "
          + ", ".join(f"{c} {v:.1e}" for c, v in worst.items()))

    if ens_new is not None:
        cutoffs = predict.resolve_thresholds(None, categories, runs)
        flips = (ens_new >= cutoffs) != (ens_saved >= cutoffs)
        rows = [{"chunk_id": ids[r], "category": categories[c], "prob_saved": float(ens_saved[r, c]),
                 "prob_new": float(ens_new[r, c]), "cutoff": float(cutoffs[c])} for r, c in zip(*np.nonzero(flips))]
        label = f"ensemble yes/no at the selected cutoffs agrees on all {flips.size} pairs"
        if used == "bf16":
            print(f"  (info) {len(rows)} yes/no disagreement(s) at the cutoffs")
        else:
            report.check(not rows, label, f"{len(rows)} disagree" if rows else "")
        if rows:
            print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    check_tokenizer(report, model_root, runs[0], chunks["chunk_text"])


# --- C: smoke test ---
def smoke_test(report: Report, raw_text: str, contract_id: str, runs, model_root, device, precision, batch_size):
    print(f"\n=== C. smoke test on an unseen contract: {contract_id} ({len(raw_text):,} chars) ===")
    predict._load_model.cache_clear()            # time a cold load, as a fresh session would see it
    start = time.time()
    predict.load_models(model_root, runs, device)
    load_s = time.time() - start

    start = time.time()
    pred = predict.predict_contract(raw_text, contract_id=contract_id, model_root=model_root, runs=runs,
                                    device=device, precision=precision, batch_size=batch_size)
    total_s = time.time() - start
    chunks = predict.chunk_contract(raw_text, contract_id)
    start = time.time()
    predict.score_chunks(chunks["chunk_text"], model_root=model_root, runs=runs[:1], device=device,
                         precision=precision, batch_size=batch_size)
    single_s = time.time() - start

    n_chunks = len(chunks)
    per_chunk = pred.groupby("chunk_id")["category"].agg(["size", "nunique"])
    report.check(len(pred) == n_chunks * predict.N_CATEGORIES, f"{n_chunks} chunks x 41 categories = {len(pred)} rows")
    report.check(chunks["chunk_id"].is_unique and set(per_chunk.index) == set(chunks["chunk_id"])
                 and (per_chunk["size"] == 41).all() and (per_chunk["nunique"] == 41).all(),
                 "chunk ids unique, each with all 41 categories once")
    report.check(bool(np.isfinite(pred["prob"]).all() and pred["prob"].between(0, 1).all()), "probabilities finite, in [0, 1]")
    report.check(bool((pred["predicted"] == (pred["prob"] >= pred["threshold"])).all()), "predicted == prob >= threshold")
    check_raw_offsets(report, raw_text, chunks, contract_id)

    print(f"  timing on {predict.resolve_device(device)}: model load {load_s:.1f}s ({len(runs)} run(s)); "
          f"predict_contract {total_s:.1f}s ({total_s / n_chunks:.2f}s/chunk, ensemble incl. chunking); "
          f"one run, scoring only {single_s:.1f}s")
    rollup = predict.contract_rollup(pred)
    found = rollup[rollup["contract_predicted"]].sort_values("max_prob", ascending=False)
    print(f"  {len(found)} of 41 categories predicted present (top 10 by max prob):")
    print(found.head(10)[["category", "max_prob", "threshold", "n_chunks_predicted"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-root", type=Path, default=None, help=f"folder holding <run>/model/; default ${predict.MODEL_ROOT_ENV}")
    parser.add_argument("--contracts", nargs="+", default=None, help="validation contract ids (default: fewest/median/most chunks)")
    parser.add_argument("--chunks-df", type=Path, default=None, help="training chunks_df.parquet, for an exact text/offset comparison")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[predict.PREDICT_BATCH_SIZE])
    parser.add_argument("--precision", choices=predict.PRECISIONS, default="auto")
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-probs", action="store_true", help="check A only (no model weights needed)")
    smoke = parser.add_mutually_exclusive_group()
    smoke.add_argument("--smoke-text", type=Path, default=None, help="unseen contract as a UTF-8 .txt (preferred)")
    smoke.add_argument("--smoke-test-json", type=int, default=None, metavar="N",
                       help="use the text of test.json contract N (text only: no labels, no metrics)")
    args = parser.parse_args()

    runs = predict.RUNS_ENSEMBLE
    device = str(predict.resolve_device(args.device))
    print(f"torch {torch.__version__}, transformers {transformers.__version__} (runs trained with "
          f"{TRAINED_WITH['torch']} / {TRAINED_WITH['transformers']}); device {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.startswith("cuda") else ""))

    report = Report()
    val_refs = load_val_probs(runs)
    val_ref = val_refs[runs[0]]
    contracts = args.contracts or pick_contracts(val_ref)
    val_ids = set(load_splits()["val"])
    not_val = [c for c in contracts if c not in val_ids]
    if not_val:
        raise ValueError(f"not validation contracts (only val may be checked against val_probs): {not_val}")
    print("validation contracts: " + "; ".join(f"{c} ({int((val_ref['contract_id'] == c).sum())} chunks)" for c in contracts))

    texts = load_contract_texts(TRAIN_JSON)
    chunks_ref = pd.read_parquet(args.chunks_df) if args.chunks_df else None
    chunks_by_contract = check_chunks(report, contracts, texts, val_ref, chunks_ref)

    if not args.skip_probs:
        for batch_size in args.batch_sizes:
            check_probs(report, chunks_by_contract, val_refs, runs, args.model_root, device, args.precision, batch_size)

        if args.smoke_text is not None:
            smoke_test(report, args.smoke_text.read_text(encoding="utf-8"), args.smoke_text.stem, runs,
                       args.model_root, device, args.precision, args.batch_sizes[0])
        elif args.smoke_test_json is not None:
            title, text = list(load_contract_texts(TEST_JSON).items())[args.smoke_test_json]
            smoke_test(report, text, title, runs, args.model_root, device, args.precision, args.batch_sizes[0])
        else:
            print("\n(no --smoke-text / --smoke-test-json given: smoke test C skipped)")

    print("\n" + ("ALL CHECKS PASSED" if not report.failures else f"{len(report.failures)} CHECK(S) FAILED:"))
    for label in report.failures:
        print(f"  - {label}")
    sys.exit(1 if report.failures else 0)


if __name__ == "__main__":
    main()
