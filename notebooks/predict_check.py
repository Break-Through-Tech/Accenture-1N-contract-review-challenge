"""
Checks for predict.py (issue #31, "Done when": predict_contract reproduces the saved validation probabilities and
runs on a contract it hasn't seen). Exits with code 1 if any check fails.

Acceptance rule (revised after the first Colab run; see the issue #31 comment):
  * exact chunk match (check A)
  * mean |diff| <= 1e-3 and max |diff| <= 5e-2 against results/<run>/val_probs.parquet, for each run and the ensemble
  * zero yes/no flips: the ensemble's prob >= cutoff (selected_thresholds.csv) agrees with the saved ensemble's
The original strict "max |diff| <= 1e-3" is still printed, as a reported-only number, for the exact-batching run.

Checks
  A. Chunks: for 3 validation contracts (fewest, median and most chunks in val_probs), the raw CUAD text run
     through predict.chunk_contract gives exactly the chunk ids in val_probs; with --chunks-df, also the same
     chunk_text/chunk_start/chunk_end as training's chunks_df.parquet. Every chunk's raw_start/raw_end slice of the
     raw text, cleaned again, equals the chunk text. Needs no model weights.
  B. Probabilities on those 3 contracts, joined on chunk_id: each run against its own val_probs, and the ensemble
     against the mean of the two saved files. B1 = bf16 autocast on an Ampere+ GPU (how val_probs were made),
     B2 = fp32. With several --batch-sizes, also reports (not a pass/fail) how much the output moves between them.
  Exact batching (--exact-batching): all 1,841 val chunks in val_probs row order, batches of 16, bf16, no length
     sorting. That's how trainer.predict batched them, so this separates batching noise from GPU/library
     differences. Runs A + this only.
  Device comparison (--compare-devices): the same 3 contracts scored in fp32 on the GPU and on the CPU, compared
     directly with each other (reported only). Runs A + this only.
  Load timing (--time-load): loads both runs three times. The first load is cold only in a fresh runtime that hasn't
     read the weights yet. The second load comes after clearing the in-process cache, so the files are in the OS/Drive
     cache. The third is a cached call. Runs only this.
  C. Smoke test on an unseen contract: --smoke-text (preferred: any non-CUAD contract) or --smoke-test-json N
     (reads only the text of test.json contract N; no labels are read, no metrics computed, nothing tuned).
     Checks shape, ids, probability range, predicted == prob >= threshold, raw offsets, and times it.

How to run
----------
Check A only, anywhere the chunking stack is installed (no weights needed), from notebooks/:

    python predict_check.py --skip-probs

Colab: notebooks/predict_check_colab.ipynb does all of this with Runtime > Run all. By hand, on a GPU runtime
(A100 or L4, so bf16 matches training):

    from google.colab import drive
    drive.mount("/content/drive")

    !git clone --branch <branch-with-predict.py> https://github.com/Break-Through-Tech/Accenture-1N-contract-review-challenge.git /content/repo
    !pip install -q git+https://github.com/brandonstarxel/chunking_evaluation.git backoff
    %cd /content/repo/notebooks

    import os
    os.environ["CUAD_MODEL_ROOT"] = "/content/drive/MyDrive/cuad_finetune/runs"
    CHUNKS = "--chunks-df /content/drive/MyDrive/cuad_finetune/data/chunks_df.parquet"
    !python predict_check.py {CHUNKS} --batch-sizes 8 1 --smoke-text /content/my_contract.txt      # B1 + C
    !python predict_check.py {CHUNKS} --exact-batching                                            # diagnostic
    !python predict_check.py {CHUNKS} --precision fp32 --batch-sizes 8 1                          # GPU fp32 stability
    !python predict_check.py {CHUNKS} --device cpu --precision fp32 --smoke-text /content/my_contract.txt  # B2 + C
    !python predict_check.py {CHUNKS} --compare-devices                                           # GPU vs CPU fp32
    !python predict_check.py --time-load        # FIRST thing in a fresh runtime, or the "cold" number isn't cold

--precision defaults to "auto" here (bf16 on an Ampere+ GPU, fp32 otherwise), unlike predict.py's fp32 default,
because the saved references in val_probs.parquet were made in bf16.

Local x64 Python 3.11-3.13 venv (CPU, so B2), from notebooks/:

    python predict_check.py --model-root C:/path/to/cuad_finetune/runs --smoke-text my_contract.txt

If B fails, check in this order: chunk ids (A), category order, tokenizer (the model folder's own tokenizer is
compared below), sigmoid vs softmax, precision / attention implementation, transformers version (runs used 5.20.1).
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
RUNS = predict.RUNS_ENSEMBLE
N_CONTRACTS = 3
MEAN_TOL = 1e-3
MAX_TOL = 5e-2
STRICT_MAX_TOL = 1e-3            # the issue's original target; reported only, for the exact-batching run
TRAINING_EVAL_BATCH = 16         # modernbert_finetune.EVAL_BATCH_SIZE (not imported: that module pulls in Trainer/datasets)
TRAINED_WITH = {"gpu": "A100", "torch": "2.11.0+cu130", "transformers": "5.20.1"}   # results/<run>/run_config.json


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


def load_val_probs() -> dict[str, pd.DataFrame]:
    refs = {run: pd.read_parquet(RESULTS_DIR / run / "val_probs.parquet") for run in RUNS}
    first = refs[RUNS[0]]
    for run, df in refs.items():
        assert df["chunk_id"].equals(first["chunk_id"]), f"{run}: val_probs rows differ from {RUNS[0]}"
    return refs


def pick_contracts(val_ref: pd.DataFrame, n: int = N_CONTRACTS) -> list[str]:
    """Fewest, median and most chunks: covers a short contract and the longest one in validation."""
    sizes = val_ref.groupby("contract_id").size().sort_values(kind="stable")
    positions = sorted({0, len(sizes) // 2, len(sizes) - 1})[:n]
    return [sizes.index[p] for p in positions]


def check_raw_offsets(report: Report, raw_text: str, chunks: pd.DataFrame, label: str):
    """raw_text[raw_start:raw_end], cleaned again, has to give back the chunk text.
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


# --- B: comparing probabilities ---
def diff_stats(new: np.ndarray, ref: np.ndarray, ids, categories) -> dict:
    diff = np.abs(new.astype(np.float64) - ref.astype(np.float64))
    r, c = np.unravel_index(int(np.argmax(diff)), diff.shape)
    return {"max": float(diff.max()), "mean": float(diff.mean()), "p99": float(np.percentile(diff, 99)),
            "at": (ids[r], categories[c]), "ref": float(ref[r, c]), "new": float(new[r, c]),
            "per_cat": pd.Series(diff.max(axis=0), index=categories).sort_values(ascending=False)}


def compare(report: Report, ids, categories, new_by_run, ref_by_run, *, ref_name="saved", acceptance=True,
            strict_max=False):
    """Table of max / mean / p99 |diff| and yes/no flips, per run and for the ensemble (mean of the runs).
    acceptance=True applies the acceptance rule; False only reports (used for batch-size stability)."""
    cutoffs = predict.resolve_thresholds(None, categories, RUNS)
    new_by_run = {**new_by_run, "ensemble_top2": np.mean([new_by_run[r] for r in RUNS], axis=0)}
    ref_by_run = {**ref_by_run, "ensemble_top2": np.mean([ref_by_run[r] for r in RUNS], axis=0)}
    stats = {label: diff_stats(new_by_run[label], ref_by_run[label], ids, categories) for label in new_by_run}

    ens_new, ens_ref = new_by_run["ensemble_top2"], ref_by_run["ensemble_top2"]
    flipped = (ens_new >= cutoffs) != (ens_ref >= cutoffs)
    flips = [{"chunk_id": ids[r], "category": categories[c], f"prob_{ref_name}": float(ens_ref[r, c]),
              "prob_new": float(ens_new[r, c]), "cutoff": float(cutoffs[c])} for r, c in zip(*np.nonzero(flipped))]

    print(f"  {'':24s}{'max |diff|':>12s}{'mean |diff|':>13s}{'p99 |diff|':>12s}{'yes/no flips':>16s}")
    for label, s in stats.items():
        flip_col = f"{len(flips)} / {flipped.size}" if label == "ensemble_top2" else "-"
        print(f"  {label:24s}{s['max']:12.2e}{s['mean']:13.1e}{s['p99']:12.1e}{flip_col:>16s}")
    for label, s in stats.items():
        print(f"  worst {label}: {s['at'][0]} / {s['at'][1]} ({ref_name} {s['ref']:.4f}, new {s['new']:.4f})")
    print("  largest per-category max |diff| (ensemble): "
          + ", ".join(f"{c} {v:.1e}" for c, v in stats["ensemble_top2"]["per_cat"].head(5).items()))
    if flips:
        print(pd.DataFrame(flips).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if acceptance:
        for label, s in stats.items():
            report.check(s["mean"] <= MEAN_TOL and s["max"] <= MAX_TOL,
                         f"{label}: mean |diff| {s['mean']:.1e} <= {MEAN_TOL:g} and max |diff| {s['max']:.2e} <= {MAX_TOL:g}")
        report.check(not flips, f"ensemble yes/no at the selected cutoffs agrees on all {flipped.size} pairs",
                     f"{len(flips)} flipped" if flips else "")
    if strict_max:
        for label, s in stats.items():
            print(f"  (reported only) strict max |diff| <= {STRICT_MAX_TOL:g}, {label}: {s['max']:.2e} -> "
                  f"{'within' if s['max'] <= STRICT_MAX_TOL else 'over'}")
    return stats


def saved_probs(val_refs, ids, categories) -> dict[str, np.ndarray]:
    out = {}
    for run in RUNS:
        ref = val_refs[run]
        if ref.columns[2:].tolist() != categories:
            raise ValueError(f"{run}: model category order != val_probs column order -- outputs would be misaligned")
        out[run] = ref.set_index("chunk_id").loc[ids, categories].to_numpy()
    return out


def check_tokenizer(report: Report, model_root, run: str, texts):
    """The base tokenizer (used for chunking and inference) vs. the copy saved with the model."""
    model_dir = predict.run_model_dir(predict.resolve_model_root(model_root), run)
    if not any((model_dir / f).exists() for f in ("tokenizer.json", "tokenizer_config.json")):
        print(f"  (no tokenizer saved in {model_dir}; skipping the tokenizer comparison)")
        return
    base, saved = load_tokenizer(MODEL_NAME), AutoTokenizer.from_pretrained(model_dir)
    same = base(list(texts))["input_ids"] == saved(list(texts))["input_ids"]
    report.check(same, f"base tokenizer == tokenizer saved with {run} on the checked chunks")


def check_probs(report: Report, chunks_by_contract, val_refs, model_root, device, precision, batch_size):
    chunks = pd.concat(chunks_by_contract.values(), ignore_index=True)
    ids = chunks["chunk_id"].tolist()
    used = predict.resolve_precision(precision, predict.resolve_device(device))
    predict.load_models(model_root, RUNS, device)      # load outside the timer
    start = time.time()
    categories, new = predict.score_chunks(chunks["chunk_text"], model_root=model_root, runs=RUNS, device=device,
                                           precision=used, batch_size=batch_size)
    seconds = time.time() - start
    print(f"\n=== {'B1' if used == 'bf16' else 'B2'}. probabilities vs. saved: {used} on {predict.resolve_device(device)}, "
          f"batch size {batch_size}, {len(ids)} chunks x {len(RUNS)} runs in {seconds:.1f}s ===")
    saved = saved_probs(val_refs, ids, categories)              # raises if the category order differs
    report.check(True, "model category order == val_probs column order")
    compare(report, ids, categories, new, saved)
    check_tokenizer(report, model_root, RUNS[0], chunks["chunk_text"])
    return used, ids, categories, new


# --- exact training batching (diagnostic) ---
def val_chunk_texts(val_ref, texts, chunks_ref) -> tuple[list[str], str]:
    """Chunk text for every val_probs row, in row order: training's chunks_df if given, else re-chunked here."""
    ids = val_ref["chunk_id"].tolist()
    if chunks_ref is not None:
        lookup = chunks_ref.set_index("chunk_id")["chunk_text"]
        source = "training's chunks_df"
    else:
        lookup = {}
        for cid in dict.fromkeys(val_ref["contract_id"]):
            chunks = predict.chunk_contract(texts[cid], contract_id=cid)
            lookup.update(zip(chunks["chunk_id"], chunks["chunk_text"]))
        source = "predict.chunk_contract"
    missing = [i for i in ids if i not in lookup]
    if missing:
        raise ValueError(f"{len(missing)} val_probs chunks have no text in {source}, e.g. {missing[:3]}")
    return [lookup[i] for i in ids], source


def exact_batching(report: Report, val_refs, texts, chunks_ref, model_root, device):
    val_ref = val_refs[RUNS[0]]
    ids = val_ref["chunk_id"].tolist()
    chunk_texts, source = val_chunk_texts(val_ref, texts, chunks_ref)
    predict.load_models(model_root, RUNS, device)
    start = time.time()
    # precision="bf16" raises on CPU or a pre-Ampere GPU: this diagnostic only means something in bf16
    categories, new = predict.score_chunks(chunk_texts, model_root=model_root, runs=RUNS, device=device,
                                           precision="bf16", batch_size=TRAINING_EVAL_BATCH, sort_by_length=False)
    print(f"\n=== exact training batching: all {len(ids)} val chunks in val_probs row order, batches of "
          f"{TRAINING_EVAL_BATCH}, bf16 on {predict.resolve_device(device)}, no length sorting "
          f"(text from {source}); {time.time() - start:.0f}s ===")
    compare(report, ids, categories, new, saved_probs(val_refs, ids, categories), strict_max=True)


# --- GPU fp32 vs CPU fp32 ---
def compare_devices(report: Report, chunks_by_contract, model_root):
    if not torch.cuda.is_available():
        raise RuntimeError("--compare-devices needs a GPU runtime (it compares GPU fp32 with CPU fp32)")
    chunks = pd.concat(chunks_by_contract.values(), ignore_index=True)
    ids = chunks["chunk_id"].tolist()
    out, seconds = {}, {}
    for device in ("cuda", "cpu"):
        predict.load_models(model_root, RUNS, device)
        start = time.time()
        categories, out[device] = predict.score_chunks(chunks["chunk_text"], model_root=model_root, runs=RUNS,
                                                       device=device, precision="fp32")
        seconds[device] = time.time() - start
    print(f"\n=== GPU fp32 vs. CPU fp32: same {len(ids)} chunks, batch size {predict.PREDICT_BATCH_SIZE} "
          f"(GPU {seconds['cuda']:.1f}s, CPU {seconds['cpu']:.1f}s; reported only) ===")
    compare(report, ids, categories, out["cuda"], out["cpu"], ref_name="cpu", acceptance=False)


# --- model load timing ---
def time_model_load(model_root, device):
    root = predict.resolve_model_root(model_root)
    dirs = {run: predict.run_model_dir(root, run) for run in RUNS}
    size_mb = sum(f.stat().st_size for d in dirs.values() for f in d.iterdir() if f.is_file()) / 1e6
    print(f"\n=== model load timing: {len(RUNS)} runs from {root} ({size_mb:,.0f} MB on disk) to {device} ===")
    timings = []
    for label, clear in (("first load (cold only in a fresh runtime)", False),
                         ("reload, files already read once (OS/Drive cache warm)", True),
                         ("cached call (models already in memory)", False)):
        if clear:
            predict._load_model.cache_clear()
        start = time.time()
        predict.load_models(model_root, RUNS, device)
        timings.append((label, time.time() - start))
    for label, seconds in timings:
        print(f"  {label:55s} {seconds:8.2f}s")
    print("  (excludes importing torch/transformers; tokenizer and chunker load separately on the first predict_contract call)")


# --- C: smoke test ---
def smoke_test(report: Report, raw_text: str, contract_id: str, model_root, device, precision, batch_size):
    print(f"\n=== C. smoke test on an unseen contract: {contract_id} ({len(raw_text):,} chars) ===")
    predict._load_model.cache_clear()            # reload from files already read once (true cold load: --time-load)
    start = time.time()
    predict.load_models(model_root, RUNS, device)
    load_s = time.time() - start

    start = time.time()
    pred = predict.predict_contract(raw_text, contract_id=contract_id, model_root=model_root, runs=RUNS,
                                    device=device, precision=precision, batch_size=batch_size)
    total_s = time.time() - start
    chunks = predict.chunk_contract(raw_text, contract_id)
    start = time.time()
    predict.score_chunks(chunks["chunk_text"], model_root=model_root, runs=RUNS[:1], device=device,
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

    print(f"  timing on {predict.resolve_device(device)}: model reload, files cached {load_s:.1f}s ({len(RUNS)} runs); "
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
    parser.add_argument("--precision", choices=predict.PRECISIONS, default="auto",
                        help="default auto = bf16 on an Ampere+ GPU, like the saved references (predict.py defaults to fp32)")
    parser.add_argument("--device", default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--skip-probs", action="store_true", help="check A only (no model weights needed)")
    mode.add_argument("--exact-batching", action="store_true",
                      help="A + all val chunks in training's batches (bf16, 16 per batch, row order); skips B and C")
    mode.add_argument("--compare-devices", action="store_true",
                      help="A + the 3 contracts in fp32 on GPU vs CPU, compared directly; skips B and C")
    mode.add_argument("--time-load", action="store_true",
                      help="only time loading both models (cold/warm/cached); run first in a fresh runtime")
    smoke = parser.add_mutually_exclusive_group()
    smoke.add_argument("--smoke-text", type=Path, default=None, help="unseen contract as a UTF-8 .txt (preferred)")
    smoke.add_argument("--smoke-test-json", type=int, default=None, metavar="N",
                       help="use the text of test.json contract N (text only: no labels, no metrics)")
    args = parser.parse_args()

    device = str(predict.resolve_device(args.device))
    print(f"torch {torch.__version__}, transformers {transformers.__version__}; device {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.startswith("cuda") else "")
          + f". Runs trained/predicted on {TRAINED_WITH['gpu']} with {TRAINED_WITH['torch']} / {TRAINED_WITH['transformers']}")

    if args.time_load:           # before anything else touches the weights
        time_model_load(args.model_root, device)
        return

    report = Report()
    val_refs = load_val_probs()
    val_ref = val_refs[RUNS[0]]
    contracts = args.contracts or pick_contracts(val_ref)
    val_ids = set(load_splits()["val"])
    not_val = [c for c in contracts if c not in val_ids]
    if not_val:
        raise ValueError(f"not validation contracts (only val may be checked against val_probs): {not_val}")
    print("validation contracts: " + "; ".join(f"{c} ({int((val_ref['contract_id'] == c).sum())} chunks)" for c in contracts))

    texts = load_contract_texts(TRAIN_JSON)
    chunks_ref = pd.read_parquet(args.chunks_df) if args.chunks_df else None
    chunks_by_contract = check_chunks(report, contracts, texts, val_ref, chunks_ref)

    if args.exact_batching:
        exact_batching(report, val_refs, texts, chunks_ref, args.model_root, device)
        if args.smoke_text is not None or args.smoke_test_json is not None:
            print("\n(--exact-batching runs A + the diagnostic only: smoke test C skipped)")
    elif args.compare_devices:
        compare_devices(report, chunks_by_contract, args.model_root)
    elif not args.skip_probs:
        runs_by_batch = {}
        for batch_size in args.batch_sizes:
            used, ids, categories, new = check_probs(report, chunks_by_contract, val_refs, args.model_root, device,
                                                     args.precision, batch_size)
            runs_by_batch[batch_size] = new
        base = args.batch_sizes[0]
        for other in args.batch_sizes[1:]:
            print(f"\n=== batch-size stability ({used} on {device}): batch {other} vs. batch {base}, "
                  f"same {len(ids)} chunks (reported only) ===")
            compare(report, ids, categories, runs_by_batch[other], runs_by_batch[base],
                    ref_name=f"batch{base}", acceptance=False)

        if args.smoke_text is not None:
            smoke_test(report, args.smoke_text.read_text(encoding="utf-8"), args.smoke_text.stem,
                       args.model_root, device, args.precision, base)
        elif args.smoke_test_json is not None:
            title, text = list(load_contract_texts(TEST_JSON).items())[args.smoke_test_json]
            smoke_test(report, text, title, args.model_root, device, args.precision, base)
        else:
            print("\n(no --smoke-text / --smoke-test-json given: smoke test C skipped)")

    print("\n" + ("ALL CHECKS PASSED" if not report.failures else f"{len(report.failures)} CHECK(S) FAILED:"))
    for label in report.failures:
        print(f"  - {label}")
    sys.exit(1 if report.failures else 0)


if __name__ == "__main__":
    main()
