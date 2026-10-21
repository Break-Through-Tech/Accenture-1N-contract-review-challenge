"""
Fine-tune ModernBERT-base on CUAD chunks: 41-category multi-label clause classification
(issue #19 = plain v1, issue #20 = the imbalance options below). Plan: docs/issue-19-modernbert-finetune-plan.md

Inputs: data/processed/{chunks_df,chunk_labels_df}.parquet (written by main_pipeline.ipynb) and
data/splits/*.txt (written by splits.py). Doesn't rebuild chunks, so it runs on Colab without the
chunking stack and trains on exactly the chunks the TF-IDF baseline used.

Run from notebooks/:

    python modernbert_finetune.py --smoke                                 # tiny CPU run, every code path
    python modernbert_finetune.py --check                                 # ~20 real steps: loss finite + speed estimate
    python modernbert_finetune.py --run-name modernbert_v1                # #19: plain
    python modernbert_finetune.py --run-name modernbert_v2_biasinit  --bias-init
    python modernbert_finetune.py --run-name modernbert_v3_posweight --pos-weight sqrt
    python modernbert_finetune.py --run-name modernbert_v4_oversample --oversample-rare 50

On Colab, add --output-dir pointing into Google Drive (checkpoints + final model + results go there).
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# imported before any parquet is read: it switches pandas to plain python strings, which
# datasets.Dataset.from_pandas needs on pandas 3 (and its tokenize_dataset is reused below)
from modernbert_tokenizer_pipeline import tokenize_dataset

import datasets
import sklearn
import transformers
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from chunking_util import MAX_LENGTH, MODEL_NAME
from eval_metrics import multilabel_metrics
from splits import SPLIT_NAMES, load_splits

NOTEBOOK_DIR = Path(__file__).resolve().parent
REPO_DIR = NOTEBOOK_DIR.parent
PROCESSED_DIR = REPO_DIR / "data" / "processed"
RUNS_DIR = NOTEBOOK_DIR / "finetune_runs"      # local checkpoints/models (gitignored)
RESULTS_DIR = REPO_DIR / "results"              # committed: metrics + predicted probabilities

# what main_pipeline.ipynb produces with the ModernBERT chunker -- anything else is stale data
EXPECTED_N_CHUNKS = 8549
EXPECTED_N_LABEL_ROWS = 350509
N_CATEGORIES = 41

# training settings (docs/issue-19-modernbert-finetune-plan.md, step 4)
LEARNING_RATE = 5e-5
WARMUP_FRACTION = 0.1
WEIGHT_DECAY = 0.01
EPOCHS = 3
BATCH_SIZE = 8
GRAD_ACCUM = 2                   # effective batch of 16
EVAL_BATCH_SIZE = 16
SEED = 0
BEST_METRIC = "macro_ap_present"  # picked on the tune set; see eval_metrics.py for why AP
EPOCH1_MIN_MICRO_AP = 0.05        # ~2x the 0.024 base rate; below this after epoch 1, something's wrong
OVERSAMPLE_FACTOR = 3             # --oversample-rare: rare-category chunks appear 3x in training
CHECK_STEPS = 20
SMOKE_SIZES = {"train": 48, "tune": 16, "val": 16}
SMOKE_EPOCHS = 2                  # 2, not 1, so picking the best of several checkpoints gets exercised


# --- data: chunk table + 41 label columns, split by contract ---
def load_chunk_table(data_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    chunks_df = pd.read_parquet(data_dir / "chunks_df.parquet")
    chunk_labels_df = pd.read_parquet(data_dir / "chunk_labels_df.parquet")
    if len(chunks_df) != EXPECTED_N_CHUNKS or len(chunk_labels_df) != EXPECTED_N_LABEL_ROWS:
        raise ValueError(
            f"{data_dir} has {len(chunks_df)} chunks / {len(chunk_labels_df)} label rows, expected "
            f"{EXPECTED_N_CHUNKS} / {EXPECTED_N_LABEL_ROWS} -- stale data, re-run main_pipeline.ipynb"
        )
    chunks_df = chunks_df[chunks_df["chunk_start"] >= 0]

    label_matrix = chunk_labels_df.pivot_table(
        index="chunk_id", columns="category", values="label", aggfunc="max", fill_value=0
    ).astype(int)
    categories = label_matrix.columns.tolist()
    assert len(categories) == N_CATEGORIES, "project rules say predict all 41 CUAD categories"

    table = chunks_df.set_index("chunk_id")[["contract_id", "chunk_text"]].join(label_matrix, how="inner")
    assert len(table) == len(chunks_df), "some chunks have no labels -- chunks_df and chunk_labels_df disagree"
    return table.reset_index(), categories


def split_chunk_table(table: pd.DataFrame, splits: dict[str, list[str]]) -> dict[str, pd.DataFrame]:
    parts = {name: table[table["contract_id"].isin(set(splits[name]))].reset_index(drop=True)
             for name in SPLIT_NAMES}
    # every chunk lands in exactly one split -- the split files and the chunk table must agree
    if sum(len(p) for p in parts.values()) != len(table):
        raise ValueError("some chunks belong to no split -- data/splits/ and data/processed/ disagree")
    return parts


def oversample_rare(train_df: pd.DataFrame, categories: list[str], min_positives: int,
                    factor: int = OVERSAMPLE_FACTOR) -> tuple[pd.DataFrame, list[str], int]:
    """Repeat training chunks that contain any category with fewer than min_positives positive chunks.
    Training data only -- tune/val are never resampled, or their scores would stop meaning anything."""
    pos = train_df[categories].sum()
    rare = pos[pos < min_positives].index.tolist()
    mask = train_df[rare].any(axis=1) if rare else pd.Series(False, index=train_df.index)
    extra = pd.concat([train_df[mask]] * (factor - 1), ignore_index=True) if mask.any() else train_df.iloc[:0]
    return pd.concat([train_df, extra], ignore_index=True), rare, int(len(extra))


def to_token_dataset(df: pd.DataFrame, categories: list[str], tokenizer) -> Dataset:
    # generic label_0..label_40 column names: category names have slashes and spaces in them
    label_cols = [f"label_{i}" for i in range(len(categories))]
    renamed = df[["chunk_text", *categories]].rename(columns=dict(zip(categories, label_cols)))
    dataset = Dataset.from_pandas(renamed, preserve_index=False)
    # raises if any chunk is longer than the 768-token model input (it shouldn't be -- see chunking_util)
    return tokenize_dataset(tokenizer, dataset, label_cols, max_length=MAX_LENGTH)


# --- the #20 options ---
def base_rate_log_odds(train_df: pd.DataFrame, categories: list[str]) -> np.ndarray:
    rate = train_df[categories].mean().to_numpy().clip(1e-4, 1 - 1e-4)
    return np.log(rate / (1 - rate))


def sqrt_pos_weight(train_df: pd.DataFrame, categories: list[str]) -> np.ndarray:
    # sqrt of negatives/positives: 4-20 here (median ~7). the full ratio (16-416) would push rare
    # categories to predict "yes" almost everywhere
    pos = train_df[categories].sum().to_numpy().clip(min=1)
    neg = len(train_df) - pos
    return np.sqrt(neg / pos)


class MultiLabelTrainer(Trainer):
    """Same BCE-with-logits loss for every run (pos_weight=None for v1/v2), averaged over every
    (chunk, category) pair in the micro-batch.

    Owning the loss here, instead of using the model's built-in one, keeps all runs on the same
    footing: Trainer only divides the loss by gradient_accumulation_steps when it thinks the model
    does *not* take loss kwargs, which depends on the model class and the transformers version.
    Setting model_accepts_loss_kwargs = False makes it always divide, so every run optimizes the
    same plain mean."""

    def __init__(self, *args, pos_weight: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits.float()     # loss in fp32 even under fp16/bf16 training
        pos_weight = None if self.pos_weight is None else self.pos_weight.to(logits.device)
        loss = F.binary_cross_entropy_with_logits(logits, labels.float(), pos_weight=pos_weight)
        return (loss, outputs) if return_outputs else loss


# --- guard rails while training ---
class StopOnBadLoss(TrainerCallback):
    """A NaN/inf training loss means the run is broken (e.g. fp16 overflow) -- stop instead of
    burning GPU time on it."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        loss = (logs or {}).get("loss")
        if loss is not None and not math.isfinite(loss):
            raise RuntimeError(f"training loss became {loss} at step {state.global_step} -- if this is fp16 "
                               f"on a T4, re-run with --fp32")


class Epoch1Check(TrainerCallback):
    """After the first evaluation, micro AP on tune should be well above the ~0.024 base rate.
    If it isn't, the model isn't learning -- stop and investigate before spending 2 more epochs."""

    def __init__(self, min_micro_ap: float):
        self.min_micro_ap = min_micro_ap
        self.done = False

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.done or not metrics:
            return
        self.done = True
        micro_ap = metrics.get("eval_micro_ap", 0.0)
        print(f"\n[epoch-1 check] tune micro AP = {micro_ap:.3f} (needs >= {self.min_micro_ap})")
        if micro_ap < self.min_micro_ap:
            raise RuntimeError(f"tune micro AP after epoch 1 is {micro_ap:.3f}, barely above the ~0.024 "
                               f"base rate -- the model isn't learning. Stopped before wasting more GPU time.")


def make_compute_metrics(categories: list[str]):
    def compute_metrics(eval_pred):
        logits = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
        probs = torch.sigmoid(torch.as_tensor(logits, dtype=torch.float32)).numpy()
        summary, _ = multilabel_metrics(eval_pred.label_ids, probs, categories)
        return summary
    return compute_metrics


# --- environment + bookkeeping ---
def pick_precision(force_fp32: bool) -> tuple[str, str]:
    """bf16 on Ampere or newer (L4, A100), fp16 on older GPUs (T4), fp32 on CPU. Uses the compute
    capability, not torch.cuda.is_bf16_supported(), which can report True on a T4 via slow emulation."""
    if not torch.cuda.is_available():
        return "fp32", f"cpu ({platform.processor() or platform.machine()})"
    device = torch.cuda.get_device_name(0)
    if force_fp32:
        return "fp32", device
    major, _ = torch.cuda.get_device_capability(0)
    return ("bf16" if major >= 8 else "fp16"), device


def git_info() -> dict:
    def run(*cmd):
        try:
            return subprocess.run(cmd, cwd=REPO_DIR, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None
    status = run("git", "status", "--porcelain")
    return {"commit": run("git", "rev-parse", "HEAD"), "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": None if status is None else bool(status)}


def save_probs(path: Path, df: pd.DataFrame, probs: np.ndarray, categories: list[str]) -> None:
    out = pd.DataFrame(probs, columns=categories)
    out.insert(0, "contract_id", df["contract_id"].to_numpy())
    out.insert(0, "chunk_id", df["chunk_id"].to_numpy())
    out.to_parquet(path, index=False)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", default="modernbert_v1", help="names the output folders")
    parser.add_argument("--bias-init", action="store_true", help="#20: start each output bias at its base-rate log-odds")
    parser.add_argument("--pos-weight", choices=["sqrt"], default=None, help="#20: class weighting in the BCE loss")
    parser.add_argument("--oversample-rare", type=int, default=None, metavar="N",
                        help=f"#20: repeat ({OVERSAMPLE_FACTOR}x) training chunks with a category that has < N positives")
    parser.add_argument("--epochs", type=float, default=None,
                        help=f"default {EPOCHS} ({SMOKE_EPOCHS} with --smoke)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=GRAD_ACCUM)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--fp32", action="store_true", help="disable fp16/bf16 (fallback if the loss goes NaN)")
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help=f"checkpoints + final model (+ results unless --results-dir); default {RUNS_DIR}/<run-name>")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="metrics + probabilities; default results/<run-name> for local runs, <output-dir>/results otherwise")
    parser.add_argument("--resume", action="store_true", help="continue from the last checkpoint in --output-dir")
    parser.add_argument("--overwrite", action="store_true", help="allow replacing existing results for this run name")
    parser.add_argument("--smoke", action="store_true", help="tiny CPU-friendly run that exercises every code path")
    parser.add_argument("--check", action="store_true", help="train a few real steps: loss finite + time estimate, saves nothing")
    parser.add_argument("--check-steps", type=int, default=CHECK_STEPS)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs is None:
        args.epochs = SMOKE_EPOCHS if args.smoke else EPOCHS
    run_name = args.run_name + ("_smoke" if args.smoke else "")
    output_dir = args.output_dir or RUNS_DIR / run_name
    if args.results_dir is not None:
        results_dir = args.results_dir
    elif args.smoke or args.check or args.output_dir is not None:
        results_dir = output_dir / "results"
    else:
        results_dir = RESULTS_DIR / run_name
    if not args.check and (results_dir / "metrics.json").exists() and not args.overwrite:
        raise FileExistsError(f"{results_dir} already has results -- pick a new --run-name or pass --overwrite")

    precision, device = pick_precision(args.fp32)
    print(f"device: {device} | precision: {precision} | transformers {transformers.__version__} | torch {torch.__version__}")

    # --- data ---
    table, categories = load_chunk_table(args.data_dir)
    splits = load_splits()
    parts = split_chunk_table(table, splits)
    if args.smoke:
        parts = {name: df.sample(n=min(len(df), SMOKE_SIZES[name]), random_state=SEED).reset_index(drop=True)
                 for name, df in parts.items()}
    train_df = parts["train"]
    oversample_info = None
    if args.oversample_rare is not None:
        train_df, rare, n_extra = oversample_rare(train_df, categories, args.oversample_rare)
        oversample_info = {"min_positives": args.oversample_rare, "factor": OVERSAMPLE_FACTOR,
                           "rare_categories": rare, "extra_rows": n_extra}
        print(f"oversampling: {len(rare)} categories with < {args.oversample_rare} positives, "
              f"+{n_extra} repeated training chunks")
    for name in SPLIT_NAMES:
        df = train_df if name == "train" else parts[name]
        print(f"{name:>5}: {len(df):5d} chunks, {df['contract_id'].nunique():3d} contracts, "
              f"{int(df[categories].to_numpy().sum()):5d} positive pairs")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = to_token_dataset(train_df, categories, tokenizer)
    tune_ds = to_token_dataset(parts["tune"], categories, tokenizer)
    val_ds = to_token_dataset(parts["val"], categories, tokenizer)

    # --- model (seeded first, so the randomly initialized classifier head is reproducible) ---
    transformers.set_seed(SEED)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(categories),
        problem_type="multi_label_classification",
        id2label=dict(enumerate(categories)),
        label2id={c: i for i, c in enumerate(categories)},
    )
    bias_init = None
    if args.bias_init:
        bias_init = base_rate_log_odds(train_df, categories)
        with torch.no_grad():
            model.classifier.bias.copy_(torch.as_tensor(bias_init, dtype=model.classifier.bias.dtype))
        print(f"bias init: log-odds from {bias_init.min():.2f} to {bias_init.max():.2f}")
    pos_weight = None
    if args.pos_weight == "sqrt":
        pos_weight = sqrt_pos_weight(train_df, categories)
        print(f"pos_weight (sqrt): {pos_weight.min():.1f} to {pos_weight.max():.1f}, median {np.median(pos_weight):.1f}")

    # --- training setup ---
    steps_per_epoch = math.ceil(math.ceil(len(train_ds) / args.batch_size) / args.grad_accum)
    total_steps = args.check_steps if args.check else math.ceil(steps_per_epoch * args.epochs)
    warmup_steps = max(1, round(WARMUP_FRACTION * total_steps))
    evaluate = not args.check
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=args.check_steps if args.check else -1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=warmup_steps,
        lr_scheduler_type="linear",
        eval_strategy="epoch" if evaluate else "no",
        save_strategy="epoch" if evaluate else "no",
        load_best_model_at_end=evaluate,
        metric_for_best_model=BEST_METRIC,
        greater_is_better=True,
        save_total_limit=1,
        save_only_model=args.smoke,            # smoke: skip the ~1.2 GB of optimizer state per checkpoint
        logging_strategy="steps",
        logging_steps=1 if (args.smoke or args.check) else 20,
        bf16=precision == "bf16",
        fp16=precision == "fp16",
        seed=SEED,
        report_to="none",
        dataloader_num_workers=0,
        dataloader_pin_memory=torch.cuda.is_available(),   # pinning only helps when copying to a GPU
    )
    callbacks = [StopOnBadLoss()]
    if not (args.smoke or args.check):
        callbacks.append(Epoch1Check(EPOCH1_MIN_MICRO_AP))
    trainer = MultiLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=tune_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=make_compute_metrics(categories),
        callbacks=callbacks,
        pos_weight=None if pos_weight is None else torch.as_tensor(pos_weight, dtype=torch.float32),
    )

    print(f"\ntraining {run_name}: {len(train_ds)} chunks, {steps_per_epoch} optimizer steps/epoch, "
          f"{total_steps} total, {warmup_steps} warmup")
    start = time.time()
    train_result = trainer.train(resume_from_checkpoint=True if args.resume else None)
    train_seconds = time.time() - start

    if args.check:
        sec_per_step = train_seconds / args.check_steps
        full_steps = steps_per_epoch * EPOCHS
        print(f"\n[check] {args.check_steps} steps in {train_seconds:.0f}s = {sec_per_step:.1f}s/step, final loss "
              f"{train_result.training_loss:.4f} (finite)")
        print(f"[check] a full {EPOCHS}-epoch run is {full_steps} steps = roughly {sec_per_step * full_steps / 60:.0f} min "
              f"of training, plus ~{EPOCHS + 1} evaluation passes (first steps include warm-up, so this runs a bit high)")
        return

    # --- final model + predictions on tune and validation ---
    model_dir = output_dir / "model"
    trainer.save_model(str(model_dir))
    results_dir.mkdir(parents=True, exist_ok=True)
    summaries, per_category = {}, []
    for name, ds in (("tune", tune_ds), ("val", val_ds)):
        pred = trainer.predict(ds)
        logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        probs = torch.sigmoid(torch.as_tensor(logits, dtype=torch.float32)).numpy()
        save_probs(results_dir / f"{name}_probs.parquet", parts[name], probs, categories)
        summary, per_cat = multilabel_metrics(parts[name][categories].to_numpy(), probs, categories)
        summaries[name] = summary
        per_category.append(per_cat.assign(split=name))
    pd.concat(per_category, ignore_index=True).to_csv(results_dir / "per_category.csv", index=False)

    metrics = {
        "run_name": run_name,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        f"best_tune_{BEST_METRIC}": trainer.state.best_metric,
        "tune": summaries["tune"],
        "val": summaries["val"],
    }
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    run_config = {
        "run_name": run_name,
        "options": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "model": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "training": {"learning_rate": args.learning_rate, "warmup_steps": warmup_steps, "weight_decay": WEIGHT_DECAY,
                     "epochs": args.epochs, "batch_size": args.batch_size, "grad_accum": args.grad_accum,
                     "effective_batch": args.batch_size * args.grad_accum, "seed": SEED,
                     "best_metric": BEST_METRIC, "precision": precision},
        "device": device,
        "data": {name: {"chunks": len(train_df if name == "train" else parts[name]),
                        "contracts": int((train_df if name == "train" else parts[name])["contract_id"].nunique())}
                 for name in SPLIT_NAMES},
        "oversampling": oversample_info,
        "bias_init": None if bias_init is None else dict(zip(categories, bias_init.round(4).tolist())),
        "pos_weight": None if pos_weight is None else dict(zip(categories, pos_weight.round(3).tolist())),
        "train_seconds": round(train_seconds, 1),
        "git": git_info(),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__, "datasets": datasets.__version__,
                     "sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "log_history": trainer.state.log_history,
    }
    (results_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str), encoding="utf-8")

    # checkpoints carry optimizer state (~1.8 GB each) and are only needed to resume an unfinished run
    shutil.rmtree(output_dir / "checkpoints", ignore_errors=True)

    print(f"\n=== {run_name} ===  (trained {train_seconds / 60:.1f} min)")
    print(f"{'':8s}{'micro AP':>9s}{'macro AP':>9s}{'macro AP*':>10s}{'micro F1':>9s}{'macro F1':>9s}{'macro F1*':>10s}")
    for name, s in summaries.items():
        print(f"{name:8s}{s['micro_ap']:9.3f}{s['macro_ap_all']:9.3f}{s['macro_ap_present']:10.3f}"
              f"{s['micro_f1']:9.3f}{s['macro_f1_all']:9.3f}{s['macro_f1_present']:10.3f}")
    print("(* = only categories with at least one positive; F1 at a 0.5 cutoff)")
    print(f"model -> {model_dir}\nresults -> {results_dir}")


if __name__ == "__main__":
    main()
