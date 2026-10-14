# Plan: Fine-Tune Initial Transformer Model (issue #19)

**Status:** draft, not yet agreed with all assignees · **Assignees:** Albert, Josiah · **Branch:** `albert-finetune`

**Done when** (from the issue): ModernBERT-base is fine-tuned on the training split, evaluated on
the validation split, and both the model checkpoint and the validation results are saved.

## Starting point (already on `main`)

- Chunks are sized with ModernBERT's own tokenizer: 750-token window, 187-token overlap, 768-token
  model input. No chunk gets cut off (longest is 752 tokens).
- `main_pipeline.ipynb` writes `data/processed/`: 8,549 chunks and 350,509 (chunk, category) labels,
  of which 8,317 (2.37%) are positive.
- Shared code to reuse: `build_chunk_labels_df()` in `chunking_util.py`, and `tokenize_dataset()` in
  `modernbert_tokenizer_pipeline.py`.
- TF-IDF baseline on the same chunks (validation, 82 contracts):

  | Baseline setup | micro F1 | macro F1 |
  |---|---|---|
  | 0.5 cutoff, trained on 260 contracts | 0.186 | 0.266 |
  | Per-category cutoffs tuned for F1 | 0.431 | 0.356 |

## Decisions

| Question | Decision | Why |
|---|---|---|
| Which contracts to use | **260 train / 66 tune / 82 validation**: the same two-step `GroupShuffleSplit` (`random_state=0`) as `threshold_tuning.ipynb`. `test.json` stays untouched. | It matches the baseline's training data. The 66 tune contracts pick the best checkpoint now and the cutoffs later in #21. The split depends only on the set of contracts, so it doesn't change when chunking changes (checked). |
| Categories | All 41, each with its own sigmoid output, plain BCE loss | Project rule. `problem_type="multi_label_classification"` is already set up in PR #25. |
| Handling rare categories | No class weighting or resampling | That's #20. This run is the reference #20 compares against. |
| Starting output bias | Set each category's output bias to its base-rate log-odds, from the training labels | It's a cheap stabilizer: the model starts near the real 2.4% rate instead of 50%. It changes the starting point, not the loss, so it doesn't overlap with #20. |
| Main comparison metric | **Average precision (AP)**, which needs no cutoff | The baseline trains on balanced 50/50 samples, so its probabilities run high, while ModernBERT sees the real rates. A 0.5 cutoff means different things for each model, so F1 at 0.5 is secondary. Comparing with tuned cutoffs waits for #21/#22. |
| Macro averages | Report the 41-category macro (matches existing numbers) **and** a macro over categories with positives | `Unlimited/All-You-Can-Eat-License` has 0 positives in validation, and sklearn scores it as 0 for both F1 and AP without warning. |
| Choosing the best checkpoint | Highest macro AP on the tune set (over categories with positives) | Loss is dominated by the 97.6% negatives and the common categories. AP counts every category equally. |
| Where to train | Colab GPU, using bf16 if supported (L4/A100), otherwise fp16 (T4) | ~5,400 training chunks of up to 768 tokens is too slow on a CPU. A T4 can't do bf16. |
| Training data on Colab | Copy `chunks_df.parquet` (9.9 MB) and `chunk_labels_df.parquet` (0.3 MB) from `data/processed/` to Drive and train from those. Don't rebuild chunks on Colab. | `chunking_evaluation` pulls in `openai`, `anthropic` and `chromadb` when imported. The parquet files are exactly what the baseline used. |
| Where outputs go | Final model weights (~600 MB) → Drive. Metrics and predicted probabilities → committed under `results/` | Too big for Git. Saving the probabilities means #21 and #22 don't need to re-run the model. Trainer checkpoints include optimizer state (~1.8 GB each), so keep `save_total_limit=1`. |

## Steps

**0. Fix branch tracking** (Albert). `albert-finetune` currently tracks `origin/main`:
`git branch --unset-upstream`, then `git push -u origin albert-finetune`.

**1. Save the contract splits** (Albert). New `notebooks/splits.py` writes
`data/splits/{train,tune,val}_contracts.txt` using the threshold notebook's split. It asserts
260 / 66 / 82 contracts and that no contract appears in two splits.

**2. Shared metrics function** (Josiah). New `notebooks/eval_metrics.py` with
`multilabel_metrics(y_true, scores, categories, threshold=0.5)`. It returns:
- micro and macro AP, with macro over all 41 categories and over categories with positives;
- micro and macro precision/recall/F1 at the cutoff, both ways;
- a per-category table.

The baseline, ModernBERT and #22 all use it.

**3. Save the baseline's scores** (Josiah, check with Dunni first since it's her notebook). Add one
cell to `threshold_tuning.ipynb` that saves `results/baseline/{tune,val}_probs.parquet` and prints
AP using `multilabel_metrics`.

**4. Training script** (Albert). New `notebooks/modernbert_finetune.py`:
- **Data:** load `chunks_df` and `chunk_labels_df` from `data/processed/`, asserting 8,549 chunks
  and 350,509 rows. Apply the split files and build one row per chunk with 41 label columns.
- **Tokenization:** reuse `tokenize_dataset()`, after changing it to `padding=False` so each
  training batch is padded only to its own longest chunk (it currently pads every row to ~750).
- **Model:** `AutoModelForSequenceClassification` with 41 labels, label names set, and
  `model.classifier.bias` set to each category's log-odds.
- **Training settings:**

  | Setting | Value |
  |---|---|
  | Learning rate | 5e-5, 10% warmup, linear decay |
  | Weight decay | 0.01 |
  | Epochs | 3 |
  | Batch size | 8, with gradient accumulation of 2 (effective 16) |
  | Evaluate and save | After each epoch, on the tune set |
  | Best checkpoint | Highest macro AP (categories with positives), `greater_is_better=True` |
  | Checkpoints kept | `save_total_limit=1` |
  | Precision | bf16 if supported, otherwise fp16 |
  | Seed | 0 |

- **Outputs:** final weights to Drive. In `results/modernbert_v1/`:
  - `tune_probs.parquet` and `val_probs.parquet`
  - `metrics.json` and `per_category.csv`
  - `run_config.json`: settings, git commit, package versions and input row counts
- **`--smoke` flag:** about 50 chunks, 1 epoch, CPU only, writing to a gitignored folder.

**5. Local smoke test** (Albert). `python modernbert_finetune.py --smoke` runs the whole flow in a
few minutes without a GPU.

**6. Colab notebook** (Albert). `notebooks/modernbert_finetune_colab.ipynb`:
1. Mount Drive and clone the repo.
2. `pip install transformers accelerate datasets` with no version pins. Don't reinstall Colab's
   torch or numpy: `requirements.txt` pins `numpy<2` and `torch<2.9` for Intel Macs, which would
   break Colab's preinstalled packages.
3. Copy the two parquet files from Drive into `data/processed/`.
4. Run about 20 steps to check the loss stays finite and measure speed, then start the full run.

**7. Full training run** (Albert). After epoch 1, micro AP on tune should be well above the 0.024
base rate. If it isn't, stop and investigate before using more GPU time.

**8. Write up and open a PR** (Albert and Josiah). Fill in:

| | micro AP | macro AP (categories with positives) | micro F1 @0.5 | macro F1 @0.5 |
|---|---|---|---|---|
| Baseline (260 train contracts) | from step 3 | from step 3 | 0.186 | 0.266 |
| ModernBERT v1 | | | | |

In the PR, also note:
- F1 at 0.5 isn't a fair comparison between these two models (see Decisions).
- The three-way split (the issue only mentions train and validation).
- The tune set is used for both checkpoint selection and #21's cutoffs, so tune scores will run
  slightly high. Validation is untouched.

## Risks

- **Rare categories:** at a 0.5 cutoff they will likely get F1 = 0, because their probabilities
  never reach 0.5. That's expected for this version; AP shows whether they're still being learned.
  #20 and #21 address it.
- **fp16 on T4:** no confirmed ModernBERT problem was found. The 20-step check in step 6 catches a
  NaN loss early; if one appears, fall back to fp32.
- **Out of GPU memory:** drop to a batch size of 4 with gradient accumulation of 4.
- **Colab disconnects:** resume from the checkpoint on Drive.
- **Training time:** not measured yet. Use the step-6 speed check before planning around it.

## Files

| New | Changed |
|---|---|
| `notebooks/splits.py`, `data/splits/*.txt` | `notebooks/modernbert_tokenizer_pipeline.py` (`padding=False`) |
| `notebooks/eval_metrics.py` | `notebooks/threshold_tuning.ipynb` (one cell, step 3) |
| `notebooks/modernbert_finetune.py` | `.gitignore` (local smoke-run folder) |
| `notebooks/modernbert_finetune_colab.ipynb` | |
| `results/baseline/`, `results/modernbert_v1/` | |
