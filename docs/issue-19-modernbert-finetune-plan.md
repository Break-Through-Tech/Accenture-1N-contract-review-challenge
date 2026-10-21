# Plan: Fine-tune ModernBERT (issue #19) and handle imbalance (issue #20)

**Owner:** Albert (both issues) · **Branch:** `albert-finetune` · **Status:** in progress

| Issue | Done when (from the issue) |
|---|---|
| #19 Fine-Tune Initial Transformer Model | ModernBERT-base is fine-tuned on the training split, evaluated on the validation split, and the model checkpoint and validation results are saved. |
| #20 Multi-Label Training & Imbalance Handling | Sigmoid outputs + BCE loss are in place, class weighting and/or sampling has been tried, and the runs are compared against the #19 model (v1). |

**Not in scope:** comparing against the TF-IDF baseline (scoring the baseline with the same metrics,
the metrics table and the comparison chart). That's #22 (Dunni, Arielle). This work hands #22 the
saved probabilities and a shared metrics function so it doesn't need to re-run any model. Choosing
cutoffs is #21 (Dunni).

## Starting point (already on `main`)

- Chunks are sized with ModernBERT's own tokenizer: 750-token window, 187-token overlap, 768-token
  model input. No chunk gets cut off (longest is 752 tokens).
- `main_pipeline.ipynb` writes `data/processed/`: 8,549 chunks and 350,509 (chunk, category) labels,
  of which 8,317 (2.37%) are positive.
- Shared code to reuse: `build_chunk_labels_df()` in `chunking_util.py`, and `tokenize_dataset()` in
  `modernbert_tokenizer_pipeline.py`.
- Imbalance in the 260 training contracts (5,422 chunks): positives per category range from 13
  (Price Restrictions) to 320 (Parties), median 101. Negatives-to-positives ratios range from 16:1
  to 416:1, median 53:1.

## Decisions

| Question | Decision | Why |
|---|---|---|
| Which contracts to use | **260 train / 66 tune / 82 validation**: the same two-step `GroupShuffleSplit` (`random_state=0`) as `threshold_tuning.ipynb`. `test.json` stays untouched. | Matches the baseline's training data. The tune set picks checkpoints and the best #20 setup, and later #21's cutoffs. The split depends only on the set of contracts, so it doesn't change when chunking changes (checked). |
| Categories and loss | All 41, each with its own sigmoid output, BCE loss (`problem_type="multi_label_classification"`, already set up in PR #25) | Project rule, and #20's first task. |
| v1 (#19) | **Plain**: no class weighting, no resampling, default output-layer start | v1 is the reference #20 compares against, so each #20 change is measured on its own. |
| Main metric | **Average precision (AP)**, micro and macro, which needs no cutoff. F1 at a 0.5 cutoff is reported too. | Cutoffs are #21's job. Class weighting shifts probabilities upward, so F1 at a fixed 0.5 would partly measure that shift rather than real improvement. AP doesn't. |
| Macro averages | Report the 41-category macro **and** a macro over categories with positives | `Unlimited/All-You-Can-Eat-License` has 0 positives in validation, and sklearn scores it as 0 for both F1 and AP without warning. |
| Choosing a checkpoint, and choosing between #20 setups | Highest macro AP on the **tune** set (over categories with positives) | Loss is dominated by the 97.6% negatives and the common categories, while AP counts every category equally. Choosing on tune keeps validation untouched, so it stays a fair final score. |
| Where to train | **Colab GPU**: bf16 if supported (L4/A100), otherwise fp16 (T4). Local CPU is the fallback. | #20 needs several runs. Locally a run takes roughly 6–7 hours (extrapolated from the smoke test's 0.74 chunks/sec on a 24-core CPU), so only one could fit in a day. |
| Training data on Colab | Copy `chunks_df.parquet` (9.9 MB) and `chunk_labels_df.parquet` (0.3 MB) from `data/processed/` to Drive and train from those. Don't rebuild chunks on Colab. | `chunking_evaluation` pulls in `openai`, `anthropic` and `chromadb` when imported. The parquet files are exactly what the baseline used. |
| Where outputs go | Final model weights (~600 MB each) → Drive. Metrics and predicted probabilities → committed under `results/<run name>/` | Too big for Git. Saved probabilities let #21 and #22 work without re-running any model. Trainer checkpoints include optimizer state (~1.8 GB each), so keep `save_total_limit=1`. |

## Runs

All runs share the settings in step 4 and differ only in the options below.

| Run | Issue | What changes from v1 | Option |
|---|---|---|---|
| **v1** `modernbert_v1` | #19 | Nothing (plain) | none |
| **v2** `modernbert_v2_biasinit` | #20 | Each category's output bias starts at its base-rate log-odds from the training labels, so the model starts near the real ~2.4% rate instead of 50%. The loss is unchanged. | `--bias-init` |
| **v3** `modernbert_v3_posweight` | #20 | Class weighting: BCE `pos_weight` = √(negatives ÷ positives) per category, giving weights of 4–20 (median 7). Full ratios (16–416) would push rare categories to predict "yes" almost everywhere. | `--pos-weight sqrt` |
| v4 (stretch) | #20 | Sampling: oversample training chunks that contain any category with fewer than 50 positives | `--oversample-rare 50` |

v4 only runs if time allows. The issue asks for "class weighting and/or sampling," so v3 alone
covers it. The best of v2–v4 by tune macro AP can be combined in one more run if time allows.

## Steps

### Part 1: issue #19 (v1)

**1. Save the contract splits.** New `notebooks/splits.py` writes
`data/splits/{train,tune,val}_contracts.txt` using the threshold notebook's split. It asserts
260 / 66 / 82 contracts and that no contract appears in two splits.

**2. Metrics function.** New `notebooks/eval_metrics.py` with
`multilabel_metrics(y_true, scores, categories, threshold=0.5)`. It returns:
- micro and macro AP, with macro over all 41 categories and over categories with positives;
- micro and macro precision/recall/F1 at the cutoff, both ways;
- a per-category table.

All runs use it, and #22 can reuse it to score the baseline.

**3. Tokenization fix.** Change `tokenize_dataset()` in `modernbert_tokenizer_pipeline.py` to
`padding=False`, so each training batch is padded only to its own longest chunk (it currently pads
every row to ~750). Re-run `python modernbert_tokenizer_pipeline.py` to make sure Josiah's smoke
test still passes.

**4. Training script.** New `notebooks/modernbert_finetune.py`:
- **Data:** load `chunks_df` and `chunk_labels_df` from `data/processed/`, asserting 8,549 chunks
  and 350,509 rows. Apply the split files and build one row per chunk with 41 label columns.
- **Model:** `AutoModelForSequenceClassification` with 41 labels and label names set.
- **Options for #20** (all off by default, so v1 is plain):
  - `--bias-init`: set `model.classifier.bias` to each category's log-odds.
  - `--pos-weight sqrt`: a small `Trainer` subclass whose `compute_loss` uses
    `BCEWithLogitsLoss(pos_weight=...)`. The default multi-label loss has no `pos_weight`.
  - `--oversample-rare N`: repeat training chunks containing a rare category. Training data only;
    tune and validation are never resampled.
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
  | Precision | bf16 if supported, otherwise fp16, fp32 on CPU |
  | Seed | 0 |

- **Outputs:** final weights to Drive. In `results/<run name>/`:
  - `tune_probs.parquet` and `val_probs.parquet`
  - `metrics.json` and `per_category.csv`, for both tune and validation
  - `run_config.json`: options, settings, git commit, package versions and input row counts
- **`--smoke` flag:** about 50 chunks, 1 epoch, CPU only, writing to a gitignored folder.

**5. Local smoke test.** Run `python modernbert_finetune.py --smoke` once plain and once with each
#20 option, so every code path is checked before using GPU time.

**6. Colab notebook.** `notebooks/modernbert_finetune_colab.ipynb`:
1. Mount Drive and clone the repo (it's public, so no login is needed).
2. `pip install transformers accelerate datasets` with no version pins. Don't reinstall Colab's
   torch or numpy: `requirements.txt` pins `numpy<2` and `torch<2.9` for Intel Macs, which would
   break Colab's preinstalled packages.
3. Copy the two parquet files from Drive into `data/processed/`.
4. Run about 20 steps to check the loss stays finite and measure speed, then start the run.
5. Choose the run with one variable (run name and options).

**7. v1 full run.** After epoch 1, micro AP on tune should be well above the 0.024 base rate. If it
isn't, stop and investigate before using more GPU time.

**8. #19 PR.** Report v1's validation results (micro/macro AP, F1 at 0.5, per-category table) and
point #21 and #22 to `results/modernbert_v1/`. Mention:
- the three-way split (the issue only mentions train and validation);
- that the tune set is used for checkpoint selection and later for #21's cutoffs, so tune scores
  will run slightly high. Validation is untouched.

### Part 2: issue #20 (v2–v4)

**9. Runs.** Run v2 and v3 on Colab with the same notebook, plus v4 if time allows.

**10. Compare against v1.** Pick the best setup by tune macro AP, and also report validation for
every run:

| Run | tune macro AP | val micro AP | val macro AP | val macro AP (categories with positives) | val micro F1 @0.5 | val macro F1 @0.5 |
|---|---|---|---|---|---|---|
| v1 plain | | | | | | |
| v2 bias init | | | | | | |
| v3 pos_weight √ | | | | | | |
| v4 oversample (stretch) | | | | | | |

Also list the categories that changed most between v1 and the best run, especially those with
fewer than 50 positives.

**11. #20 PR.** The comparison table, which setup is recommended for #21/#22, and the saved
results for every run.

## Order for this session

1. Steps 1–4, then the step 5 smoke tests, all locally.
2. Copy the parquet files to Drive and set up the Colab notebook (step 6).
3. Start the v1 run (step 7). While it trains, write the #19 PR text.
4. Run v2 and v3 back-to-back (step 9), then v4 if time allows.
5. Fill in the #20 comparison (steps 10–11).

If Colab isn't available, run v1 locally overnight (roughly 6–7 hours) and run #20's options on
Colab later.

## Risks

- **Rare categories:** at a 0.5 cutoff, v1 will likely get F1 = 0 for rare categories because
  their probabilities never reach 0.5. That's expected; AP shows whether they're still being learned.
- **pos_weight too strong:** if v3 floods the rare categories with "yes," its F1 at 0.5 drops even
  when AP improves. That's why AP is the main metric. If needed, cap the weights lower.
- **fp16 on T4:** no confirmed ModernBERT problem was found. The 20-step check catches a NaN loss
  early; if one appears, fall back to fp32.
- **Out of GPU memory:** drop to a batch size of 4 with gradient accumulation of 4.
- **Colab limits:** the free tier can disconnect or run out of GPU time. Checkpoints on Drive let a
  run resume, and v4 is the first thing to drop.
- **Training time:** unknown on Colab until the step 6 speed check.

## Files

| New | Changed |
|---|---|
| `notebooks/splits.py`, `data/splits/*.txt` | `notebooks/modernbert_tokenizer_pipeline.py` (`padding=False`) |
| `notebooks/eval_metrics.py` | `.gitignore` (local smoke-run folder) |
| `notebooks/modernbert_finetune.py` | |
| `notebooks/modernbert_finetune_colab.ipynb` | |
| `results/modernbert_v1/` … `results/modernbert_v4_oversample/` | |
