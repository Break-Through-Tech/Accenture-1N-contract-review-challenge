# ModernBERT results (issues #19 and #20)

Five fine-tuning runs of `answerdotai/ModernBERT-base` on the CUAD chunks, scored on the 66 tune
and 82 validation contracts. **Recommended model for #21 and #22: `modernbert_v2_biasinit`.**

How the runs were set up, and why: [docs/issue-19-modernbert-finetune-plan.md](../docs/issue-19-modernbert-finetune-plan.md).

## The runs

All runs use the same data, split, code version (commit `cdf8f70`), GPU (A100, bf16) and settings
(learning rate 5e-5, effective batch 16, seed 0), and differ only in the option shown. Each took
4–8 minutes to train.

| Folder | Issue | What's different | Option |
|---|---|---|---|
| `modernbert_v1` | #19 | Nothing: plain BCE loss, 3 epochs | none |
| `modernbert_v2_biasinit` | #20 | Each category's output starts at its real positive rate instead of 50% | `--bias-init` |
| `modernbert_v3_posweight` | #20 | Class weights in the loss: √(negatives ÷ positives), 4.0–20.4 per category | `--pos-weight sqrt` |
| `modernbert_v4_oversample` | #20 | Training chunks containing any of the 10 categories with < 50 training positives appear 3×  (+494 rows) | `--oversample-rare 50` |
| `modernbert_v1_6ep` | control | v1 trained for 6 epochs instead of 3, to separate "better setup" from "more training" | `--epochs 6` |

## Results (82 validation contracts, 1,841 chunks, 1,691 positive pairs)

| Run | tune macro AP* | val micro AP | val macro AP* | val precision | val recall | val micro F1 @0.5 | val macro F1 @0.5 | Categories with F1 = 0 |
|---|---|---|---|---|---|---|---|---|
| v1 plain | 0.472 | 0.694 | 0.478 | 0.809 | 0.521 | 0.634 | 0.359 | 12 |
| **v2 bias init** | **0.569** | **0.761** | **0.584** | 0.826 | 0.608 | **0.700** | **0.494** | 5 |
| v3 class weights | 0.525 | 0.705 | 0.547 | 0.582 | **0.701** | 0.636 | **0.494** | **3** |
| v4 oversample rare | 0.513 | 0.715 | 0.540 | 0.806 | 0.556 | 0.658 | 0.425 | 7 |
| v1, 6 epochs | 0.557 | 0.736 | 0.540 | **0.842** | 0.584 | 0.690 | 0.455 | 8 |

\* AP averaged over the 40 categories with at least one validation positive (41 on tune).
`Unlimited/All-You-Can-Eat-License` has none in validation. Precision, recall and F1 use a 0.5
cutoff and are micro-averaged unless marked macro; macro F1 is over all 41 categories, matching how
earlier baseline numbers were computed. **AP is the main metric** because it doesn't depend on a
cutoff, and the runs put their probabilities on different scales (class weights push them up).

**How sure we are.** I resampled the 82 validation contracts with replacement 1,000 times and
recomputed each difference:

| Comparison | micro AP difference (95% interval) | macro AP* difference (95% interval) |
|---|---|---|
| v2 − v1 | +0.067 (+0.054 to +0.080) | +0.110 (+0.084 to +0.139) |
| v3 − v1 | +0.011 (−0.000 to +0.023) | +0.065 (+0.042 to +0.088) |
| v4 − v1 | +0.021 (+0.011 to +0.031) | +0.058 (+0.034 to +0.078) |
| v1 6 epochs − v1 | +0.042 (+0.028 to +0.055) | +0.062 (+0.040 to +0.084) |
| v2 − v1 6 epochs | +0.025 (+0.012 to +0.039) | +0.048 (+0.025 to +0.073) |
| v2 − v3 | +0.056 (+0.043 to +0.069) | +0.045 (+0.014 to +0.075) |

These intervals cover which contracts happen to be in validation, not run-to-run training
randomness: each setup was trained once, with one seed. The v2 vs v1 gap is large enough to trust.
The smaller v2 vs v3 gap is likely real, but a second seed would confirm it.

## Findings

1. **v2 (bias init) is the best setup** on every main score, chosen on the tune set and confirmed
   on validation. It also beats plain v1 trained for twice as long, so the gain isn't just faster
   training. Without it, the model spends early training learning that "no" is the usual answer;
   starting at the real rates skips that.
2. **All three #20 options beat plain v1** on macro AP (+0.06 to +0.11), mostly by lifting the
   rarer categories.
3. **Class weights (v3) trade precision for recall:** recall 0.70 vs 0.52, precision 0.58 vs 0.81,
   so F1 at 0.5 barely moves. It has the fewest categories at F1 = 0 (3) and the best average AP on
   the rarest categories (0.36 vs v2's 0.33), but it's weaker on everything else.
4. **Plain training levels off around epoch 5:** the 6-epoch run peaked at epoch 5 (tune macro AP
   0.557) and dipped slightly at epoch 6. v2 was still improving at epoch 3 (0.542 → 0.569).
5. **Rare categories improved but aren't solved.** For the 9 categories with fewer than 50 training
   positives (and some validation positives), average AP went from 0.17 (v1) to 0.33 (v2). With v2,
   these still score F1 = 0 at a 0.5 cutoff:
   - Price Restrictions (13 training positives; near zero in every run)
   - Non-Disparagement
   - Source Code Escrow (only 2 validation positives, so very noisy)
   - Most Favored Nation
   - Competitive Restriction Exception

   Per-category cutoffs (#21) are the natural next lever.
6. **Common categories are strong already:** v2 validation AP is 0.96 for Governing Law and Insurance,
   0.95 for Expiration Date, and 0.92 for Parties and Audit Rights.

## Files in each run folder

| File | What it is |
|---|---|
| `val_probs.parquet`, `tune_probs.parquet` | One row per chunk: `chunk_id`, `contract_id`, then one probability column per category (41). These are the model's outputs; nothing needs re-running. |
| `metrics.json` | Overall scores for tune and validation (`micro_ap`, `macro_ap_all`, `macro_ap_present`, precision, recall and F1 at 0.5, counts), plus the best checkpoint. |
| `per_category.csv` | Per category and split: number of positives, AP, precision, recall, F1, number predicted "yes". |
| `run_config.json` | Options, settings, GPU, git commit, package versions, per-category class weights or starting biases, and the full training log (loss per 20 steps, tune scores per epoch). |

To score a run's probabilities (or any model's) the same way, from `notebooks/` after running
`main_pipeline.ipynb`:

```python
import pandas as pd
from eval_metrics import multilabel_metrics

probs = pd.read_parquet("../results/modernbert_v2_biasinit/val_probs.parquet")
labels = pd.read_parquet("../data/processed/chunk_labels_df.parquet").pivot_table(
    index="chunk_id", columns="category", values="label", aggfunc="max", fill_value=0)
cats = labels.columns.tolist()
summary, per_category = multilabel_metrics(labels.loc[probs["chunk_id"], cats].to_numpy(), probs[cats].to_numpy(), cats)
```

## Notes for #21 and #22

- **#21 (cutoffs):** pick cutoffs on `tune_probs.parquet` and report them on `val_probs.parquet`,
  never the other way round. The tune set already chose each run's best epoch, so scores on tune
  run slightly high. Validation is untouched.
- **#22 (baseline comparison):** score the TF-IDF baseline with `multilabel_metrics` on the same
  82 validation contracts (`data/splits/val_contracts.txt`), so both models use identical metrics.
  Compare AP first. F1 at 0.5 is unfair to the baseline, which trains on balanced samples and so
  puts its probabilities on a different scale.
- **Model weights** (~600 MB per run) aren't in the repo. They're in Albert's Google Drive under
  `cuad_finetune/runs/<run>/model/`; ask Albert for access if a step needs the model itself rather
  than its saved predictions.
- **To reproduce a run:** open `notebooks/modernbert_finetune_colab.ipynb` in Colab and use the
  run's option from the table above.
