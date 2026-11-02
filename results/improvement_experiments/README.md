# Targeted improvement experiments (issue #24, milestone 8A)

Three experiments, each changing one thing relative to a parent model, run in
[notebooks/improvement_experiments.ipynb](../../notebooks/improvement_experiments.ipynb). They build on
[error_analysis.ipynb](../../notebooks/error_analysis.ipynb) (PR #29). All choices are made on the 66 tune contracts,
the 82 validation contracts are only used to report, `test.json` is never touched, and all 41 categories are predicted.

**Current best model (unchanged):** `ensemble_top2`, the mean probability of `modernbert_v2_biasinit` and
`modernbert_v1_6ep`, with per-category best-F1 cutoffs. Validation macro F1 0.559, micro F1 0.664, macro AP 0.593.

## Results

| ID | Change | Parent | tune macro AP | val macro AP | val micro F1 | val macro F1 | Verdict |
|---|---|---|---|---|---|---|---|
| E0 | Current best (`ensemble_top2`, best-F1 cutoffs) | | 0.590 | 0.593 | 0.664 | 0.559 | Reference |
| E1 | Threshold floor: fewer than N tune positives keep 0.5 (N by 5-fold CV on tune) | E0 | 0.590 | 0.593 | 0.664 | 0.559 | Didn't help (CV picks no floor) |
| v2 | `modernbert_v2_biasinit` (lr 5e-5, bias init) | | 0.569 | 0.584 | 0.640 | 0.536 | Parent of E2/E3 |
| E2 | Learning rate 3e-5 | v2 | 0.508 | 0.528 | 0.608 | 0.492 | Hurt |
| E3 | Class weights `pos_weight=sqrt` | v2 | 0.570 | 0.587 | 0.648 | 0.537 | Neutral overall, better on rare categories |
| Follow-up | Ensemble of E3 + v2 | E0 | 0.588 | 0.606 | 0.644 | 0.543 | Not selected (tune macro AP below E0) |

F1 columns use per-category best-F1 cutoffs picked on tune. Macro averages are over categories with positives.

**Validation AP differences vs. v2** (1,000 resamples of the validation contracts, 95% interval):

| Comparison | micro AP | macro AP |
|---|---|---|
| E2 minus v2 | -0.036 (-0.044 to -0.026) | -0.056 (-0.081 to -0.037) |
| E3 minus v2 | -0.017 (-0.026 to -0.009) | +0.002 (-0.023 to +0.029) |

## Findings

- **E1 (threshold floor) didn't help.** Picking N by cross-validated tune macro F1 gives N = 1, the existing rule.
  A floor of N = 3 is noted as a precision option, seen after validation and not selected: it gives
  Affiliate License-Licensor and Most Favored Nation (2 tune positives each) the 0.5 cutoff instead of 0.01 and 0.02.
  Validation false positives drop from 684 to 506 and micro F1 rises from 0.664 to 0.696 at flat macro F1, but those
  two categories lose most of their recall.
- **E2 (lower learning rate) hurt,** with high confidence. With 3 epochs the model learns too slowly; v2 was already
  still improving at epoch 3.
- **E3 (class weights on top of bias init) is the useful result.** Overall AP ties v2, but mean AP on the 9 rare
  categories (fewer than 50 training positives) rises from 0.33 to 0.37, and at the plain 0.5 cutoff it has the best
  single-run macro F1 (0.545 vs 0.506) and recall (0.729). The cost is a small drop in micro AP on common categories.
- **The best model stays `ensemble_top2`.** E3 + v2 scored 0.588 tune macro AP vs 0.590 for the current ensemble.

## Note on clause grouping

Arielle's unmerged `clause-grouping` branch maps the 41 categories into 12 groups and drops Document Name and Parties.
These experiments stay on 41. For reference: of the current best model's 684 validation false positives, 261 are
confusions inside one group (63 from Affiliate License-Licensor on licensing chunks) and 24 are Document Name or
Parties predictions. About 42% of false alarms would no longer count under grouped labels. That's an upper bound,
since grouping also changes which chunks are positive.

## Caveats

- Each setup was trained once, with seed 0. The intervals cover which contracts are in validation, not training
  randomness.
- E2 and E3 ran on the same hardware and settings as v2 (A100, bf16, torch 2.11.0, same training script) but with
  transformers 5.21.0 instead of 5.20.1. No control rerun of v2 was done.
- The runs were named `..._baisinit_...` in Colab. Folders and `run_name` fields were corrected to `biasinit`; the
  Google Drive paths in `run_config.json` and `metrics.json` keep the original spelling because that's where the
  models are stored.

## Next experiments

- (GPU) v2 + class weights trained for 5 or 6 epochs.
- (No GPU) add E3 to the ensemble with its own cutoffs, or use E3's probabilities for rare categories only.

## Files

| File | What it is |
|---|---|
| `experiment_log.csv` | One row per experiment: config, parent, tune and validation metrics |
| `e1_floor_sweep.csv` | Every floor N tried: cross-validated tune scores, report-only validation scores |
| `e1_selected_thresholds.csv` | Per-category cutoffs for E0, E1 and the noted N = 3 option |
| `e1_floored_categories_n3.csv` | What N = 3 changes for the two floored categories |
| `e2_e3_vs_v2.csv` | v2, E2 and E3 under 0.5 and best-F1 cutoffs |
| `e2_e3_bootstrap.csv` | AP differences vs. v2 with 95% intervals |
| `grouping_fp_breakdown.csv` | Current best model's validation false positives, by relation to the 12-group mapping |

The E2 and E3 runs themselves are in `results/modernbert_v5_biasinit_lr3e-5/` and
`results/modernbert_v6_biasinit_posweight/`, in the same format as the other runs (see `results/README.md`).
