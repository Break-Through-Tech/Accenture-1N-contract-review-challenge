"""
One metrics function for every chunk-level, 41-category model (ModernBERT runs, and the TF-IDF
baseline in #22), so all of them get scored exactly the same way.

Two things it does on purpose:

- Average precision (AP) is the headline number. It scores how well each category's scores rank
  positive chunks above negative ones, without picking a cutoff. Models trained differently
  (balanced subsampling, class weighting, plain BCE) put their probabilities on different scales,
  so F1 at one fixed cutoff partly measures that scale instead of real skill. F1 at the cutoff is
  still reported, since it's what a reviewer would actually see.
- Macro averages come two ways. "all" averages over every category, counting a category with no
  positives in the eval set as 0 -- that's what sklearn does silently, and it's how the existing
  baseline numbers were computed, so it stays comparable. "present" averages only over categories
  that have at least one positive in the eval set, which is the fairer number (e.g.
  Unlimited/All-You-Can-Eat-License has 0 positives in validation, which caps the "all" macro
  at 40/41).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


def multilabel_metrics(y_true, scores, categories, threshold=0.5):
    """Score one (n_chunks x n_categories) set of predictions.

    y_true: 0/1 labels. scores: probabilities (or any scores where higher = more likely yes).
    threshold: a single cutoff, or one per category (array of len(categories)).

    Returns (summary, per_category): summary is a flat dict of floats/ints (easy to log or save as
    json), per_category is one row per category, in the order of `categories`."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    if y_true.shape != scores.shape or y_true.shape[1] != len(categories):
        raise ValueError(f"shape mismatch: y_true {y_true.shape}, scores {scores.shape}, "
                         f"{len(categories)} categories")
    if np.isnan(scores).any():
        raise ValueError("scores contain NaN -- the model output is broken, not just bad")

    y_pred = (scores >= np.asarray(threshold, dtype=float)).astype(int)
    n_pos = y_true.sum(axis=0)
    present = n_pos > 0

    # per-category AP; categories with no positives get 0 (sklearn's convention) and are left out
    # of the "present" average below
    ap = np.zeros(len(categories))
    for i in np.flatnonzero(present):
        ap[i] = average_precision_score(y_true[:, i], scores[:, i])

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )

    summary = {
        "micro_ap": float(average_precision_score(y_true.ravel(), scores.ravel())),
        "macro_ap_all": float(ap.mean()),
        "macro_ap_present": float(ap[present].mean()),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "macro_f1_all": float(f1.mean()),
        "macro_f1_present": float(f1[present].mean()),
        "n_chunks": int(y_true.shape[0]),
        "n_positive_pairs": int(y_true.sum()),
        "base_rate": float(y_true.mean()),
        "n_categories_present": int(present.sum()),
    }
    per_category = pd.DataFrame({
        "category": list(categories),
        "n_positive": n_pos,
        "ap": ap,
        "precision": p,
        "recall": r,
        "f1": f1,
        "n_predicted_yes": y_pred.sum(axis=0),
    })
    return summary, per_category
