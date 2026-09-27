"""
Evidence-preservation audit for the CUAD chunking pipeline.

The chunking config below is already settled (see alec-eda-01.ipynb for how we
picked the chunker, and chunk_size_sweep.py / check_768_config.py for the
window-size sweep that took us from 512 to 768 tokens). At 768 tokens we get
0 OVERSIZED evidence spans (down from 6 at 512), preservation climbs to 99.7%
(from 99.1%), and we end up with 38% fewer chunks (8,031 vs 13,075) than the
512/128 setup. So this script isn't re-litigating chunker choice — it just
checks how well that fixed config preserves labeled evidence spans across all
408 CUAD training contracts, broken down by category.

Run it from notebooks/ (same convention as the rest of the repo, since we
import cuad_cleaning.py as a sibling module rather than a package):

    cd notebooks
    python evidence_preservation_audit.py
"""

import json
import random
from pathlib import Path

import pandas as pd
import tiktoken
from chunking_evaluation.chunking import RecursiveTokenChunker

from cuad_cleaning import clean_text_with_map, build_reverse_map, relocate_answer

# --- config: this is the settled chunking choice, all in one place ---
WINDOW_SIZE = 768
OVERLAP = 192
ENCODING_NAME = "cl100k_base"
CANDIDATE_WINDOW_SIZES_FOR_OVERSIZE_CHECK = (512, 768, 1024)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "cuad" / "train_separate_questions.json"
OUTPUT_CSV = Path(__file__).resolve().parent / "evidence_preservation_by_category.csv"

N_SANITY_CHECK_SAMPLES = 10
RANDOM_SEED = 0


# --- loading + cleaning the CUAD data (same shape as alec-eda-01.ipynb / main_pipeline.ipynb) ---
def load_cuad_df(path) -> pd.DataFrame:
    with open(path) as f:
        raw = json.load(f)
    df = pd.json_normalize(raw["data"], record_path=["paragraphs"], meta=["title"])
    df = df.reset_index(names="doc_id")
    df = df.explode("qas", ignore_index=True)
    df = pd.concat([df.drop(columns="qas"), pd.json_normalize(df["qas"])], axis=1)
    df["category"] = df["question"].str.extract(r'related to "([^"]+)"')
    df = df.explode("answers", ignore_index=True)
    df = pd.concat(
        [df.drop(columns="answers"), pd.json_normalize(df["answers"]).rename(columns={"text": "answer_text"})],
        axis=1,
    )

    # "id" is per-QA in the raw CUAD json, not per-answer, so a QA with multiple
    # answers explodes into several rows sharing one id (happens on CUADv1.json
    # and test.json, though not here on train_separate_questions.json). Only
    # touch the ids that actually collide so the rest stay untouched.
    dupe_mask = df["id"].duplicated(keep=False)
    if dupe_mask.any():
        suffix = df.groupby("id").cumcount().astype(str)
        df.loc[dupe_mask, "id"] = df.loc[dupe_mask, "id"] + "__ans" + suffix[dupe_mask]
    return df


def build_clean_df(path) -> pd.DataFrame:
    df = load_cuad_df(path)

    unique_title_context_pairs = len(df[["title", "context"]].drop_duplicates())
    unique_titles = df["title"].nunique()
    if unique_title_context_pairs != unique_titles:
        raise ValueError("found a title with more than one different context")

    title_to_clean_text = {}
    title_to_reverse_map = {}
    for title, context in df.drop_duplicates("title")[["title", "context"]].values:
        cleaned, clean_to_raw = clean_text_with_map(context)
        title_to_clean_text[title] = cleaned
        title_to_reverse_map[title] = build_reverse_map(len(context), clean_to_raw, len(cleaned))
    df["text"] = df["title"].map(title_to_clean_text)

    annotations = df.apply(relocate_answer, axis=1, title_to_reverse_map=title_to_reverse_map)
    df = pd.concat([df, annotations], axis=1)

    clean_df = df.rename(columns={"title": "contract_id"})[
        ["contract_id", "id", "text", "category", "is_impossible", "annotation_text", "annotation_start"]
    ].reset_index(drop=True)
    assert clean_df["id"].is_unique, "clean_df ids must be unique — sanity_check_offsets() indexes by id"
    return clean_df


# --- chunking + recovering each chunk's offset back in the source text ---
def chunk_text_with_offsets(chunker, text: str):
    """Figure out each chunk's (start, end) position in `text`.

    We search the raw text directly rather than a whitespace-normalized copy.
    RecursiveTokenChunker / FixedTokenChunker never rewrite content, so a
    chunk is always an exact substring of `text` — every position str.find()
    hands back is correct by construction. A normalized-text approach has to
    collapse whitespace runs to build its position map, and mapping a match
    back through that collapse can silently miscount. That's not just
    theoretical: an earlier normalization-based version of this function
    recovered a chunk one character too long on a contract with a long,
    periodic redacted section (the same ~40-char line repeated hundreds of
    times, with a stray double-newline landing mid-chunk) — the collapsed-run
    bookkeeping got it wrong.

    `search_from` just keeps a chunk from being found before the previous one
    started, which helps when the exact same text repeats verbatim (that same
    redacted section, or boilerplate like a page footer). It won't always pick
    the "intended" occurrence in pathological cases, but whichever occurrence
    it lands on, the recovered (start, end) always reproduces the chunk's text
    exactly — there's no ambiguity about correctness, only about position.
    If the bounded search comes up empty, we report that honestly as a
    failure ((-1, -1)) instead of guessing. Failed chunks get filtered out of
    evidence matching downstream (the `chunk_start >= 0` checks) and counted
    in the offset-recovery-failure warning main() prints.
    """
    chunks = chunker.split_text(text)
    offsets = []
    search_from = 0
    for chunk in chunks:
        if not chunk.strip():
            offsets.append((-1, -1))
            continue

        start = text.find(chunk, search_from)
        if start == -1:
            offsets.append((-1, -1))
            continue

        end = start + len(chunk)
        offsets.append((start, end))
        search_from = start
    return list(zip(chunks, offsets))


def build_chunks_df(chunker, clean_df: pd.DataFrame) -> pd.DataFrame:
    contracts = clean_df.drop_duplicates("contract_id")[["contract_id", "text"]]
    rows = []
    for contract_id, text in contracts.itertuples(index=False):
        for i, (chunk_text, (start, end)) in enumerate(chunk_text_with_offsets(chunker, text)):
            rows.append({
                "contract_id": contract_id,
                "chunk_id": f"{contract_id}__chunk{i}",
                "chunk_index": i,
                "chunk_text": chunk_text,
                "chunk_start": start,
                "chunk_end": end,
            })
    return pd.DataFrame(rows)


# --- classifying each evidence span as preserved, split, oversized, or uncovered ---
def classify_spans(clean_df: pd.DataFrame, chunks_df: pd.DataFrame, token_len_fn, window_size: int) -> pd.DataFrame:
    evidenced = clean_df[~clean_df["is_impossible"]].copy()
    evidenced["annotation_end"] = evidenced["annotation_start"] + evidenced["annotation_text"].str.len()
    evidenced["span_tokens"] = evidenced["annotation_text"].apply(token_len_fn)

    results = []
    for contract_id, group in evidenced.groupby("contract_id"):
        contract_chunks = chunks_df[(chunks_df["contract_id"] == contract_id) & (chunks_df["chunk_start"] >= 0)]
        for _, ev in group.iterrows():
            if ev["span_tokens"] > window_size:
                # Too big for any chunk of this size to hold, full stop —
                # no need to even look at where the boundaries fall.
                bucket = "OVERSIZED"
            else:
                overlapping = contract_chunks[
                    (ev["annotation_start"] < contract_chunks["chunk_end"])
                    & (ev["annotation_end"] > contract_chunks["chunk_start"])
                ]
                if len(overlapping) == 0:
                    # Different from SPLIT: nothing even touches this span's
                    # range, which points to a real coverage gap upstream
                    # (e.g. every chunk over that range failed offset
                    # recovery and got filtered out), not just a boundary cut.
                    bucket = "NO_COVERAGE"
                else:
                    fully_contained = (
                        (overlapping["chunk_start"] <= ev["annotation_start"])
                        & (overlapping["chunk_end"] >= ev["annotation_end"])
                    ).any()
                    bucket = "PRESERVED" if fully_contained else "SPLIT"
            results.append({
                "id": ev["id"],
                "contract_id": contract_id,
                "category": ev["category"],
                "annotation_start": ev["annotation_start"],
                "annotation_end": ev["annotation_end"],
                "span_tokens": ev["span_tokens"],
                "bucket": bucket,
            })
    return pd.DataFrame(results)


# --- sanity checks: make sure the offset math actually holds before we trust the report ---
def sanity_check_offsets(clean_df, chunks_df, spans_df, n=N_SANITY_CHECK_SAMPLES, seed=RANDOM_SEED):
    rng = random.Random(seed)
    preserved = spans_df[spans_df["bucket"] == "PRESERVED"]
    sample_ids = rng.sample(list(preserved["id"]), min(n, len(preserved)))
    evidenced = clean_df.set_index("id")

    # Sampling only PRESERVED spans catches a bug that falsely marks something
    # PRESERVED, but it'd miss a bug that wrongly demotes a genuinely-preserved
    # span to SPLIT (say, an off-by-one that shrinks a chunk's end by one
    # char) — that span would just quietly drop out of the sampled pool.
    #
    # Also, a bug inside chunk_text_with_offsets() would corrupt chunk_start/
    # chunk_end and chunk_text in lockstep, so comparing evidence spans
    # against chunks_df can't catch that either — it already carries the same
    # bug. Only comparing chunks_df back against the actual source text can.
    # So we do that first, exhaustively, since it's cheap (just slicing) and
    # it's the only independent ground truth we have.
    print("\n--- checking every chunk's recorded offsets reproduce its own chunk_text ---")
    contract_text_by_id = clean_df.drop_duplicates("contract_id").set_index("contract_id")["text"]
    valid_chunks = chunks_df[chunks_df["chunk_start"] >= 0]
    bad_chunks = [
        row.chunk_id for row in valid_chunks.itertuples(index=False)
        if contract_text_by_id[row.contract_id][row.chunk_start:row.chunk_end] != row.chunk_text
    ]
    print(f"checked {len(valid_chunks)} chunks, {len(bad_chunks)} have offsets that don't reproduce chunk_text")
    if bad_chunks:
        raise RuntimeError(
            f"Offset arithmetic sanity check FAILED — {len(bad_chunks)} chunks' recorded offsets don't "
            f"reproduce their own chunk_text: {bad_chunks[:5]}"
        )

    # A second, cheaper check on top of that: this one catches a bug in the
    # overlap/containment logic in classify_spans() itself, rather than a bug
    # in the underlying chunk offsets we just verified above.
    print("\n--- checking no SPLIT span is actually fully contained by some chunk (would mean a wrong demotion) ---")
    wrongly_split = []
    for eid in spans_df.loc[spans_df["bucket"] == "SPLIT", "id"]:
        row = evidenced.loc[eid]
        contract_id = row["contract_id"]
        ann_start = int(row["annotation_start"])
        ann_end = ann_start + len(row["annotation_text"])
        contract_chunks = chunks_df[(chunks_df["contract_id"] == contract_id) & (chunks_df["chunk_start"] >= 0)]
        truly_contained = (
            (contract_chunks["chunk_start"] <= ann_start) & (contract_chunks["chunk_end"] >= ann_end)
        ).any()
        if truly_contained:
            wrongly_split.append(eid)
    print(f"checked {int((spans_df['bucket'] == 'SPLIT').sum())} SPLIT spans, {len(wrongly_split)} wrongly demoted")
    if wrongly_split:
        raise RuntimeError(
            f"Offset arithmetic sanity check FAILED — {len(wrongly_split)} spans are labeled SPLIT but are "
            f"actually fully contained by a chunk (wrongly demoted from PRESERVED): {wrongly_split[:5]}"
        )

    print("\n--- offset arithmetic sanity check (re-extracting spans from computed chunk positions) ---")
    n_pass = 0
    for eid in sample_ids:
        row = evidenced.loc[eid]
        contract_id = row["contract_id"]
        contract_text = row["text"]
        ann_start = int(row["annotation_start"])
        ann_end = ann_start + len(row["annotation_text"])

        # check 1: the annotation offsets line up against the contract's own text
        matches_contract_text = contract_text[ann_start:ann_end] == row["annotation_text"]

        # check 2: translate those same offsets into chunk-local coordinates
        # and see if the chunk's own text reproduces the span — this is the
        # part that actually exercises the offset-recovery arithmetic.
        contract_chunks = chunks_df[(chunks_df["contract_id"] == contract_id) & (chunks_df["chunk_start"] >= 0)]
        containing = contract_chunks[
            (contract_chunks["chunk_start"] <= ann_start) & (contract_chunks["chunk_end"] >= ann_end)
        ]
        matches_chunk_text = False
        if len(containing) > 0:
            chunk_text = containing.iloc[0]["chunk_text"]
            chunk_start = containing.iloc[0]["chunk_start"]
            extracted = chunk_text[ann_start - chunk_start: ann_end - chunk_start]
            matches_chunk_text = extracted == row["annotation_text"]

        ok = matches_contract_text and matches_chunk_text
        n_pass += int(ok)
        print(f"  {eid[:70]:70s} contract_text_match={matches_contract_text} chunk_text_match={matches_chunk_text}")

    print(f"sanity check: {n_pass}/{len(sample_ids)} passed")
    if n_pass != len(sample_ids):
        raise RuntimeError("Offset arithmetic sanity check FAILED — do not trust the report below.")


def main():
    encoding = tiktoken.get_encoding(ENCODING_NAME)

    def token_len(text: str) -> int:
        return len(encoding.encode(text))

    print(f"loading and cleaning CUAD data from {DATA_PATH} ...")
    clean_df = build_clean_df(DATA_PATH)
    print(f"{clean_df['contract_id'].nunique()} contracts, {len(clean_df)} category rows")

    chunker = RecursiveTokenChunker(
        chunk_size=WINDOW_SIZE,
        chunk_overlap=OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=token_len,
    )
    print(f"chunking with RecursiveTokenChunker(chunk_size={WINDOW_SIZE}, "
          f"chunk_overlap={OVERLAP}, encoding={ENCODING_NAME}) ...")
    chunks_df = build_chunks_df(chunker, clean_df)
    n_offset_fail = int((chunks_df["chunk_start"] == -1).sum())
    print(f"{len(chunks_df)} chunks across {chunks_df['contract_id'].nunique()} contracts "
          f"({n_offset_fail} chunk offset-recovery failures)")
    if n_offset_fail:
        print(f"WARNING: {n_offset_fail} chunks could not be located in their contract's text "
              f"and were excluded from evidence matching below.")

    spans_df = classify_spans(clean_df, chunks_df, token_len, WINDOW_SIZE)
    n_no_coverage = int((spans_df["bucket"] == "NO_COVERAGE").sum())
    if n_no_coverage:
        print(f"WARNING: {n_no_coverage} evidence spans had zero overlapping chunks (NO_COVERAGE) — "
              f"this is a coverage gap, not a normal SPLIT, and likely traces back to offset-recovery "
              f"failures upstream.")

    sanity_check_offsets(clean_df, chunks_df, spans_df)

    # --- overall bucket counts ---
    total = len(spans_df)
    counts = spans_df["bucket"].value_counts().reindex(["PRESERVED", "SPLIT", "OVERSIZED", "NO_COVERAGE"], fill_value=0)
    print("\n=== Overall evidence-preservation audit ===")
    print(f"total evidence spans: {total}")
    for bucket, n in counts.items():
        print(f"  {bucket:10s}: {n:5d} ({n / total * 100:5.1f}%)")

    # --- per-category breakdown ---
    cat_summary = (
        spans_df.groupby("category")["bucket"]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=["PRESERVED", "SPLIT", "OVERSIZED", "NO_COVERAGE"], fill_value=0)
    )
    cat_summary["n_positive"] = cat_summary.sum(axis=1)
    cat_summary["preservation_rate_pct"] = (cat_summary["PRESERVED"] / cat_summary["n_positive"] * 100).round(1)
    cat_summary = cat_summary.sort_values("preservation_rate_pct", ascending=True)
    cat_summary = cat_summary[["n_positive", "PRESERVED", "SPLIT", "OVERSIZED", "NO_COVERAGE", "preservation_rate_pct"]]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cat_summary.to_csv(OUTPUT_CSV)
    print(f"\nper-category breakdown (sorted by preservation rate, ascending) saved to {OUTPUT_CSV}")
    print(cat_summary.to_string())

    # --- token-length distribution of evidence spans ---
    span_tokens = spans_df["span_tokens"]
    percentiles = span_tokens.quantile([0.5, 0.9, 0.95, 0.99])
    print("\n=== Evidence span token-length distribution ===")
    print(f"median={percentiles[0.5]:.0f}  p90={percentiles[0.9]:.0f}  "
          f"p95={percentiles[0.95]:.0f}  p99={percentiles[0.99]:.0f}  max={span_tokens.max():.0f}")

    print("\noversized fraction at candidate window sizes:")
    oversize_fracs = {}
    for w in CANDIDATE_WINDOW_SIZES_FOR_OVERSIZE_CHECK:
        n_over = int((span_tokens > w).sum())
        frac = n_over / total * 100
        oversize_fracs[w] = frac
        print(f"  window={w:4d}: {n_over:4d} / {total} spans OVERSIZED ({frac:.2f}%)")

    # --- written summary, generated from the numbers computed above ---
    worst = cat_summary.head(5)
    n_split = int(counts["SPLIT"])
    n_oversized = int(counts["OVERSIZED"])
    if n_split > n_oversized:
        dominant_failure = f"SPLIT ({n_split} vs {n_oversized} OVERSIZED)"
    elif n_oversized > n_split:
        dominant_failure = f"OVERSIZED ({n_oversized} vs {n_split} SPLIT)"
    else:
        dominant_failure = f"SPLIT and OVERSIZED equally ({n_split} each)"
    justify_larger_window = percentiles[0.95] > WINDOW_SIZE or percentiles[0.99] > WINDOW_SIZE

    print("\n=== Summary ===")
    print(f"At window={WINDOW_SIZE}/overlap={OVERLAP}, {int(counts['PRESERVED'])}/{total} "
          f"({counts['PRESERVED'] / total * 100:.1f}%) of evidence spans are fully preserved in one chunk.")
    print(f"Failures are dominated by {dominant_failure}.")
    print("Worst-affected categories (lowest preservation rate, with support):")
    for cat, row in worst.iterrows():
        print(f"  {cat:40s} preservation={row['preservation_rate_pct']:5.1f}%  n={int(row['n_positive']):4d}  "
              f"(SPLIT={int(row['SPLIT'])}, OVERSIZED={int(row['OVERSIZED'])})")
    if justify_larger_window:
        print(f"p95/p99 span length ({percentiles[0.95]:.0f}/{percentiles[0.99]:.0f} tokens) EXCEEDS the "
              f"{WINDOW_SIZE}-token window, so a larger window (768 or 1024) would reduce OVERSIZED failures in "
              f"the tail: {oversize_fracs[768]:.2f}% of spans remain OVERSIZED at 768 tokens, "
              f"{oversize_fracs[1024]:.2f}% at 1024.")
    else:
        print(f"p95/p99 span length ({percentiles[0.95]:.0f}/{percentiles[0.99]:.0f} tokens) stays within the "
              f"{WINDOW_SIZE}-token window, so window size is not the bottleneck for the bulk of spans — "
              f"remaining SPLIT failures are more likely fixable by tuning overlap or boundary placement "
              f"than by moving to a larger window.")


if __name__ == "__main__":
    main()