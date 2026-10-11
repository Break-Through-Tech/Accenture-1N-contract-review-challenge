"""
How big are our current chunks once ModernBERT's own tokenizer sees them?

The current chunker sizes chunks with tiktoken's cl100k_base (768 tokens, 192
overlap). The transformer we're planning to fine-tune (ModernBERT-base) uses a
different tokenizer, so a "768-token" chunk could come out longer on its side
and get truncated at training time. If it does, a chunk can be labeled positive
for a category even though the evidence sat in the part the model never sees.

This script measures that gap on the current chunks, so we know how far off the
cl100k sizing is before switching chunking over to the ModernBERT tokenizer.
The old chunking config is pinned below on purpose (not imported from
chunking_util.py), so this stays a record of the "before" setup even after
chunking_util.py moves to the new config.

Run it from notebooks/ (same convention as chunking_util.py):

    cd notebooks
    python tokenizer_gap_check.py
"""

import pandas as pd
import tiktoken
from chunking_evaluation.chunking import RecursiveTokenChunker
from transformers import AutoTokenizer

from chunking_util import DATA_PATH, build_clean_df, build_chunks_df

# --- the chunking config we're measuring (the pre-ModernBERT one) ---
OLD_WINDOW_SIZE = 768
OLD_OVERLAP = 192
OLD_ENCODING_NAME = "cl100k_base"

# --- the model we're measuring against ---
MODEL_NAME = "answerdotai/ModernBERT-base"
MAX_LENGTH = 768        # planned model input length, [CLS]/[SEP] included
N_SPECIAL_TOKENS = 2    # ModernBERT wraps every input as [CLS] ... [SEP]


def kept_char_end(offsets, n_content_tokens_kept):
    """Char position (within the chunk) where the text the model actually sees stops,
    given the chunk's token offsets and how many content tokens survive truncation."""
    if len(offsets) <= n_content_tokens_kept:
        return None                                             # nothing gets cut
    return offsets[n_content_tokens_kept - 1][1]                # end char of the last token that survives


def main():
    encoding = tiktoken.get_encoding(OLD_ENCODING_NAME)

    def cl100k_len(text: str) -> int:
        return len(encoding.encode(text))

    print(f"loading and cleaning CUAD data from {DATA_PATH} ...")
    clean_df = build_clean_df(DATA_PATH)

    chunker = RecursiveTokenChunker(
        chunk_size=OLD_WINDOW_SIZE,
        chunk_overlap=OLD_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=cl100k_len,
    )
    print(f"chunking with the current config ({OLD_WINDOW_SIZE}/{OLD_OVERLAP}, {OLD_ENCODING_NAME}) ...")
    chunks_df = build_chunks_df(chunker, clean_df)
    chunks_df = chunks_df[chunks_df["chunk_start"] >= 0].reset_index(drop=True)
    print(f"{len(chunks_df)} chunks across {chunks_df['contract_id'].nunique()} contracts")

    # --- token counts under both tokenizers ---
    print(f"tokenizing every chunk with {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoded = tokenizer(
        chunks_df["chunk_text"].tolist(),
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    chunks_df["cl100k_tokens"] = chunks_df["chunk_text"].apply(cl100k_len)
    chunks_df["model_tokens"] = [len(ids) + N_SPECIAL_TOKENS for ids in encoded["input_ids"]]
    chunks_df["ratio"] = chunks_df["model_tokens"] / chunks_df["cl100k_tokens"]

    print("\n=== Chunk length: cl100k vs ModernBERT (ModernBERT count includes [CLS]/[SEP]) ===")
    quantiles = [0.5, 0.9, 0.95, 0.99, 1.0]
    summary = chunks_df[["cl100k_tokens", "model_tokens", "ratio"]].quantile(quantiles)
    summary.index = ["median", "p90", "p95", "p99", "max"]
    print(summary.round(3).to_string())

    # --- which chunks would get truncated at MAX_LENGTH, and where the cut lands ---
    n_content_kept = MAX_LENGTH - N_SPECIAL_TOKENS
    chunks_df["kept_end"] = [
        chunks_df.at[i, "chunk_end"] if cut is None else chunks_df.at[i, "chunk_start"] + cut
        for i, cut in enumerate(kept_char_end(offs, n_content_kept) for offs in encoded["offset_mapping"])
    ]
    over = chunks_df[chunks_df["model_tokens"] > MAX_LENGTH]
    print(f"\nchunks over {MAX_LENGTH} ModernBERT tokens: {len(over)} / {len(chunks_df)} "
          f"({len(over) / len(chunks_df) * 100:.2f}%)")
    if len(over):
        cut_tokens = over["model_tokens"] - MAX_LENGTH
        print(f"tokens cut off per over-limit chunk: median={cut_tokens.median():.0f}  max={cut_tokens.max():.0f}")

    # --- what that truncation does to the evidence ---
    evidenced = clean_df[~clean_df["is_impossible"]].copy()
    evidenced["annotation_start"] = evidenced["annotation_start"].astype(int)
    evidenced["annotation_end"] = evidenced["annotation_start"] + evidenced["annotation_text"].str.len()

    n_preserved_before = n_preserved_after = 0
    lost_positive_labels = set()     # (chunk_id, category) positives whose evidence is entirely in the cut-off tail
    kept_positive_labels = set()     # (chunk_id, category) positives with at least some evidence the model still sees
    for contract_id, group in evidenced.groupby("contract_id"):
        contract_chunks = chunks_df[chunks_df["contract_id"] == contract_id]
        starts = contract_chunks["chunk_start"].values
        ends = contract_chunks["chunk_end"].values
        kept_ends = contract_chunks["kept_end"].values
        chunk_ids = contract_chunks["chunk_id"].values
        for ev in group.itertuples(index=False):
            a, b = ev.annotation_start, ev.annotation_end
            # "preserved" = some chunk holds the whole span -- before vs after the model truncates
            n_preserved_before += bool(((starts <= a) & (ends >= b)).any())
            n_preserved_after += bool(((starts <= a) & (kept_ends >= b)).any())
            # chunk labels are overlap-based, so check overlap with the full chunk vs the kept part
            for cid, s, e, ke in zip(chunk_ids, starts, ends, kept_ends):
                if a < e and b > s:
                    key = (cid, ev.category)
                    if a < ke:
                        kept_positive_labels.add(key)
                    else:
                        lost_positive_labels.add(key)
    # a (chunk, category) label only goes bad if *none* of its evidence survives in that chunk
    bad_labels = lost_positive_labels - kept_positive_labels
    n_positive_labels = len(lost_positive_labels | kept_positive_labels)

    n_spans = len(evidenced)
    print("\n=== Impact of truncating at the model's input limit ===")
    print(f"evidence spans fully inside some chunk, before truncation: {n_preserved_before}/{n_spans} "
          f"({n_preserved_before / n_spans * 100:.2f}%)")
    print(f"evidence spans fully inside some chunk, after truncation:  {n_preserved_after}/{n_spans} "
          f"({n_preserved_after / n_spans * 100:.2f}%)")
    print(f"positive (chunk, category) labels whose evidence would be entirely cut off: "
          f"{len(bad_labels)}/{n_positive_labels} ({len(bad_labels) / n_positive_labels * 100:.2f}%)")
    if bad_labels:
        by_cat = pd.Series([cat for _, cat in bad_labels]).value_counts()
        print("most affected categories:")
        print(by_cat.head(10).to_string())

    # --- a suggested safety margin for the new chunker, from the observed ratio ---
    p99_ratio = chunks_df["ratio"].quantile(0.99)
    print(f"\np99 ModernBERT/cl100k ratio: {p99_ratio:.3f} -- a {OLD_WINDOW_SIZE}-token cl100k window "
          f"is roughly {OLD_WINDOW_SIZE * p99_ratio:.0f} ModernBERT tokens in the worst 1% of chunks")


if __name__ == "__main__":
    main()
