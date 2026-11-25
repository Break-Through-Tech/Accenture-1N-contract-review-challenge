"""Extract sentence evidence from positive CUAD validation chunks.

Run from the repository root after ``main_pipeline.ipynb`` and ``error_analysis.ipynb``
have produced their parquet and threshold files:

    python notebooks/evidence_span_extraction.py

This uses saved ``ensemble_top2`` probabilities and tune-selected thresholds, so
model weight files are not needed. Extracted offsets refer to cleaned contract
text, and ``prob`` remains the chunk-level probability rather than a sentence score.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

REPO_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_DIR / "data" / "processed"
RESULTS_DIR = REPO_DIR / "results"
OUTPUT_DIR = RESULTS_DIR / "evidence_span_extraction"
ENSEMBLE_RUNS = ("modernbert_v2_biasinit", "modernbert_v1_6ep")
EXPECTED_CATEGORIES = 41

# These are localization cues only. The chunk classifier remains responsible
# for deciding whether the category is present in a chunk.
CATEGORY_CUES = {
    "Affiliate License-Licensee": (r"\blicensee'?s? affiliates?\b", r"\blicensee affiliates?\b"),
    "Affiliate License-Licensor": (r"\blicensor'?s? affiliates?\b", r"\blicensor affiliates?\b"),
    "Agreement Date": (r"\bdated as of\b", r"\bmade (?:as of|on)\b", r"\bentered into (?:as of|on)\b"),
    "Anti-Assignment": (r"\bassign(?:ment|ed)?\b", r"\btransfer(?:red)?\b.{0,60}\bconsent\b"),
    "Audit Rights": (r"\baudit\b", r"\binspect\b.{0,50}\b(?:books|records|premises)\b"),
    "Cap On Liability": (r"\bcap on liability\b", r"\blimitation of liability\b", r"\bliability\b.{0,50}\bcap\b"),
    "Change Of Control": (r"\bchange of control\b", r"\bchange in control\b"),
    "Competitive Restriction Exception": (r"\bexception\b.{0,60}\b(?:compete|exclusiv|solicit)\b", r"\bcarve[- ]?out\b"),
    "Covenant Not To Sue": (r"\bcovenant not to sue\b", r"\bwaive.{0,40}\bclaim\b"),
    "Document Name": (r"\b(?:agreement|contract)\b",),
    "Effective Date": (r"\beffective date\b", r"\beffective as of\b", r"\bcommencement date\b"),
    "Exclusivity": (r"\bexclusiv(?:e|ity|ely)\b", r"\bsole supplier\b", r"\ball requirements\b"),
    "Expiration Date": (r"\bexpiration\b", r"\bexpire[sd]?\b", r"\binitial term\b"),
    "Governing Law": (r"\bgoverned by\b", r"\bgoverning law\b", r"\blaws of\b", r"\bchoice of law\b"),
    "Insurance": (r"\binsurance\b", r"\binsured\b", r"\bcoverage\b"),
    "Ip Ownership Assignment": (r"\bassign(?:ment)?\b.{0,60}\b(?:intellectual property|ownership|title)\b",
                                r"\bright, title and interest\b"),
    "Irrevocable Or Perpetual License": (r"\birrevocable\b.{0,60}\blicen[cs]e\b",
                                         r"\bperpetual\b.{0,60}\blicen[cs]e\b"),
    "Joint Ip Ownership": (r"\bjoint(?:ly)? own(?:ed|ership)?\b", r"\bco[- ]own(?:ed|ership)?\b"),
    "License Grant": (r"\bgrant(?:s|ed)?\b.{0,80}\blicen[cs]", r"\blicen[cs]e grant\b"),
    "Liquidated Damages": (r"\bliquidated damages\b", r"\btermination fee\b"),
    "Minimum Commitment": (r"\bminimum\b.{0,60}\b(?:purchase|order|commitment|quantity|units?)\b",
                           r"\bminimum commitment\b"),
    "Most Favored Nation": (r"\bmost[- ]favou?red nation\b", r"\bmore favorable terms\b",
                            r"\bless favorable terms\b"),
    "No-Solicit Of Customers": (r"\bsolicit(?:ation)?\b.{0,60}\b(?:customer|client|partner)s?\b",
                                r"\b(?:customer|client|partner)s?\b.{0,60}\bsolicit(?:ation)?\b"),
    "No-Solicit Of Employees": (r"\bsolicit(?:ation)?\b.{0,60}\b(?:employee|personnel|contractor)s?\b",
                                r"\bnot hire\b"),
    "Non-Compete": (r"\bnon[- ]compete\b", r"\bcompete\b", r"\bcompetitive restriction\b"),
    "Non-Disparagement": (r"\bnon[- ]disparag", r"\bdisparag(?:e|ement)\b"),
    "Non-Transferable License": (r"\bnon[- ]transferable\b", r"\btransfer\b.{0,60}\blicen[cs]e\b",
                                 r"\bsublicen[cs]e\b"),
    "Notice Period To Terminate Renewal": (r"\bnon[- ]renewal\b", r"\bnotice\b.{0,60}\brenewal\b",
                                           r"\bterminate\b.{0,60}\brenewal\b"),
    "Parties": (r"\bby and between\b", r"\bparties hereto\b", r"\b(?:company|customer|client)\b.{0,40}\b(inc\.?|llc|ltd\.?)\b"),
    "Post-Termination Services": (r"\bafter (?:the )?(?:termination|expiration)\b", r"\bpost[- ]termination\b",
                                  r"\btransition assistance\b"),
    "Price Restrictions": (r"\bprice\b.{0,60}\b(?:increase|decrease|change|adjust)\b",
                           r"\b(?:increase|decrease|change|adjust)\b.{0,60}\bprice\b"),
    "Renewal Term": (r"\brenew(?:al|ed|ing)?\b", r"\bsuccessive term\b", r"\bextension\b"),
    "Revenue/Profit Sharing": (r"\bshare\b.{0,60}\b(?:revenue|profit)s?\b",
                               r"\b(?:revenue|profit)s?\b.{0,60}\bshare\b", r"\broyalt(?:y|ies)\b"),
    "Rofr/Rofo/Rofn": (r"\bright of first refusal\b", r"\bright of first offer\b",
                       r"\bright of first negotiation\b"),
    "Source Code Escrow": (r"\bsource code\b.{0,60}\bescrow\b", r"\bescrow\b.{0,60}\bsource code\b"),
    "Termination For Convenience": (r"\bterminate\b.{0,60}\bwithout cause\b",
                                    r"\btermination for convenience\b"),
    "Third Party Beneficiary": (r"\bthird[- ]party beneficiar", r"\bbeneficiar(?:y|ies)\b.{0,60}\bthird[- ]party\b"),
    "Uncapped Liability": (r"\bunlimited liability\b", r"\bwithout limitation\b.{0,60}\bliabilit",
                           r"\bno cap\b.{0,60}\bliabilit"),
    "Unlimited/All-You-Can-Eat-License": (r"\bunlimited\b.{0,60}\blicen[cs]e\b", r"\ball[- ]you[- ]can[- ]eat\b",
                                         r"\benterprise[- ]wide licen[cs]e\b"),
    "Volume Restriction": (r"\bvolume\b.{0,60}\b(?:limit|threshold|restriction|exceed)\b",
                           r"\b(?:usage|volume)\b.{0,60}\bexceed\b"),
    "Warranty Duration": (r"\bwarranty\b.{0,60}\b(?:period|month|year|duration)\b",
                          r"\b(?:period|month|year|duration)\b.{0,60}\bwarrant"),
}

SENTENCE_BOUNDARY = re.compile(r"[.!?]+[\"')\]]*(?=\s|$)|\n+")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Split at sentence punctuation/newlines and return trimmed half-open offsets."""
    spans = []
    start = 0
    for boundary in SENTENCE_BOUNDARY.finditer(text):
        end = boundary.end()
        left, right = start, end
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if left < right:
            spans.append((left, right))
        start = end
    left, right = start, len(text)
    while left < right and text[left].isspace():
        left += 1
    while right > left and text[right - 1].isspace():
        right -= 1
    if left < right:
        spans.append((left, right))
    return spans


def extract_evidence(
    chunk_text: str,
    chunk_start: int,
    category: str,
    contract_text: str,
    approach: str = "regex",
) -> list[dict]:
    """Select cue-matched sentences or a longest-sentence fallback from a positive chunk.

    ``regex`` keeps every sentence matching at least one category cue.
    ``scored`` keeps only the sentence(s) matching the most distinct cues.
    """
    if category not in CATEGORY_CUES:
        raise ValueError(f"no evidence cues configured for category {category!r}")
    if approach not in {"regex", "scored"}:
        raise ValueError(f"unknown sentence extraction approach {approach!r}")
    chunk_end = chunk_start + len(chunk_text)
    if chunk_start < 0 or contract_text[chunk_start:chunk_end] != chunk_text:
        raise ValueError("chunk text does not match contract text at its recorded offsets")
    spans = sentence_spans(chunk_text)
    cues = [re.compile(pattern, re.IGNORECASE) for pattern in CATEGORY_CUES[category]]
    scores = {
        span: sum(bool(cue.search(chunk_text[span[0]:span[1]])) for cue in cues)
        for span in spans
    }
    if approach == "scored":
        best_score = max(scores.values(), default=0)
        selected = [span for span in spans if best_score and scores[span] == best_score]
        method = "category_cue_sentence_score"
    else:
        selected = [span for span in spans if scores[span]]
        method = "category_regex"
    if not selected and spans:
        selected = [max(spans, key=lambda span: span[1] - span[0])]
        method = "longest_sentence_fallback"
    return [
        {
            "category": category,
            "evidence_text": contract_text[chunk_start + start:chunk_start + end],
            "start": chunk_start + start,
            "end": chunk_start + end,
            "method": method,
        }
        for start, end in selected
    ]


def merge_overlapping_hits(predictions: pd.DataFrame, contract_texts: dict[str, str]) -> pd.DataFrame:
    """Merge overlapping hits by contract/category and rebuild text from contract offsets."""
    columns = list(predictions.columns)
    if predictions.empty:
        return predictions.copy()
    result = []
    ordered = predictions.sort_values(["contract_id", "category", "start", "end"], kind="stable")
    for (contract_id, category), group in ordered.groupby(["contract_id", "category"], sort=False):
        active = None
        for hit in group.to_dict("records"):
            if active is not None and hit["start"] < active["end"]:
                active["end"] = max(active["end"], hit["end"])
                active["prob"] = max(active["prob"], hit["prob"])
                active["chunk_ids"] = sorted(set(active["chunk_ids"]) | set(hit["chunk_ids"]))
                if hit["method"] == "category_regex":
                    active["method"] = "category_regex"
            else:
                if active is not None:
                    active["evidence_text"] = contract_texts[contract_id][active["start"]:active["end"]]
                    result.append(active)
                active = hit
        if active is not None:
            active["evidence_text"] = contract_texts[contract_id][active["start"]:active["end"]]
            result.append(active)
    return pd.DataFrame(result, columns=columns).sort_values(
        ["contract_id", "start", "category"], kind="stable"
    ).reset_index(drop=True)


def calculate_overlap(predictions: pd.DataFrame, gold: pd.DataFrame) -> dict:
    """Calculate category/contract-matched prediction overlap and gold-span coverage."""
    gold = gold.drop_duplicates(["contract_id", "category", "annotation_start", "annotation_end"])
    gold_by_pair = {
        key: list(zip(group["annotation_start"], group["annotation_end"]))
        for key, group in gold.groupby(["contract_id", "category"], sort=False)
    }
    hit_count = sum(
        any(row.start < end and row.end > start
            for start, end in gold_by_pair.get((row.contract_id, row.category), []))
        for row in predictions.itertuples(index=False)
    )
    covered_count = 0
    for row in gold.itertuples(index=False):
        pair_hits = predictions[
            (predictions["contract_id"] == row.contract_id)
            & (predictions["category"] == row.category)
        ]
        if ((pair_hits["start"] < row.annotation_end) & (pair_hits["end"] > row.annotation_start)).any():
            covered_count += 1
    return {
        "overlap_rate": hit_count / len(predictions) if len(predictions) else 0.0,
        "gold_span_coverage_rate": covered_count / len(gold) if len(gold) else 0.0,
        "predictions": len(predictions),
        "overlapping_predictions": hit_count,
        "gold_spans": len(gold),
        "covered_gold_spans": covered_count,
    }


def load_ensemble_probs(results_dir: Path) -> pd.DataFrame:
    """Average saved validation probabilities from the two ensemble runs."""
    frames = [
        pd.read_parquet(results_dir / run / "val_probs.parquet").set_index("chunk_id")
        for run in ENSEMBLE_RUNS
    ]
    first, second = frames
    if set(first.index) != set(second.index):
        raise ValueError("ensemble run probability files have different chunk ids")
    second = second.loc[first.index]
    categories = [column for column in first.columns if column != "contract_id"]
    if categories != [column for column in second.columns if column != "contract_id"]:
        raise ValueError("ensemble probability files have different category columns")
    probabilities = (first[categories].astype(float) + second[categories].astype(float)) / 2
    probabilities.insert(0, "contract_id", first["contract_id"])
    probabilities = probabilities.reset_index()
    if probabilities["chunk_id"].duplicated().any():
        raise ValueError("validation probabilities contain duplicate chunk ids")
    return probabilities


def run(
    data_dir: Path,
    results_dir: Path,
    output_dir: Path,
    n_contracts: int | None = None,
    approach: str = "regex",
) -> dict:
    required = [
        data_dir / "chunks_df.parquet",
        data_dir / "contracts_df.parquet",
        data_dir / "chunk_evidence_df.parquet",
        REPO_DIR / "data" / "splits" / "val_contracts.txt",
        results_dir / "error_analysis" / "selected_thresholds.csv",
        *[results_dir / model / "val_probs.parquet" for model in ENSEMBLE_RUNS],
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing pipeline artifacts:\n" + "\n".join(map(str, missing)))

    validation_ids = [
        contract_id for contract_id in
        (REPO_DIR / "data" / "splits" / "val_contracts.txt").read_text(encoding="utf-8").splitlines()
        if contract_id.strip()
    ]
    if len(validation_ids) != 82 or len(set(validation_ids)) != 82:
        raise ValueError(f"expected 82 unique validation contracts, found {len(set(validation_ids))}")
    selected_ids = set(validation_ids if n_contracts is None else validation_ids[:n_contracts])
    if len(selected_ids) < 5:
        raise ValueError("at least five validation contracts are required")

    chunks = pd.read_parquet(data_dir / "chunks_df.parquet")
    chunks = chunks[chunks["contract_id"].isin(selected_ids) & (chunks["chunk_start"] >= 0)].copy()
    contracts = pd.read_parquet(data_dir / "contracts_df.parquet")
    contracts = contracts[contracts["contract_id"].isin(selected_ids)]
    contract_texts = dict(contracts[["contract_id", "text"]].itertuples(index=False, name=None))
    if set(selected_ids) - set(contract_texts):
        raise ValueError("some selected validation contracts are missing from contracts_df.parquet")

    probs = load_ensemble_probs(results_dir)
    chunks = chunks.merge(probs, on=["chunk_id", "contract_id"], how="inner", validate="one_to_one")
    thresholds = pd.read_csv(results_dir / "error_analysis" / "selected_thresholds.csv")
    if thresholds["category"].duplicated().any() or "threshold_ensemble_top2" not in thresholds:
        raise ValueError("threshold file must have unique categories and threshold_ensemble_top2")
    threshold_by_category = thresholds.set_index("category")["threshold_ensemble_top2"].to_dict()
    categories = [column for column in probs.columns if column not in {"chunk_id", "contract_id"}]
    if len(categories) != EXPECTED_CATEGORIES or set(categories) != set(threshold_by_category):
        raise ValueError("probabilities and tune-selected thresholds must cover the same 41 categories")

    rows = []
    for _, chunk in chunks.iterrows():
        for category in categories:
            probability = float(chunk[category])
            if probability < threshold_by_category[category]:
                continue
            for evidence in extract_evidence(
                chunk["chunk_text"], int(chunk["chunk_start"]), category,
                contract_texts[chunk["contract_id"]], approach=approach,
            ):
                rows.append({
                    "contract_id": chunk["contract_id"],
                    "category": category,
                    "prob": probability,
                    **evidence,
                    "chunk_ids": [chunk["chunk_id"]],
                })
    columns = ["contract_id", "category", "prob", "evidence_text", "start", "end", "method", "chunk_ids"]
    predictions = merge_overlapping_hits(pd.DataFrame(rows, columns=columns), contract_texts)

    gold = pd.read_parquet(data_dir / "chunk_evidence_df.parquet")
    gold = gold[
        gold["contract_id"].isin(selected_ids)
        & gold["annotation_start"].notna()
        & gold["annotation_end"].notna()
    ].copy()
    gold["annotation_start"] = gold["annotation_start"].astype(int)
    gold["annotation_end"] = gold["annotation_end"].astype(int)
    metrics = calculate_overlap(predictions, gold)

    output_dir.mkdir(parents=True, exist_ok=True)
    export = predictions.copy()
    export["chunk_ids"] = export["chunk_ids"].apply(lambda ids: ";".join(ids))
    export.to_csv(output_dir / "predictions.csv", index=False)
    summary = {
        "approach": (
            "all category-cue-matching sentences with longest-sentence fallback"
            if approach == "regex"
            else "highest distinct category-cue sentence score with longest-sentence fallback"
        ),
        "ensemble": list(ENSEMBLE_RUNS),
        "thresholds": "per-category best-F1 cutoffs selected on tune",
        "validation_contracts": sorted(selected_ids),
        "validation_contract_count": len(selected_ids),
        "overlap_definition": "same contract/category and intersecting half-open character offsets",
        "probability_scope": "chunk-level classifier probability; not sentence/span confidence",
        "fallback_predictions": int((predictions["method"] == "longest_sentence_fallback").sum())
        if not predictions.empty else 0,
        **metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-contracts", type=int, default=None, help="default: all 82 validation contracts")
    parser.add_argument("--approach", choices=("regex", "scored"), default="regex")
    args = parser.parse_args()
    output_dir = args.output_dir or OUTPUT_DIR / args.approach
    summary = run(args.data_dir, args.results_dir, output_dir, args.n_contracts, args.approach)
    print(json.dumps({key: value for key, value in summary.items() if key != "validation_contracts"}, indent=2))


if __name__ == "__main__":
    main()
