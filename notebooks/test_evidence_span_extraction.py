import unittest
from pathlib import Path

import pandas as pd

from evidence_span_extraction import (
    CATEGORY_CUES,
    calculate_overlap,
    extract_evidence,
    merge_overlapping_hits,
    sentence_spans,
)

REPO_DIR = Path(__file__).resolve().parent.parent


class EvidenceSpanExtractionTests(unittest.TestCase):
    def test_cues_cover_all_model_categories(self):
        thresholds = pd.read_csv(REPO_DIR / "results" / "error_analysis" / "selected_thresholds.csv")
        self.assertEqual(set(thresholds["category"]), set(CATEGORY_CUES))

    def test_sentence_spans_preserve_exact_substrings(self):
        text = "First clause. Second clause!\nThird clause"
        self.assertEqual(
            [text[start:end] for start, end in sentence_spans(text)],
            ["First clause.", "Second clause!", "Third clause"],
        )

    def test_regex_selects_sentence_and_contract_offsets(self):
        text = "General recital. Either party may terminate without cause on thirty days' notice."
        hit = extract_evidence(text, 0, "Termination For Convenience", text)[0]
        self.assertEqual(hit["method"], "category_regex")
        self.assertEqual(text[hit["start"]:hit["end"]], hit["evidence_text"])
        self.assertTrue(hit["evidence_text"].startswith("Either party"))

    def test_scored_approach_keeps_sentence_with_most_distinct_cues(self):
        first = "Either party may terminate this Agreement without cause on thirty days' written notice."
        second = "The parties may terminate at any time."
        text = f"{first} {second}"
        hits = extract_evidence(text, 0, "Termination For Convenience", text, approach="scored")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["method"], "category_cue_sentence_score")
        self.assertEqual(hits[0]["evidence_text"], first)

    def test_rejects_bad_chunk_offset(self):
        with self.assertRaises(ValueError):
            extract_evidence("chunk text", 0, "Audit Rights", "different contract")

    def test_overlapping_hits_merge_and_rebuild_text_from_contract(self):
        contract = "0123456789abcdefghij"
        hits = pd.DataFrame([
            {"contract_id": "c1", "category": "Audit Rights", "prob": 0.7,
             "evidence_text": "23456", "start": 2, "end": 7, "method": "category_regex",
             "chunk_ids": ["chunk0"]},
            {"contract_id": "c1", "category": "Audit Rights", "prob": 0.8,
             "evidence_text": "6789a", "start": 6, "end": 11, "method": "longest_sentence_fallback",
             "chunk_ids": ["chunk1"]},
        ])
        merged = merge_overlapping_hits(hits, {"c1": contract})
        self.assertEqual(len(merged), 1)
        self.assertEqual((merged.iloc[0].start, merged.iloc[0].end), (2, 11))
        self.assertEqual(merged.iloc[0].evidence_text, contract[2:11])
        self.assertEqual(merged.iloc[0].prob, 0.8)
        self.assertEqual(merged.iloc[0].chunk_ids, ["chunk0", "chunk1"])

    def test_overlap_metrics_match_contract_and_category(self):
        predictions = pd.DataFrame([
            {"contract_id": "c1", "category": "Audit Rights", "start": 10, "end": 20},
            {"contract_id": "c2", "category": "Audit Rights", "start": 10, "end": 20},
        ])
        gold = pd.DataFrame([
            {"contract_id": "c1", "category": "Audit Rights",
             "annotation_start": 15, "annotation_end": 25},
        ])
        metrics = calculate_overlap(predictions, gold)
        self.assertEqual(metrics["overlap_rate"], 0.5)
        self.assertEqual(metrics["gold_span_coverage_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
