
"""
Phase 7B: CUAD Validation Error Analysis

Analyze false positives and false negatives from
the existing validation results.
"""

from pathlib import Path
import pandas as pd

# Load existing error files
ROOT = Path(__file__).resolve().parent.parent
FOLDER = ROOT / "results" / "error_analysis"

fp = pd.read_csv(FOLDER / "fp_examples.csv")
fn = pd.read_csv(FOLDER / "fn_examples.csv")

assert (fp["error"] == "FP").all()
assert (fn["error"] == "FN").all()
assert len(fp) >= 10 and len(fn) >= 10

print("False positives:", len(fp))
print("False negatives:", len(fn))

# Combine errors
review = pd.concat([fp, fn], ignore_index=True)

# Calculate chunk lengths
review["chunk_word_count"] = (
    review["chunk_text"].fillna("").str.split().str.len()
)

review["length_group"] = pd.cut(
    review["chunk_word_count"],
    bins=[-1, 299, 499, float("inf")],
    labels=["Short", "Medium", "Long"]
)

# Identify 10 required examples of each error type
review["required_review"] = (
    review.groupby("error").cumcount() < 10
)

# Review notes based on the supplied examples
# These are possible explanations, not verified causes.

review_notes = {
    ("FP", "Anti-Assignment"):
        ("Conflicting agreement restrictions may resemble "
         "assignment restrictions.",
         "Add negative examples distinguishing these clauses."),

    ("FP", "Audit Rights"):
        ("The text contains an audit procedure, suggesting "
         "a possible annotation mismatch.",
         "Check the original audit-rights annotation."),

    ("FP", "Governing Law"):
        ("The text explicitly mentions Choice of Law, "
         "suggesting a possible labeling mismatch.",
         "Check the gold label and chunk alignment."),

    ("FP", "Parties"):
        ("Party names appear in the chunk, but the "
         "annotation may be assigned to another chunk.",
         "Review neighboring chunks and labels."),

    ("FP", "Document Name"):
        ("The document title appears explicitly, "
         "suggesting a possible annotation mismatch.",
         "Check document-name annotation boundaries."),

    ("FP", "Post-Termination Services"):
        ("Continued supply obligations may resemble "
         "post-termination services.",
         "Add examples distinguishing these obligations."),

    ("FP", "Renewal Term"):
        ("References to continuing an agreement may "
         "be confused with actual renewal provisions.",
         "Add negative examples of ongoing agreements."),

    ("FP", "Agreement Date"):
        ("Multiple dates in an amendment may create "
         "ambiguity about the agreement date.",
         "Check which date the annotation identifies."),

    ("FP", "Minimum Commitment"):
        ("A purchase quantity may resemble a minimum "
         "purchase commitment.",
         "Distinguish fixed quantities from minimum obligations."),

    ("FP", "License Grant"):
        ("IP-related language may resemble a license grant "
         "without actually granting a license.",
         "Add hard negatives involving IP disclosures."),

    ("FP", "Revenue/Profit Sharing"):
        ("Royalty language may resemble revenue sharing.",
         "Clarify the distinction in training examples."),

    ("FP", "Exclusivity"):
        ("License restrictions may resemble exclusivity "
         "while the relevant clause is in another chunk.",
         "Check neighboring chunks and clause definitions."),

    ("FP", "Warranty Duration"):
        ("Payment deadlines may be confused with "
         "warranty duration.",
         "Add negative examples involving payment deadlines."),

    ("FP", "Effective Date"):
        ("The effective date appears explicitly, "
         "suggesting a possible annotation mismatch.",
         "Verify the effective-date label."),

    ("FP", "Cap On Liability"):
        ("Indemnification limits may resemble a liability cap.",
         "Distinguish indemnity procedures from liability caps."),

    ("FN", "Irrevocable Or Perpetual License"):
        ("Perpetual rights are expressed indirectly.",
         "Include varied wording for perpetual licenses."),

    ("FN", "Ip Ownership Assignment"):
        ("Ownership transfer is described using different wording.",
         "Include phrases such as give title and convey rights."),

    ("FN", "Termination For Convenience"):
        ("Service termination may be confused with "
         "termination of the full agreement.",
         "Clarify the category definition."),

    ("FN", "Anti-Assignment"):
        ("Facility disposition language may not clearly "
         "describe assignment restrictions.",
         "Review the annotation and category scope."),

    ("FN", "Revenue/Profit Sharing"):
        ("The provided evidence span is too limited "
         "to clearly indicate revenue sharing.",
         "Check evidence-span quality."),

    ("FN", "Uncapped Liability"):
        ("Injunction language may not clearly indicate "
         "uncapped monetary liability.",
         "Review the annotation and remedy definitions."),

    ("FN", "Cap On Liability"):
        ("The clause refers to remedies elsewhere "
         "rather than stating an explicit cap.",
         "Check whether cross-references qualify."),

    ("FN", "Third Party Beneficiary"):
        ("The clause is embedded within a larger section.",
         "Include examples from longer mixed-topic passages.")
}

# Add review notes to the dataframe
def get_note(row, position):
    note = review_notes.get(
        (row["error"], row["category"])
    )
    if note is None:
        return "Needs manual review"
    return note[position]


review["likely_cause"] = review.apply(
    lambda row: get_note(row, 0), axis=1
)

review["suggested_fix"] = review.apply(
    lambda row: get_note(row, 1), axis=1
)

review["review_status"] = (
    "Proposed explanation - needs verification"
)

# Analyze errors by category
print("\n=== Errors by Category ===")

category_counts = pd.crosstab(
    review["category"],
    review["error"]
)

print(category_counts)

# Analyze errors by chunk length
print("\n=== Errors by Chunk Length ===")

length_counts = pd.crosstab(
    review["error"],
    review["length_group"]
)

print(length_counts)

print("\nAverage chunk length:")
print(
    review.groupby("error")["chunk_word_count"]
    .mean()
    .round(1)
)

# Existing false positive patterns
print("\n=== False Positive Patterns ===")
print(fp["pattern"].value_counts())

# False negative evidence containment
print("\n=== False Negative Evidence ===")
print(fn["fully_contained"].value_counts(dropna=False))

# Prediction probabilities
print("\n=== Prediction Probabilities ===")

print(
    review.groupby("error")["prob"]
    .agg(["min", "median", "max"])
    .round(4)
)

# Save review results without overwriting existing files
output = FOLDER / "phase7b_my_error_review.csv"

if not output.exists():
    review.to_csv(output, index=False)
    print("\nSaved review:", output)
else:
    print("\nReview CSV already exists; not overwritten.")

# --------------------------------------------------
# Findings and recommendations
# --------------------------------------------------

print("\n=== Phase 7B Findings ===")

print("""
1. FALSE POSITIVES

Several false positives appear to involve confusion
between similar legal concepts, such as license grants
and other intellectual property provisions.

Some examples contain explicit clause evidence despite
being labeled false positives. These may indicate
annotation or chunk-alignment issues.

2. FALSE NEGATIVES

All 15 supplied false-negative examples have annotated
evidence fully contained within their chunks.

This suggests that missing context across chunk
boundaries does not fully explain these errors.

Some missed clauses use indirect legal wording,
while others contain potentially ambiguous annotations.

3. CHUNK LENGTH

The selected false-negative examples tend to contain
longer chunks than the false-positive examples.

Longer chunks may make relevant clauses harder to
identify, although this sample cannot establish
a causal relationship.

4. CATEGORY AND WORDING PATTERNS

Possible issues include:
- Confusion between similar clause categories
- Indirect or uncommon legal terminology
- Relevant evidence embedded in longer sections
- Annotation inconsistencies
- Context split across neighboring chunks

5. SUGGESTED IMPROVEMENTS

- Add hard-negative examples for similar clauses.
- Include more varied legal wording in training data.
- Review potentially inconsistent CUAD annotations.
- Examine neighboring chunks for boundary-related errors.
- Test shorter chunks or additional context for
  long passages.

6. LIMITATIONS

This analysis uses 15 selected false positives and
15 selected false negatives, not the full validation set.

The explanations are hypotheses based on the exported
text and annotations, not verified model root causes.

The proposed improvements have not been tested.
""")