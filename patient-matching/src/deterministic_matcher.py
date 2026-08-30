"""
Tier 1: deterministic matching.

WHAT THIS TIER DOES:
Deterministic matching only says "these two records are the same person"
when both records carry the *same value* for a strong shared identifier
(here: MRN — Medical Record Number). It never guesses based on how similar
two names look, or how close two dates of birth are. That makes it very
high precision (it should almost never be wrong), but low recall (it can
only catch matches where a strong ID actually exists on both sides).

WHY BLANK-VS-BLANK MUST NOT COUNT AS A MATCH:
Most EMS records in our dataset don't capture an MRN at all -- the field
is simply empty. If we matched on "ems_mrn == ehr_mrn" without also
requiring both sides to be *non-empty*, then two records that both happen
to have blank MRNs would look "equal" (empty string equals empty string)
and get wrongly counted as a confirmed match. That would completely
undermine the "almost never wrong" guarantee this tier exists to provide.
So the rule explicitly requires both values to be present.

WHY THIS TIER ALONE ISN'T ENOUGH:
Most of our synthetic true matches don't have a usable MRN on the EMS
side (that's realistic -- EMS crews usually don't know the hospital's
internal number). Those pairs are genuinely undecidable by this tier, and
that's fine: they're supposed to fall through to a fuzzy/probabilistic
tier (rapidfuzz, then splink) that can reason about noisy fields like
names and partial addresses. This script measures exactly how big that
gap is.
"""

import pandas as pd


# Which columns represent a "strong" shared identifier worth deterministic
# matching on. Our synthetic dataset only has MRN, but writing this as a
# list (rather than hard-coding a single column name everywhere) makes it
# easy to add e.g. "ssn" later if the dataset grows one.
STRONG_ID_FIELD_PAIRS = [
    ("ems_mrn", "ehr_mrn"),
]


def _is_present(value) -> bool:
    """A field 'counts' only if it isn't missing/blank after stripping whitespace."""
    if pd.isna(value):
        return False
    return str(value).strip() != ""


def deterministic_match(row: pd.Series) -> bool:
    """
    Return True if `row` (one EMS/EHR candidate pair) shares an exact,
    non-blank value on at least one strong identifier field.
    """
    for ems_field, ehr_field in STRONG_ID_FIELD_PAIRS:
        ems_value = row[ems_field]
        ehr_value = row[ehr_field]
        if _is_present(ems_value) and _is_present(ehr_value) and str(ems_value).strip() == str(ehr_value).strip():
            return True
    return False


def evaluate(df: pd.DataFrame) -> dict:
    """
    Run deterministic_match over every row and compare against the
    ground-truth `is_match` column. Returns a dict of counts/rates we can
    print or log. This is evaluation-only logic -- the matcher itself
    (deterministic_match) never looks at `is_match`.
    """
    predictions = df.apply(deterministic_match, axis=1)

    n_true_matches = int((df["is_match"] == 1).sum())
    n_non_matches = int((df["is_match"] == 0).sum())

    true_positives = int((predictions & (df["is_match"] == 1)).sum())
    false_positives = int((predictions & (df["is_match"] == 0)).sum())
    n_predicted = int(predictions.sum())

    recall_on_true_matches = true_positives / n_true_matches if n_true_matches else 0.0
    precision = true_positives / n_predicted if n_predicted else float("nan")

    return {
        "n_pairs": len(df),
        "n_true_matches": n_true_matches,
        "n_non_matches": n_non_matches,
        "n_predicted_matches": n_predicted,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "recall_on_true_matches": recall_on_true_matches,
        "precision": precision,
        "predictions": predictions,
    }


def main():
    df = pd.read_csv("data/synthetic_ems_ehr_pairs.csv", dtype=str, keep_default_na=False)
    df["is_match"] = df["is_match"].astype(int)

    results = evaluate(df)

    print("=== Tier 1: Deterministic matching (MRN) ===")
    print(f"Total pairs evaluated:        {results['n_pairs']}")
    print(f"True matches in dataset:      {results['n_true_matches']}")
    print(f"Non-matches in dataset:       {results['n_non_matches']}")
    print()
    print(f"Pairs flagged as a match:     {results['n_predicted_matches']}")
    print(f"  - correct (true positive):  {results['true_positives']}")
    print(f"  - wrong (false positive):   {results['false_positives']}")
    print()
    print(f"Recall on true matches:       {results['recall_on_true_matches']:.1%}  "
          f"(fraction of the 250 real matches this tier alone catches)")
    print(f"Precision:                    {results['precision']:.1%}  "
          f"(fraction of this tier's own guesses that were correct)")

    if results["false_positives"] == 0:
        print("\nNo false matches produced -- consistent with the goal of near-perfect precision.")
    else:
        print(f"\nWARNING: {results['false_positives']} false match(es) produced -- investigate before trusting this tier.")


if __name__ == "__main__":
    main()
