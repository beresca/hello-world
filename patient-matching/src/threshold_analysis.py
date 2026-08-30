"""
Step 7: set operating thresholds for the Tier 3 (Splink) probabilistic scorer.

WHAT THIS ANSWERS:
Step 6 built the evaluation harness and left "pick an actual decision
threshold" as an open question, since a match_probability score by itself
isn't a decision -- something has to translate it into an action. This
script computes the full precision/recall tradeoff across every possible
threshold on the realistic, imbalanced full cross-join (not the curated
500-pair set), then defines a THREE-WAY policy rather than a single
cutoff:

  score >= AUTO_MATCH_THRESHOLD   -> auto-match, no human review
  REVIEW_FLOOR <= score < AUTO_MATCH_THRESHOLD -> send to manual review
  score < REVIEW_FLOOR            -> treat as a non-match, no action

WHY A THREE-WAY POLICY INSTEAD OF ONE THRESHOLD:
A single cutoff forces every borderline pair into either "confidently
correct" or "confidently wrong," with no room for genuine uncertainty.
Given the framing that a false-positive MERGE (combining two different
patients' records) is the costly error here -- far worse than a record
that just needs a person to glance at it -- it's worth spending some
manual review capacity to avoid ever auto-matching on shaky evidence,
while also not silently discarding a real match that scored low due to
noisy fields (per Step 6, that's a known failure mode: short last names
losing recall even with phonetic matching in the model).

HOW THE TWO THRESHOLDS WERE CHOSEN (see NOTES.md "Step 7" for the full
reasoning and the explicit caveat about how far to trust these exact
numbers):
  - AUTO_MATCH_THRESHOLD is the lowest probability in this evaluation that
    produces zero observed false positives across all 250,000 candidate
    comparisons -- the most conservative cut point the data supports.
  - REVIEW_FLOOR is set just below the lowest probability any true match
    ever received in this evaluation, so that no genuine match is ever
    silently auto-rejected without a human seeing it.

Both numbers are a starting point pinned to THIS synthetic evaluation,
not a universal constant -- see the caveat in NOTES.md before treating
them as final.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from probabilistic_matcher import build_settings, load_record_tables, train_and_predict

AUTO_MATCH_THRESHOLD = 0.995
REVIEW_FLOOR = 0.015


def get_full_predictions() -> pd.DataFrame:
    """Retrain Tier 3 and score the full 500x500 EMS-to-EHR cross-join, with
    ground truth attached. Re-run rather than read from disk for the same
    reason evaluate_matching.py does: the 56MB cross-join file isn't
    committed to git, and retraining this tiny dataset takes well under a
    second."""
    ems_df, ehr_df, pairs = load_record_tables()
    n_true_matches = int((pairs["is_match"] == 1).sum())
    prior = n_true_matches / (len(ems_df) * len(ehr_df))
    settings = build_settings(prior)
    preds, _linker = train_and_predict(ems_df, ehr_df, settings)

    l_sources = preds["source_dataset_l"].unique()
    r_sources = preds["source_dataset_r"].unique()
    assert list(l_sources) == ["ehr"] and list(r_sources) == ["ems"], (
        f"Unexpected l/r source assignment: l={l_sources}, r={r_sources}."
    )
    preds = preds.rename(columns={"unique_id_l": "ehr_record_id", "unique_id_r": "ems_record_id"})

    labeled = pairs[["ems_record_id", "ehr_record_id", "is_match"]]
    full = preds.merge(labeled, on=["ems_record_id", "ehr_record_id"], how="left")
    full["is_match"] = full["is_match"].fillna(0).astype(int)
    return full


def band_summary(df: pd.DataFrame, auto_match_threshold: float, review_floor: float) -> None:
    band_auto = df[df["match_probability"] >= auto_match_threshold]
    band_review = df[(df["match_probability"] >= review_floor) & (df["match_probability"] < auto_match_threshold)]
    band_reject = df[df["match_probability"] < review_floor]

    total_true = int(df["is_match"].sum())

    print(f"=== Three-way policy: auto-match >= {auto_match_threshold}, "
          f"review >= {review_floor}, else reject ===\n")

    for name, band in [
        (f"AUTO-MATCH (>= {auto_match_threshold})", band_auto),
        (f"REVIEW QUEUE [{review_floor}, {auto_match_threshold})", band_review),
        (f"AUTO-REJECT (< {review_floor})", band_reject),
    ]:
        n = len(band)
        tm = int((band["is_match"] == 1).sum())
        nm = int((band["is_match"] == 0).sum())
        print(f"{name:36} n={n:>8,}  true_matches={tm:>4}  non_matches={nm:>8,}")

    tp = int((band_auto["is_match"] == 1).sum())
    fp = int((band_auto["is_match"] == 0).sum())
    missed = int((band_reject["is_match"] == 1).sum())
    review_true = int((band_review["is_match"] == 1).sum())

    print()
    print(f"Auto-match precision: {tp}/{tp + fp} = {tp / (tp + fp) if (tp+fp) else float('nan'):.6f}")
    print(f"Fraction of all true matches auto-matched: {tp}/{total_true} = {tp / total_true:.4f}")
    print(f"True matches sent to manual review instead: {review_true} "
          f"({review_true / total_true:.1%} of all true matches)")
    print(f"True matches silently auto-rejected (missed entirely): {missed}")
    print(f"Review queue size as a fraction of all candidate pairs: "
          f"{len(band_review) / len(df):.4%} ({len(band_review):,} pairs)")


def main():
    print("Retraining Tier 3 and scoring the full cross-join "
          "(reusing src/probabilistic_matcher.py; a few seconds)...\n")
    full = get_full_predictions()
    print(f"Full cross-join: {len(full):,} comparisons "
          f"({int((full.is_match==1).sum())} true matches)\n")

    y_true = full["is_match"].values
    y_score = full["match_probability"].values
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    precision, recall = precision[:-1], recall[:-1]
    curve = pd.DataFrame({"threshold": thresholds, "precision": precision, "recall": recall})

    zero_fp = curve[curve["precision"] >= 0.999999].sort_values("threshold")
    print("Lowest threshold with zero observed false positives (candidate auto-match cutoff):")
    print(zero_fp.head(1).to_string(index=False))
    print()

    perfect_recall = curve[curve["recall"] >= 0.999999].sort_values("threshold", ascending=False)
    print("Highest threshold that still catches every true match (guides the review floor):")
    print(perfect_recall.head(1).to_string(index=False))
    print()

    curve.to_csv("data/threshold_precision_recall_curve.csv", index=False)
    print(f"Wrote full precision/recall-by-threshold curve "
          f"({len(curve)} points) to data/threshold_precision_recall_curve.csv\n")

    band_summary(full, AUTO_MATCH_THRESHOLD, REVIEW_FLOOR)


if __name__ == "__main__":
    main()
