"""
Step 6: evaluation harness for the Tier 3 (Splink) probabilistic scorer.

WHAT THIS ANSWERS THAT STEP 5 DIDN'T:
Step 5 showed the match_probability distributions for true matches and
non-matches don't overlap on our 500 designed (250/250) pairs, and left
"pick an actual decision threshold" as an open question. This script picks
that apart properly: how precision and recall trade off across candidate
thresholds, what a confusion matrix looks like at a given cutoff, and
whether performance holds up evenly across two subgroups worth checking
separately -- pairs where Tier 1 already had a shot via a captured EMS
MRN, and pairs split by how long the last name is (since Step 4.5 and
Step 5 both flagged short names as a specific weak spot for
Jaro-Winkler-based comparisons).

WHY WE RE-RUN THE PIPELINE INSTEAD OF READING A SAVED CSV:
The full 500x500 cross-join predictions file is ~56MB and deliberately
NOT committed to git (see NOTES.md Step 5) -- it's cheap to regenerate
(well under a second, since our whole dataset is tiny) by re-importing
Step 5's own functions and re-running training + prediction, so that's
what this script does rather than assuming a stale file sits on disk.

WHY TWO DIFFERENT DENOMINATORS SHOW UP BELOW:
The threshold/precision/recall table and confusion matrix use the FULL
500 x 500 = 250,000 cross-join, because that's the realistic, heavily
imbalanced picture a real deployment's threshold actually has to survive
(250 true matches against 249,750 candidates that are not the right
person) -- our curated 500-pair, 50/50 file was built for clarity in
Steps 4-5, not to reflect how rare a real match actually is among
candidates. The MRN-presence and last-name-length breakdowns use the 500
designed pairs instead, since "does this EMS record happen to carry an
MRN" and "how long is this last name" are properties of our labeled
evaluation set, not something meaningful to slice the full cross-join by.
"""

import numpy as np
import pandas as pd

from probabilistic_matcher import build_settings, load_record_tables, train_and_predict

THRESHOLDS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]


def confusion_counts(df: pd.DataFrame, threshold: float) -> tuple[int, int, int, int]:
    predicted_match = df["match_probability"] >= threshold
    actual_match = df["is_match"] == 1
    tp = int((predicted_match & actual_match).sum())
    fp = int((predicted_match & ~actual_match).sum())
    fn = int((~predicted_match & actual_match).sum())
    tn = int((~predicted_match & ~actual_match).sum())
    return tp, fp, fn, tn


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
    return precision, recall, f1


def threshold_table(df: pd.DataFrame, thresholds=THRESHOLDS) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        tp, fp, fn, tn = confusion_counts(df, t)
        precision, recall, f1 = precision_recall_f1(tp, fp, fn)
        rows.append({
            "threshold": t, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
        })
    return pd.DataFrame(rows)


def print_confusion_matrix(df: pd.DataFrame, threshold: float, title: str) -> None:
    tp, fp, fn, tn = confusion_counts(df, threshold)
    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    print(f"=== Confusion matrix @ threshold {threshold} -- {title} ===")
    print(f"{'':20}{'predicted match':>18}{'predicted non-match':>22}")
    print(f"{'actual match':20}{tp:>18,}{fn:>22,}")
    print(f"{'actual non-match':20}{fp:>18,}{tn:>22,}")
    print(f"precision={precision:.4f}  recall={recall:.4f}  f1={f1:.4f}\n")


def bucket_last_name_length(row) -> str:
    # Splink's prediction output retains the compared columns as
    # last_name_l/last_name_r (l=ehr, r=ems -- see probabilistic_matcher.py's
    # note on Splink's alphabetical l/r assignment), not the original
    # ems_last_name/ehr_last_name names from the raw dataset.
    longer = max(len(str(row["last_name_l"])), len(str(row["last_name_r"])))
    if longer <= 4:
        return "<=4"
    if longer <= 7:
        return "5-7"
    return "8+"


def main():
    print("Retraining the Tier 3 model and generating fresh predictions "
          "(reusing src/probabilistic_matcher.py; a few seconds)...\n")

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

    labeled = pairs[["ems_record_id", "ehr_record_id", "pair_id", "is_match",
                      "ems_patient_identifier_mrn", "ehr_patient_identifier_mrn"]]
    full = preds.merge(
        labeled.drop(columns=["ems_patient_identifier_mrn", "ehr_patient_identifier_mrn"]),
        on=["ems_record_id", "ehr_record_id"], how="left",
    )
    full["is_match"] = full["is_match"].fillna(0).astype(int)

    print(f"Full cross-join: {len(full):,} comparisons "
          f"({int((full.is_match==1).sum())} true matches, "
          f"{int((full.is_match==0).sum())} non-matches)\n")

    # --- 1. Precision/recall across thresholds, on the realistic full population ---
    print("=== Precision / recall / F1 across thresholds (full 250,000-pair cross-join) ===")
    tbl = threshold_table(full)
    print(tbl.to_string(index=False, formatters={
        "threshold": "{:.3f}".format, "precision": "{:.4f}".format,
        "recall": "{:.4f}".format, "f1": "{:.4f}".format,
    }))
    print()

    # --- 2. Confusion matrix at a representative threshold ---
    print_confusion_matrix(full, 0.5, "full 250,000-pair cross-join")

    # --- 3. Same, on the 500 designed pairs, for continuity with Steps 3-5 ---
    designed = full[full["pair_id"].notna()].merge(labeled, on=["pair_id", "is_match"], how="left",
                                                     suffixes=("", "_dup"))
    designed = designed.loc[:, ~designed.columns.str.endswith("_dup")]
    print_confusion_matrix(designed, 0.5, "500 designed (250/250) pairs")

    # --- 4. Breakdown by whether the EMS side happened to capture an MRN ---
    # This is the pairs Tier 1 could have had a shot at resolving on its own
    # (whether or not the captured value was actually correct); the question
    # here is whether Tier 3 is equally reliable regardless.
    designed["ems_mrn_present"] = designed["ems_patient_identifier_mrn"].fillna("").astype(str).str.strip() != ""
    print("=== Breakdown by EMS-side MRN presence (500 designed pairs, threshold 0.5) ===")
    for present, group in designed.groupby("ems_mrn_present"):
        tp, fp, fn, tn = confusion_counts(group, 0.5)
        precision, recall, f1 = precision_recall_f1(tp, fp, fn)
        label = "MRN present on EMS side" if present else "MRN absent on EMS side"
        print(f"{label:28} n={len(group):>4}  true_matches={int((group.is_match==1).sum()):>3}  "
              f"precision={precision:.4f}  recall={recall:.4f}")
    print()

    # --- 5. Breakdown by last-name length ---
    designed["last_name_len_bucket"] = designed.apply(bucket_last_name_length, axis=1)
    print("=== Breakdown by last-name length -- longer of the two sides (500 designed pairs, threshold 0.5) ===")
    bucket_order = ["<=4", "5-7", "8+"]
    for bucket in bucket_order:
        group = designed[designed["last_name_len_bucket"] == bucket]
        if len(group) == 0:
            continue
        tp, fp, fn, tn = confusion_counts(group, 0.5)
        precision, recall, f1 = precision_recall_f1(tp, fp, fn)
        n_true = int((group.is_match == 1).sum())
        print(f"{bucket:>4} chars   n={len(group):>4}  true_matches={n_true:>3}  "
              f"precision={precision:.4f}  recall={recall:.4f}  (missed: {fn})")

    designed.to_csv("data/evaluation_designed_pairs_with_buckets.csv", index=False)


if __name__ == "__main__":
    main()
