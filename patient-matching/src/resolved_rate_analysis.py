"""
Step 9: how much of the match-rate gap is a review-queue problem vs. a
real ceiling.

WHAT THIS ANSWERS:
Step 7 picked one three-way policy (auto-match >= 0.995, review down to
0.015, else reject) and showed it produces zero missed true matches on
this evaluation. This script asks the next, sharper question: as we move
the auto-match cutoff around, how much of any resulting gap is just "more
things sitting in the review queue instead of being auto-matched" (a
solvable, tunable cost -- more review-queue capacity or more automation
risk tolerance) versus "a true match that no threshold setting could
route into the review queue without flooding it" (a real ceiling that
needs better features or a referential data source, not a different
threshold).

Everything here runs against the full, realistic 500x500 = 250,000-pair
cross-join, same as Steps 6-7, re-generated in memory rather than reading
a possibly-stale local CSV (the full cross-join file is deliberately not
committed to git -- see NOTES.md Step 5).
"""

import pandas as pd

from probabilistic_matcher import build_settings, load_record_tables, train_and_predict

CURRENT_REVIEW_FLOOR = 0.015
CURRENT_AUTO_MATCH = 0.995


def get_full_predictions() -> pd.DataFrame:
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


def three_way_report(df: pd.DataFrame, auto_match_threshold: float, review_floor: float, label: str) -> dict:
    total_pairs = len(df)
    total_true = int(df["is_match"].sum())

    band_auto = df[df["match_probability"] >= auto_match_threshold]
    band_review = df[(df["match_probability"] >= review_floor) & (df["match_probability"] < auto_match_threshold)]
    band_reject = df[df["match_probability"] < review_floor]

    auto_tp = int((band_auto["is_match"] == 1).sum())
    auto_fp = int((band_auto["is_match"] == 0).sum())
    review_tp = int((band_review["is_match"] == 1).sum())
    review_fp = int((band_review["is_match"] == 0).sum())
    reject_tp = int((band_reject["is_match"] == 1).sum())  # the ceiling loss

    auto_precision = auto_tp / len(band_auto) if len(band_auto) else float("nan")
    review_prevalence = review_tp / len(band_review) if len(band_review) else float("nan")

    # Overall resolved rate: fraction of ALL true matches that land somewhere
    # a human (or the auto-match rule) can correctly resolve them, i.e.
    # everything NOT in the reject band. Assumes a competent reviewer
    # correctly resolves every review-queue item -- an upper bound on real
    # human performance, not a guarantee.
    resolved_rate = (auto_tp + review_tp) / total_true if total_true else float("nan")
    ceiling_loss_rate = reject_tp / total_true if total_true else float("nan")

    report = {
        "label": label,
        "auto_match_threshold": auto_match_threshold,
        "review_floor": review_floor,
        "pct_auto": len(band_auto) / total_pairs * 100,
        "pct_review": len(band_review) / total_pairs * 100,
        "pct_reject": len(band_reject) / total_pairs * 100,
        "auto_n": len(band_auto), "auto_tp": auto_tp, "auto_fp": auto_fp,
        "auto_precision": auto_precision,
        "review_n": len(band_review), "review_tp": review_tp, "review_fp": review_fp,
        "review_true_match_prevalence": review_prevalence,
        "reject_n": len(band_reject), "reject_tp_ceiling_loss": reject_tp,
        "resolved_rate": resolved_rate,
        "ceiling_loss_rate": ceiling_loss_rate,
        "total_true_matches": total_true,
    }
    return report


def print_report(r: dict) -> None:
    print(f"--- {r['label']}  (auto-match >= {r['auto_match_threshold']}, review floor {r['review_floor']}) ---")
    print(f"  Band sizes (% of all {r['auto_n']+r['review_n']+r['reject_n']:,} pairs):")
    print(f"    Auto-match: {r['pct_auto']:.4f}%  (n={r['auto_n']:,}, {r['auto_tp']} true matches, "
          f"{r['auto_fp']} false positives, precision={r['auto_precision']:.4f})")
    print(f"    Review queue: {r['pct_review']:.4f}%  (n={r['review_n']:,}, {r['review_tp']} true matches, "
          f"{r['review_fp']} non-matches, true-match prevalence={r['review_true_match_prevalence']:.4f})")
    print(f"    Auto-reject: {r['pct_reject']:.4f}%  (n={r['reject_n']:,}, "
          f"{r['reject_tp_ceiling_loss']} true matches missed entirely -- THE CEILING LOSS)")
    print(f"  Overall resolved rate (auto-match + review, assuming review resolves correctly): "
          f"{r['resolved_rate']:.4f}  ({r['auto_tp']+r['review_tp']}/{r['total_true_matches']} true matches)")
    print(f"  Ceiling loss rate (unreachable by review at this floor): {r['ceiling_loss_rate']:.4f}")
    print()


def main():
    print("Retraining Tier 3 and scoring the full cross-join "
          "(reusing src/probabilistic_matcher.py; a few seconds)...\n")
    full = get_full_predictions()
    total_true = int(full["is_match"].sum())
    print(f"Full cross-join: {len(full):,} comparisons ({total_true} true matches)\n")

    reports = []

    # 1-4: current thresholds from Step 7
    reports.append(three_way_report(full, CURRENT_AUTO_MATCH, CURRENT_REVIEW_FLOOR, "Current (Step 7)"))

    # 5: alternate auto-match cutoffs, review floor held fixed -- isolates
    # the automation/precision tradeoff from the ceiling question, since
    # the floor (not the auto-match threshold) is what determines whether
    # any true match becomes unreachable.
    for t in [0.99, 0.9, 0.5]:
        reports.append(three_way_report(full, t, CURRENT_REVIEW_FLOOR, f"Auto-match threshold {t}"))

    # Bonus: raise the review floor itself, to show what a genuine ceiling
    # looks like (this is what actually produces reject-band losses, not
    # moving the auto-match threshold).
    for floor in [0.05, 0.1, 0.5]:
        reports.append(three_way_report(full, CURRENT_AUTO_MATCH, floor, f"Review floor raised to {floor}"))

    for r in reports:
        print_report(r)

    out = pd.DataFrame(reports)
    out.to_csv("data/resolved_rate_scenarios.csv", index=False)
    print("Wrote data/resolved_rate_scenarios.csv")


if __name__ == "__main__":
    main()
