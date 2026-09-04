"""
Step 11: does adding phone_number to the Tier 3 model actually help?

WHERE THIS CAME FROM:
Phone was never part of the Splink model in Steps 5-7 -- the original
comparisons list only ever covered first/last name, DOB, address, and
last-name phonetics. A review of a prototype UI mockup raised the
question directly: EMS documentation of phone number is assumed to be
too sparse to bother with, but when it IS captured, it's likely correct.
Checked that against our own data before building anything (see NOTES.md
Step 11): on this synthetic dataset, EMS phone is captured on 86.8% of
true matches (217/250) -- not sparse at all here -- and when present on
both sides, it matches exactly 100% of the time (217/217), with zero
coincidental matches among the 214 non-match pairs with both sides
present. That's a strong, currently-unused signal, so this script tests
whether adding it actually moves the needle, rather than assuming it
will.

WHY EXACT MATCH, NOT A FUZZY COMPARISON:
Checked the generator (src/generate_synthetic_data.py) before picking a
comparison type: phone is either copied byte-for-byte from the person's
record or blanked out (15% chance) -- there is no typo/corruption
function ever applied to it, the same structural pattern Step 5 already
found for address. So `cl.ExactMatch`, not a Jaro-Winkler-threshold
comparison, is the data-supported choice -- a fuzzy comparison would add
partial-agreement levels this dataset has no real examples to train them
from, the same reasoning that ruled out a fuzzy address comparison.

METHOD:
Trains two models fresh from the same data: the unmodified Step 5-7
baseline (no phone), and a variant with one added comparison
(cl.ExactMatch("phone_number")) and nothing else changed. Both are
scored against the full 250,000-pair cross-join and run through the same
three-way policy (auto-match >= 0.995, review >= 0.015) established in
Step 7, so the comparison is apples-to-apples at a fixed, already-
justified operating point -- not a re-tuned threshold that could make a
weaker model look artificially better.
"""

import pandas as pd
import splink.comparison_library as cl
from splink import DuckDBAPI, Linker, SettingsCreator

from probabilistic_matcher import load_record_tables, train_and_predict

AUTO_MATCH_THRESHOLD = 0.995
REVIEW_FLOOR = 0.015

HARD_CASES = {
    "P00234": "Liu -> iu (dropped first letter, Step 4.5)",
    "P00037": "Berry -> erry (dropped first letter, Step 5)",
}


def build_settings_with_phone(prior_match_probability: float) -> SettingsCreator:
    """Identical to probabilistic_matcher.build_settings(), with exactly
    one addition: cl.ExactMatch("phone_number") appended to the
    comparisons list. Nothing else about the model structure changes."""
    return SettingsCreator(
        link_type="link_only",
        unique_id_column_name="unique_id",
        probability_two_random_records_match=prior_match_probability,
        comparisons=[
            cl.NameComparison("first_name"),
            cl.NameComparison("last_name"),
            cl.DateOfBirthComparison(
                "date_of_birth", input_is_string=True, datetime_format="%Y-%m-%d"
            ),
            cl.ExactMatch("address"),
            cl.ExactMatch("last_name_soundex"),
            cl.ExactMatch("phone_number"),
        ],
        blocking_rules_to_generate_predictions=["1=1"],
    )


def train_baseline(ems_df, ehr_df, prior):
    from probabilistic_matcher import build_settings
    return train_and_predict(ems_df, ehr_df, build_settings(prior))


def train_with_phone(ems_df, ehr_df, prior):
    return train_and_predict(ems_df, ehr_df, build_settings_with_phone(prior))


def score_full_cross_join(preds: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    l_sources = preds["source_dataset_l"].unique()
    r_sources = preds["source_dataset_r"].unique()
    assert list(l_sources) == ["ehr"] and list(r_sources) == ["ems"], (
        f"Unexpected l/r source assignment: l={l_sources}, r={r_sources}."
    )
    preds = preds.rename(columns={"unique_id_l": "ehr_record_id", "unique_id_r": "ems_record_id"})
    labeled = pairs[["ems_record_id", "ehr_record_id", "pair_id", "is_match"]]
    full = preds.merge(labeled, on=["ems_record_id", "ehr_record_id"], how="left")
    full["is_match"] = full["is_match"].fillna(0).astype(int)
    return full


def three_way_report(df: pd.DataFrame, label: str) -> dict:
    total_pairs = len(df)
    total_true = int(df["is_match"].sum())

    band_auto = df[df["match_probability"] >= AUTO_MATCH_THRESHOLD]
    band_review = df[(df["match_probability"] >= REVIEW_FLOOR) & (df["match_probability"] < AUTO_MATCH_THRESHOLD)]
    band_reject = df[df["match_probability"] < REVIEW_FLOOR]

    auto_tp = int((band_auto["is_match"] == 1).sum())
    auto_fp = int((band_auto["is_match"] == 0).sum())
    review_tp = int((band_review["is_match"] == 1).sum())
    reject_tp = int((band_reject["is_match"] == 1).sum())

    return {
        "label": label,
        "pct_auto": len(band_auto) / total_pairs * 100,
        "pct_review": len(band_review) / total_pairs * 100,
        "pct_reject": len(band_reject) / total_pairs * 100,
        "auto_n": len(band_auto), "auto_tp": auto_tp, "auto_fp": auto_fp,
        "auto_precision": auto_tp / len(band_auto) if len(band_auto) else float("nan"),
        "review_n": len(band_review), "review_tp": review_tp,
        "reject_n": len(band_reject), "reject_tp_ceiling_loss": reject_tp,
        "resolved_rate": (auto_tp + review_tp) / total_true if total_true else float("nan"),
    }


def print_report(r: dict) -> None:
    print(f"--- {r['label']} ---")
    print(f"  Auto-match:   {r['pct_auto']:.4f}%  (n={r['auto_n']}, tp={r['auto_tp']}, fp={r['auto_fp']}, "
          f"precision={r['auto_precision']:.4f})")
    print(f"  Review queue: {r['pct_review']:.4f}%  (n={r['review_n']}, tp={r['review_tp']})")
    print(f"  Auto-reject:  {r['pct_reject']:.4f}%  (n={r['reject_n']}, "
          f"ceiling loss={r['reject_tp_ceiling_loss']})")
    print(f"  Resolved rate: {r['resolved_rate']:.4f}")
    print()


def print_learned_phone_weight(linker) -> None:
    model = linker.misc.save_model_to_json("data/phone_experiment_trained_model.json", overwrite=True)
    for comparison in model["comparisons"]:
        if comparison.get("output_column_name") != "phone_number":
            continue
        print("=== Learned weight for the new phone_number comparison ===")
        import numpy as np
        for level in comparison["comparison_levels"]:
            if level.get("is_null_level"):
                continue
            label = level.get("label_for_charts", "?")
            m, u = level.get("m_probability"), level.get("u_probability")
            if m is None or u is None:
                print(f"  {label:<25} (untrained)")
                continue
            weight = np.log2(m / u)
            print(f"  {label:<25} m={m:.4f}  u={u:.6f}  weight={weight:+.3f}")
        print()


def main():
    print("Loading data and training both models fresh (a few seconds each)...\n")
    ems_df, ehr_df, pairs = load_record_tables()
    n_true_matches = int((pairs["is_match"] == 1).sum())
    prior = n_true_matches / (len(ems_df) * len(ehr_df))

    print("=== Training BASELINE (no phone) -- unmodified Step 5-7 model ===\n")
    baseline_preds, _ = train_baseline(ems_df, ehr_df, prior)
    baseline_full = score_full_cross_join(baseline_preds, pairs)

    print("\n=== Training WITH PHONE (phone_number added as cl.ExactMatch) ===\n")
    phone_preds, phone_linker = train_with_phone(ems_df, ehr_df, prior)
    phone_full = score_full_cross_join(phone_preds, pairs)

    print_learned_phone_weight(phone_linker)

    baseline_report = three_way_report(baseline_full, "BASELINE (no phone)")
    phone_report = three_way_report(phone_full, "WITH PHONE")

    print("=" * 70)
    print("THREE-WAY BAND BREAKDOWN -- BASELINE vs WITH PHONE (full 250,000-pair cross-join)")
    print("=" * 70 + "\n")
    print_report(baseline_report)
    print_report(phone_report)

    print("=== Delta (with phone - baseline) ===")
    for key in ["pct_auto", "pct_review", "pct_reject", "auto_precision", "resolved_rate"]:
        delta = phone_report[key] - baseline_report[key]
        print(f"  {key}: {delta:+.4f}")
    print()

    # Did the two specific hard cases move at all?
    print("=== The two known hard cases, before vs after ===")
    for pair_id, description in HARD_CASES.items():
        b = baseline_full[baseline_full.pair_id == pair_id]
        p = phone_full[phone_full.pair_id == pair_id]
        if len(b) == 0 or len(p) == 0:
            print(f"  {pair_id} ({description}): not found in cross-join (unexpected)")
            continue
        b_prob = b.match_probability.iloc[0]
        p_prob = p.match_probability.iloc[0]
        print(f"  {pair_id} ({description}):")
        print(f"    baseline match_probability:   {b_prob:.6f}")
        print(f"    with-phone match_probability: {p_prob:.6f}")
        print(f"    delta: {p_prob - b_prob:+.6f}")
    print()

    pd.DataFrame([baseline_report, phone_report]).to_csv("data/phone_experiment_summary.csv", index=False)
    baseline_full.to_csv("data/phone_experiment_predictions_baseline.csv", index=False)
    phone_full.to_csv("data/phone_experiment_predictions_with_phone.csv", index=False)
    print("Wrote data/phone_experiment_summary.csv, "
          "data/phone_experiment_predictions_{baseline,with_phone}.csv, "
          "data/phone_experiment_trained_model.json")


if __name__ == "__main__":
    main()
