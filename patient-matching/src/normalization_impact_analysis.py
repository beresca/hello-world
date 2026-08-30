"""
Step 10: measure the actual impact of field normalization, rather than
assuming it matters.

WHAT THIS SCRIPT DOES:
Builds two parallel versions of the similarity/matching pipeline that
differ in exactly one way -- how much text normalization is applied to
first_name, last_name, address, and phone_number before comparing them:

  RAW:        fields used exactly as they appear in the synthetic CSV --
              no lowercasing, no whitespace/punctuation stripping, no
              address standardization.
  NORMALIZED: lowercase, strip extra whitespace and punctuation/dashes,
              and standardize common address abbreviations (St/Street,
              Ave/Avenue, Apt/Unit, etc.) using `usaddress` to parse the
              address into components first, so only the street-type and
              occupancy-type words get standardized rather than blindly
              find-and-replacing substrings that might appear elsewhere
              (e.g. a place named "Stanley").

Both versions are run through the SAME Splink model structure (comparisons)
Steps 5-7 already defined -- this experiment changes what values feed
those comparisons, not the comparisons themselves -- and through the same
Step 6/7 evaluation harness (precision/recall, and the Step 7 three-way
auto-match/review/reject bands).

Phone is included as a Step 4-style standalone similarity feature (as
asked), but was never part of the Tier 3 Splink model in Steps 5-7, and
isn't added to it here either -- this script reports phone's raw-vs-
normalized delta as a feature-level finding only, separate from the
Splink precision/recall results.
"""

import re

import numpy as np
import pandas as pd
import usaddress
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from probabilistic_matcher import build_settings, train_and_predict

AUTO_MATCH_THRESHOLD = 0.995
REVIEW_FLOOR = 0.015

# Canonical form each variant standardizes to -- direction (abbreviate vs.
# expand) doesn't matter for matching, only that both sides land on the
# same token consistently.
STREET_TYPE_CANONICAL = {
    "street": "street", "st": "street", "str": "street",
    "avenue": "avenue", "ave": "avenue", "av": "avenue",
    "drive": "drive", "dr": "drive",
    "boulevard": "boulevard", "blvd": "boulevard",
    "lane": "lane", "ln": "lane",
    "court": "court", "ct": "court",
    "road": "road", "rd": "road",
    "circle": "circle", "cir": "circle",
    "place": "place", "pl": "place",
    "terrace": "terrace", "ter": "terrace",
    "parkway": "parkway", "pkwy": "parkway",
    "highway": "highway", "hwy": "highway",
    "trail": "trail", "trl": "trail",
    "square": "square", "sq": "square",
    "crossing": "crossing", "xing": "crossing",
    "way": "way", "loop": "loop", "run": "run",
    "spur": "spur", "spurs": "spur", "bend": "bend",
    "walk": "walk", "walks": "walk", "path": "path",
    "cove": "cove", "coves": "cove", "port": "port", "ports": "port",
    "harbor": "harbor", "harbors": "harbor", "haven": "haven",
    "extension": "extension", "extensions": "extension", "ext": "extension",
    "mission": "mission", "mount": "mount", "mountain": "mountain", "mountains": "mountain",
    "summit": "summit", "expressway": "expressway", "freeway": "freeway",
    "gateway": "gateway", "junction": "junction", "junctions": "junction",
    "point": "point", "points": "point", "view": "view", "views": "view",
    "vista": "vista", "vistas": "vista", "village": "village",
    "shores": "shore", "shore": "shore", "island": "island", "islands": "island",
    "grove": "grove", "groves": "grove", "meadow": "meadow", "meadows": "meadow",
    "field": "field", "fields": "field", "forge": "forge", "forges": "forge",
    "manor": "manor", "manors": "manor", "mills": "mill", "mill": "mill",
    "park": "park", "parks": "park", "pass": "pass", "pike": "pike",
    "plain": "plain", "plains": "plain", "prairie": "prairie", "ridge": "ridge", "ridges": "ridge",
    "ranch": "ranch", "rapid": "rapid", "rapids": "rapid",
    "station": "station", "stream": "stream", "streams": "stream",
    "throughway": "throughway", "trace": "trace", "track": "track",
    "underpass": "underpass", "union": "union", "valley": "valley", "valleys": "valley",
    "well": "well", "wells": "well", "canyon": "canyon", "cliff": "cliff", "cliffs": "cliff",
    "corner": "corner", "corners": "corner", "creek": "creek", "crest": "crest",
    "dale": "dale", "dam": "dam", "divide": "divide", "estate": "estate", "estates": "estate",
    "falls": "fall", "ferry": "ferry", "flat": "flat", "flats": "flat",
    "ford": "ford", "fords": "ford", "fort": "fort", "garden": "garden", "gardens": "garden",
    "glen": "glen", "glens": "glen", "green": "green", "greens": "green",
    "hill": "hill", "hills": "hill", "hollow": "hollow", "inlet": "inlet",
    "isle": "isle", "key": "key", "keys": "key", "knoll": "knoll", "knolls": "knoll",
    "lake": "lake", "lakes": "lake", "land": "land", "landing": "landing",
    "light": "light", "lights": "light", "lock": "lock", "locks": "lock",
    "lodge": "lodge", "motorway": "motorway", "neck": "neck", "orchard": "orchard",
    "oval": "oval", "overpass": "overpass",
}

OCCUPANCY_TYPE_CANONICAL = {
    "apartment": "unit", "apt": "unit", "unit": "unit",
    "suite": "unit", "ste": "unit",
    "building": "unit", "bldg": "unit",
    "floor": "unit", "fl": "unit",
    "room": "unit", "rm": "unit",
    "department": "unit", "dept": "unit",
    "box": "box",
}


def normalize_text(s) -> str:
    """Lowercase, strip punctuation/dashes to spaces, collapse whitespace."""
    if not isinstance(s, str) or s.strip() == "":
        return s if isinstance(s, str) else ""
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_phone(s) -> str:
    """Digits only -- '(555) 123-4567' and '555.123.4567' both become
    '5551234567', which is the only sensible normalization for a phone
    number (formatting punctuation carries no information at all)."""
    if not isinstance(s, str) or s.strip() == "":
        return s if isinstance(s, str) else ""
    return re.sub(r"\D", "", s)


def normalize_address(s: str) -> str:
    """
    Parse the address into components with usaddress and standardize the
    street-type and occupancy-type words specifically (St->street,
    Ave->avenue, Apt->unit, Suite->unit, ...), rather than a blind
    find-and-replace across the whole string, which could wrongly rewrite
    a place name (e.g. "Stanley") that happens to contain an abbreviation.
    Falls back to plain text normalization if usaddress can't parse it
    (rare for Faker-style addresses, but real EMS-entered addresses could
    be messy enough to fail parsing) or tags it as an ambiguous/non-street
    address (e.g. our synthetic APO/FPO military addresses).
    """
    if not isinstance(s, str) or s.strip() == "":
        return s if isinstance(s, str) else ""
    try:
        tagged, addr_type = usaddress.tag(s)
    except usaddress.RepeatedLabelError:
        return normalize_text(s)
    if addr_type != "Street Address":
        return normalize_text(s)

    parts = []
    for label, value in tagged.items():
        v = value.lower().strip(".")
        if label == "StreetNamePostType":
            v = STREET_TYPE_CANONICAL.get(v, v)
        elif label == "OccupancyType":
            v = OCCUPANCY_TYPE_CANONICAL.get(v, v)
        parts.append(v)
    return normalize_text(" ".join(parts))


def name_similarity(a: str, b: str) -> float:
    if not a or not b:
        return np.nan
    return JaroWinkler.normalized_similarity(a, b)


def address_similarity(a: str, b: str) -> float:
    if not a or not b:
        return np.nan
    return fuzz.token_sort_ratio(a, b) / 100.0


def phone_similarity(a: str, b: str) -> float:
    if not a or not b:
        return np.nan
    return JaroWinkler.normalized_similarity(a, b)


def build_feature_comparison(pairs: pd.DataFrame) -> pd.DataFrame:
    """Step-4-style standalone similarity features, computed twice per
    field pair -- once raw, once normalized -- so the delta is visible
    row by row, not just in aggregate."""
    df = pairs.copy()

    for field, norm_fn, sim_fn in [
        ("first_name", normalize_text, name_similarity),
        ("last_name", normalize_text, name_similarity),
        ("address", normalize_address, address_similarity),
        ("phone_number", normalize_phone, phone_similarity),
    ]:
        ems_col, ehr_col = f"ems_{field}", f"ehr_{field}"
        df[f"{field}_similarity_raw"] = df.apply(
            lambda r: sim_fn(r[ems_col], r[ehr_col]), axis=1
        )
        df[f"{field}_similarity_normalized"] = df.apply(
            lambda r: sim_fn(norm_fn(r[ems_col]), norm_fn(r[ehr_col])), axis=1
        )
    return df


def build_record_tables(pairs: pd.DataFrame, use_normalization: bool):
    """Same shape as probabilistic_matcher.load_record_tables, but with
    first_name/last_name/address optionally passed through the
    normalization functions before Splink ever sees them. date_of_birth
    and the phonetic codes are left exactly as Steps 5-7 built them --
    this experiment is scoped to name/address per the existing Splink
    model's actual comparisons."""
    text_fn = normalize_text if use_normalization else (lambda x: x)
    addr_fn = normalize_address if use_normalization else (lambda x: x)

    def build_side(prefix):
        cols = {
            f"{prefix}_record_id": "unique_id",
            f"{prefix}_first_name": "first_name",
            f"{prefix}_last_name": "last_name",
            f"{prefix}_date_of_birth": "date_of_birth",
            f"{prefix}_address": "address",
            f"{prefix}_last_name_soundex": "last_name_soundex",
            f"{prefix}_last_name_nysiis": "last_name_nysiis",
        }
        side = pairs[list(cols.keys())].rename(columns=cols).copy()
        side["first_name"] = side["first_name"].apply(text_fn).replace("", np.nan)
        side["last_name"] = side["last_name"].apply(text_fn).replace("", np.nan)
        side["address"] = side["address"].apply(addr_fn).replace("", np.nan)
        side["date_of_birth"] = side["date_of_birth"].replace("", np.nan)
        side["last_name_soundex"] = side["last_name_soundex"].replace("", np.nan)
        side["last_name_nysiis"] = side["last_name_nysiis"].replace("", np.nan)
        return side

    return build_side("ems"), build_side("ehr")


def run_splink_variant(pairs: pd.DataFrame, use_normalization: bool) -> pd.DataFrame:
    ems_df, ehr_df = build_record_tables(pairs, use_normalization)
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

    predicted_match_05 = df["match_probability"] >= 0.5
    actual_match = df["is_match"] == 1
    tp05 = int((predicted_match_05 & actual_match).sum())
    fp05 = int((predicted_match_05 & ~actual_match).sum())
    fn05 = int((~predicted_match_05 & actual_match).sum())
    precision_05 = tp05 / (tp05 + fp05) if (tp05 + fp05) else float("nan")
    recall_05 = tp05 / (tp05 + fn05) if (tp05 + fn05) else float("nan")

    return {
        "label": label,
        "n_pairs": total_pairs,
        "total_true_matches": total_true,
        "pct_auto": len(band_auto) / total_pairs * 100,
        "pct_review": len(band_review) / total_pairs * 100,
        "pct_reject": len(band_reject) / total_pairs * 100,
        "auto_n": len(band_auto), "auto_tp": auto_tp, "auto_fp": auto_fp,
        "auto_precision": auto_tp / len(band_auto) if len(band_auto) else float("nan"),
        "review_n": len(band_review), "review_tp": review_tp,
        "reject_n": len(band_reject), "reject_tp_ceiling_loss": reject_tp,
        "resolved_rate": (auto_tp + review_tp) / total_true if total_true else float("nan"),
        "precision_at_0.5": precision_05,
        "recall_at_0.5": recall_05,
    }


def print_three_way(r: dict) -> None:
    print(f"--- {r['label']} ---")
    print(f"  Auto-match:   {r['pct_auto']:.4f}%  (n={r['auto_n']}, tp={r['auto_tp']}, fp={r['auto_fp']}, "
          f"precision={r['auto_precision']:.4f})")
    print(f"  Review queue: {r['pct_review']:.4f}%  (n={r['review_n']}, tp={r['review_tp']})")
    print(f"  Auto-reject:  {r['pct_reject']:.4f}%  (n={r['reject_n']}, "
          f"ceiling loss={r['reject_tp_ceiling_loss']})")
    print(f"  Resolved rate: {r['resolved_rate']:.4f}")
    print(f"  Precision/recall @ 0.5: {r['precision_at_0.5']:.4f} / {r['recall_at_0.5']:.4f}")
    print()


def main():
    pairs = pd.read_csv("data/synthetic_ems_ehr_pairs_features.csv", dtype=str, keep_default_na=False)
    pairs["is_match"] = pairs["is_match"].astype(int)

    # --- Feature-level comparison (Step 4 style): does normalization move
    # the standalone similarity scores at all? ---
    featured = build_feature_comparison(pairs)
    featured.to_csv("data/normalization_feature_comparison.csv", index=False)

    print("=== Feature-level raw vs. normalized similarity (mean by ground truth) ===\n")
    for field in ["first_name", "last_name", "address", "phone_number"]:
        raw_col, norm_col = f"{field}_similarity_raw", f"{field}_similarity_normalized"
        means = featured.groupby("is_match")[[raw_col, norm_col]].mean().round(6)
        means.index = means.index.map({1: "true matches", 0: "non-matches"})
        print(f"-- {field} --")
        print(means)
        # NaN != NaN in pandas, so a plain != would count "both sides
        # missing on both variants" as a change -- compare with NaN
        # treated as equal to NaN instead.
        n_rows_changed = (~(featured[raw_col].round(6) == featured[norm_col].round(6)) &
                           ~(featured[raw_col].isna() & featured[norm_col].isna())).sum()
        n_comparable = featured[raw_col].notna().sum()
        print(f"   rows where the score actually changed: {n_rows_changed} / {n_comparable} comparable pairs\n")

    # --- Splink model comparison (Step 5-7 model, raw vs normalized inputs) ---
    print("\n=== Training Splink RAW variant (unmodified fields) ===\n")
    raw_full = run_splink_variant(pairs, use_normalization=False)

    print("\n=== Training Splink NORMALIZED variant (lowercased, punctuation-stripped, address-standardized) ===\n")
    norm_full = run_splink_variant(pairs, use_normalization=True)

    raw_full.to_csv("data/normalization_splink_predictions_raw.csv", index=False)
    norm_full.to_csv("data/normalization_splink_predictions_normalized.csv", index=False)

    raw_report = three_way_report(raw_full, "RAW (no normalization)")
    norm_report = three_way_report(norm_full, "NORMALIZED")

    print("\n" + "=" * 70)
    print("THREE-WAY BAND BREAKDOWN, RAW vs. NORMALIZED (full 250,000-pair cross-join)")
    print("=" * 70 + "\n")
    print_three_way(raw_report)
    print_three_way(norm_report)

    print("=== Delta (normalized - raw) ===")
    for key in ["pct_auto", "pct_review", "pct_reject", "auto_precision", "resolved_rate",
                "precision_at_0.5", "recall_at_0.5"]:
        delta = norm_report[key] - raw_report[key]
        print(f"  {key}: {delta:+.4f}")

    pd.DataFrame([raw_report, norm_report]).to_csv("data/normalization_impact_summary.csv", index=False)
    print("\nWrote data/normalization_feature_comparison.csv, "
          "data/normalization_splink_predictions_raw.csv, "
          "data/normalization_splink_predictions_normalized.csv, "
          "data/normalization_impact_summary.csv")


if __name__ == "__main__":
    main()
