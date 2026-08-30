"""
Tier 3: probabilistic matching with Splink (Fellegi-Sunter model).

WHAT THIS TIER DOES, AND WHY IT'S DIFFERENT FROM TIERS 1 AND 2:
Tier 1 only fires when two records share a confirmed identifier -- no
judgment call, just a lookup. Tier 2 computed similarity scores per field,
but never combined them into one decision, and never learned which fields
were more trustworthy than others. Tier 3 is where those field-level
scores actually get combined into a single match probability, using a
formal statistical model (Fellegi-Sunter) rather than a hand-picked
threshold or an unweighted average. See NOTES.md ("Step 5") for a plain-
language walkthrough of how that model reasons about which fields matter
more -- this file is the implementation of that idea.

WHY WE RESHAPE THE DATA HERE:
Steps 2-4 worked with one CSV where each row is already a specific EMS/EHR
*pair*. Splink doesn't consume data in that shape -- it's built to take
one table per data source (here: all EMS records, all EHR records) and
generate its own candidate pairs (via blocking + its comparison engine),
because that's what lets it learn general-purpose weights and then apply
them to compare *any* two records, not just the ones we happened to
pre-pair. So step one here is un-pairing our CSV back into two separate
per-record tables, each with its own unique ID.

One consequence worth calling out: Splink's comparisons need to work on
individual per-record columns (e.g. l.first_name vs r.first_name), so they
can't take one of Step 4's already-*pairwise* similarity numbers (like
first_name_similarity) as a direct input -- that number was already
computed by comparing two records, and Splink needs to do that comparison
itself in order to learn from it. What we CAN and do carry over directly
from Step 4 is the last_name Soundex/NYSIIS codes, since a phonetic code
is a genuine per-record attribute (a property of one name, not a
comparison between two) -- Splink compares those codes for exact
agreement itself. For first/last name, DOB, and address we let Splink's
own comparison library recompute the same kinds of techniques we built by
hand in Step 4 (Jaro-Winkler for names, a component-aware date comparison,
Levenshtein for address) -- same ideas, implemented by Splink's engine so
it can also learn from them, which our precomputed numbers can't be used
for directly.

WHY WE SCORE THE FULL CROSS-JOIN, NOT JUST OUR 500 DESIGNED PAIRS:
This dataset has 500 EMS records and 500 EHR records. Only 250 of them
were built as "the same person" pairs; the other 250 EMS records and 250
EHR records were built from independent, unrelated identities (Step 2's
non-match design). If we score every EMS record against every EHR record
(500 x 500 = 250,000 comparisons), exactly 250 of those 250,000 are real
matches -- a much more realistic picture of the actual problem (an
ambulance record has exactly one correct hospital record among many
candidates, not a 50/50 shot) than our curated 500-pair file, which was
built 50/50 on purpose to make Step 4's per-field scores easy to inspect.
We still report results filtered back down to our original 500 designed
pairs, since that's the labeled comparison the user has been tracking
across every step -- but the full 250,000-pair picture is there too, and
is arguably the more honest one.

NULLS, NOT BLANK STRINGS:
Missing fields in our CSV are stored as "" (empty string), which is how
Steps 2-4 represented "not captured." Splink's model explicitly reasons
about missing data (each comparison gets a dedicated "null level" that's
excluded from scoring, exactly like Tier 1's blank-vs-blank guard and
Step 4's NaN-for-missing convention) -- but only if it's told those
values are actually missing. If we handed Splink literal empty strings,
it would treat "" == "" as a confirmed exact match between two records
that both simply lack the field, which would be the same false-positive
bug we deliberately avoided in Tier 1. So the reshape step below converts
"" to a real missing value (NaN) for every field Splink will compare.
"""

import numpy as np
import pandas as pd
import splink.comparison_library as cl
from splink import DuckDBAPI, Linker, SettingsCreator, block_on

FEATURES_CSV = "data/synthetic_ems_ehr_pairs_features.csv"


def _blank_to_nan(series: pd.Series) -> pd.Series:
    """Convert the "" convention used throughout this project to a real
    missing value, so Splink's null-handling applies instead of a blank
    ever being compared as if it were real data."""
    return series.replace("", np.nan)


def load_record_tables(path: str = FEATURES_CSV):
    """
    Read the Step 4 features CSV and split it back into two per-record
    tables -- one for EMS (ePCR), one for EHR (ADT) -- the shape Splink
    expects. Also returns the original pair-level dataframe (with
    is_match) for evaluation later.
    """
    pairs = pd.read_csv(path, dtype=str, keep_default_na=False)
    pairs["is_match"] = pairs["is_match"].astype(int)

    ems_df = pairs[[
        "ems_record_id", "ems_first_name", "ems_last_name", "ems_date_of_birth",
        "ems_address", "ems_last_name_soundex", "ems_last_name_nysiis",
    ]].rename(columns={
        "ems_record_id": "unique_id",
        "ems_first_name": "first_name",
        "ems_last_name": "last_name",
        "ems_date_of_birth": "date_of_birth",
        "ems_address": "address",
        "ems_last_name_soundex": "last_name_soundex",
        "ems_last_name_nysiis": "last_name_nysiis",
    })

    ehr_df = pairs[[
        "ehr_record_id", "ehr_first_name", "ehr_last_name", "ehr_date_of_birth",
        "ehr_address", "ehr_last_name_soundex", "ehr_last_name_nysiis",
    ]].rename(columns={
        "ehr_record_id": "unique_id",
        "ehr_first_name": "first_name",
        "ehr_last_name": "last_name",
        "ehr_date_of_birth": "date_of_birth",
        "ehr_address": "address",
        "ehr_last_name_soundex": "last_name_soundex",
        "ehr_last_name_nysiis": "last_name_nysiis",
    })

    for df in (ems_df, ehr_df):
        for col in ["first_name", "last_name", "date_of_birth", "address",
                    "last_name_soundex", "last_name_nysiis"]:
            df[col] = _blank_to_nan(df[col])

    return ems_df, ehr_df, pairs


def build_settings(prior_match_probability: float) -> SettingsCreator:
    """
    Define the Fellegi-Sunter model structure: which fields we compare, and
    how many levels of partial agreement each comparison distinguishes.

    `prior_match_probability` is the Fellegi-Sunter "prior" -- the
    probability that two records picked at random (before looking at any
    field) are a match. In a real deployment you would estimate this with
    `linker.training.estimate_probability_two_random_records_match()`
    using a trusted deterministic rule and a recall estimate for it,
    since you wouldn't know the true answer. Because we generated this
    dataset ourselves, we can just compute it directly instead: 250 true
    matches out of 500 x 500 = 250,000 possible EMS-to-EHR comparisons.
    """
    return SettingsCreator(
        link_type="link_only",
        unique_id_column_name="unique_id",
        probability_two_random_records_match=prior_match_probability,
        comparisons=[
            # Same technique as Step 4 (Jaro-Winkler), but computed by
            # Splink's own engine, with several agreement *levels* rather
            # than one blended score -- see NOTES.md for why that matters.
            cl.NameComparison("first_name"),
            cl.NameComparison("last_name"),
            # A purpose-built date comparison: exact match, a 1-character
            # edit (typo), then widening date-difference bands. Conceptually
            # the same goal as Step 4's compare_dob() (reward partial
            # agreement, don't just say "match"/"no match"), implemented
            # with Splink's own date-difference logic rather than our
            # component-based one.
            cl.DateOfBirthComparison(
                "date_of_birth", input_is_string=True, datetime_format="%Y-%m-%d"
            ),
            # Checked the actual Levenshtein-distance distribution on true
            # matches before picking this: it's sharply bimodal (exact
            # match, distance 0; or a totally different address, distance
            # 30-50) with essentially nothing in between. That makes sense
            # given how Step 2 generates this noise -- either the EMS
            # record uses the same home address, or a wholesale different
            # scene address, never a typo'd version of the same one. A
            # Levenshtein-distance-threshold comparison (as Step 4 might
            # suggest by analogy to name comparisons) would add fake
            # "partial similarity" levels the data doesn't actually
            # support and that EM can't train from real examples, so this
            # is a plain exact-match comparison instead. This is also a
            # step down from Step 4's token_sort_ratio, which at least
            # handles word-reordering -- Splink's built-in library has no
            # token-based comparison out of the box.
            cl.ExactMatch("address"),
            # This one IS a direct reuse of a Step 4 output: the Soundex
            # code is a per-record attribute, so we can hand Splink the
            # already-computed code and just ask "do these match exactly."
            cl.ExactMatch("last_name_soundex"),
        ],
        # No blocking at prediction time: with only 250,000 possible
        # comparisons total, scoring literally everything is cheap, and it
        # means we don't risk silently excluding a true match whose noisy
        # fields would have failed a stricter blocking rule. Passed as a
        # raw SQL string (not block_on(), which builds an l.col = r.col
        # equality condition out of its arguments -- "1=1" needs to be used
        # as-is, a literal always-true predicate, not treated as a column).
        blocking_rules_to_generate_predictions=["1=1"],
    )


MODEL_JSON_PATH = "data/splink_trained_model.json"


def train_and_predict(ems_df: pd.DataFrame, ehr_df: pd.DataFrame, settings: SettingsCreator):
    """
    Fit the Fellegi-Sunter model's m and u probabilities, then score every
    EMS-EHR comparison. Returns (predictions_df, linker) -- the linker is
    returned too so the caller can inspect/save the trained model.
    """
    db_api = DuckDBAPI()
    linker = Linker([ems_df, ehr_df], settings, db_api=db_api, input_table_aliases=["ems", "ehr"])

    # u probabilities: how often each comparison level occurs among pairs
    # that are (almost certainly) NOT matches. Estimated by randomly
    # sampling pairs from the full cross-join -- valid because true matches
    # are such a small fraction (250 of 250,000) that a random sample is
    # overwhelmingly non-matches by default, no labels required. Our whole
    # dataset only has 250,000 possible pairs total, so max_pairs here
    # covers the entire population rather than a subsample of it.
    linker.training.estimate_u_using_random_sampling(max_pairs=2.5e5, seed=42)

    # m probabilities: how often each comparison level occurs among pairs
    # that ARE matches -- the harder half, since we don't (in general) know
    # which pairs those are. Estimated via Expectation-Maximisation (EM):
    # start with a guess, see which pairs it thinks are probable matches,
    # re-estimate from those, repeat until it stabilizes. Each EM run needs
    # a blocking rule restricting it to a plausible-candidate subset (running
    # it over the full 250,000 mostly-obvious-non-matches would drown out the
    # signal); running it twice, with complementary blocking rules, means
    # every comparison gets estimated from a round where it wasn't the
    # thing being blocked on (a field can't teach the model anything in a
    # round where it was forced to match by the blocking rule itself).
    linker.training.estimate_parameters_using_expectation_maximisation(
        block_on("first_name", "last_name")
    )
    linker.training.estimate_parameters_using_expectation_maximisation(
        block_on("date_of_birth")
    )

    predictions = linker.inference.predict()
    return predictions.as_pandas_dataframe(), linker


def print_learned_weights(linker) -> None:
    """
    Print each comparison's learned levels with their m probability, u
    probability, and the resulting match weight (log2(m/u)) -- the actual
    numbers the Fellegi-Sunter model learned from this dataset, i.e. the
    concrete answer to "how does it decide some fields matter more."
    """
    model = linker.misc.save_model_to_json(MODEL_JSON_PATH, overwrite=True)
    print(f"Saved trained model configuration to {MODEL_JSON_PATH}\n")

    print("=== Learned match weights by comparison level ===")
    print(f"{'comparison':<20} {'level':<45} {'m':>8} {'u':>10} {'weight (log2 m/u)':>18}")
    for comparison in model["comparisons"]:
        name = comparison.get("output_column_name", comparison.get("comparison_description", "?"))
        for level in comparison["comparison_levels"]:
            if level.get("is_null_level"):
                continue
            label = level.get("label_for_charts", "?")
            m = level.get("m_probability")
            u = level.get("u_probability")
            if m is None or u is None:
                print(f"{name:<20} {label:<45} {'(untrained)':>8} {'':>10} {'':>18}")
                continue
            weight = np.log2(m / u)
            print(f"{name:<20} {label:<45} {m:>8.4f} {u:>10.6f} {weight:>18.3f}")
    print()


def main():
    ems_df, ehr_df, pairs = load_record_tables()

    n_true_matches = int((pairs["is_match"] == 1).sum())
    n_total_comparisons = len(ems_df) * len(ehr_df)
    prior = n_true_matches / n_total_comparisons

    print(f"EMS records: {len(ems_df)}, EHR records: {len(ehr_df)}")
    print(f"Total possible comparisons: {n_total_comparisons:,}")
    print(f"Known true matches: {n_true_matches} -> prior probability_two_random_records_match = {prior:.6f}\n")

    settings = build_settings(prior)
    preds, linker = train_and_predict(ems_df, ehr_df, settings)
    print_learned_weights(linker)

    # Splink assigns which input table is "l" and which is "r" internally
    # (observed: alphabetically by source_dataset name -- "ehr" sorts
    # before "ems" -- not by the order passed to Linker() or
    # input_table_aliases). Rather than assume an order, read it from the
    # source_dataset_l/source_dataset_r columns Splink actually produced.
    l_sources = preds["source_dataset_l"].unique()
    r_sources = preds["source_dataset_r"].unique()
    assert list(l_sources) == ["ehr"] and list(r_sources) == ["ems"], (
        f"Unexpected l/r source assignment: l={l_sources}, r={r_sources}. "
        "Update the rename below to match."
    )
    preds = preds.rename(columns={"unique_id_l": "ehr_record_id", "unique_id_r": "ems_record_id"})

    # Re-attach the ground truth: join Splink's predictions back to our
    # original 500 designed pairs on (ems_record_id, ehr_record_id). Any
    # predicted pair that ISN'T one of our 500 designed pairs is, by
    # construction (see module docstring), also a genuine non-match.
    labeled = pairs[["ems_record_id", "ehr_record_id", "pair_id", "is_match"]]
    scored = preds.merge(labeled, on=["ems_record_id", "ehr_record_id"], how="left")
    scored["is_match"] = scored["is_match"].fillna(0).astype(int)

    out_path = "data/splink_predictions_full_cross_join.csv"
    scored.to_csv(out_path, index=False)
    print(f"Wrote {len(scored):,} scored comparisons (full cross-join) to {out_path}\n")

    print("=== Match probability distribution across ALL {:,} comparisons ===".format(len(scored)))
    print(scored.groupby("is_match")["match_probability"].describe().rename(
        index={1: "true matches", 0: "non-matches"}
    ).round(4))
    print()

    # Now the comparison the user has been tracking since Step 3: our
    # original 500 designed, 50/50 pairs.
    designed = scored[scored["pair_id"].notna()].copy()
    designed_path = "data/splink_predictions_designed_pairs.csv"
    designed.to_csv(designed_path, index=False)
    print(f"Wrote {len(designed)} scored designed pairs to {designed_path}\n")

    print("=== Match probability distribution on the 500 designed pairs ===")
    print(designed.groupby("is_match")["match_probability"].describe().rename(
        index={1: "true matches", 0: "non-matches"}
    ).round(4))
    print()

    print("=== Sample of designed TRUE MATCH pairs and their match_probability ===")
    sample_cols = ["pair_id", "ems_record_id", "ehr_record_id", "match_probability", "match_weight"]
    print(designed[designed.is_match == 1].sort_values("match_probability")[sample_cols].head(8).to_string(index=False))
    print()
    print("=== Sample of designed NON-MATCH pairs and their match_probability ===")
    print(designed[designed.is_match == 0].sort_values("match_probability", ascending=False)[sample_cols].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
