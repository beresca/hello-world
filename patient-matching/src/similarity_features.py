"""
Tier 2 groundwork: similarity features for pairs without a shared strong ID.

WHY THIS EXISTS:
Tier 1 (deterministic_matcher.py) only resolves pairs that share a
confirmed identifier (MRN). We showed that only catches ~6% of true
matches on our synthetic dataset -- the other ~94% have no shared ID to
check at all, so Tier 1 correctly stays silent on them. This module
builds the *evidence* a later decision layer (a threshold rule, then
splink's probabilistic model) will need to judge those remaining pairs:
for each pair, how similar do the demographic fields actually look?

Each similarity score answers a narrower question than "is this a match,"
because different fields fail in different ways, and lumping them into
one number too early throws away information a later model could use:
  - Names get typos and nicknames -> string-edit-distance metrics
    (Jaro-Winkler) and phonetic codes (Soundex/NYSIIS) each catch
    different kinds of name corruption.
  - DOBs get digit transpositions -> a purpose-built comparison that
    understands "same month/day, wrong year" is more informative than
    treating the date as an opaque string.
  - Addresses are multi-word strings where EMS may capture a scene
    location instead of a home address -> a token-based string similarity
    handles word reordering and partial overlap better than a raw
    character-by-character comparison would.

We compute these features for every pair (not only the ones Tier 1
missed) because it costs nothing at this scale and lets us sanity-check
the scores against pairs where we already know the answer -- e.g.
confirming that pairs Tier 1 already resolved via MRN also tend to score
high on these fuzzy measures, and seeing how much lower non-matches score
by comparison. In a production pipeline you'd normally skip this
computation for pairs Tier 1 already resolved, to avoid wasted work; we
keep a `tier1_deterministic_match` column so that filtering is still easy
to do downstream.
"""

import numpy as np
import pandas as pd
import jellyfish
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from deterministic_matcher import deterministic_match


def _present(value) -> bool:
    """True if value is a non-blank string. Missing data should score as
    'unknown' (NaN), never silently as 0 -- a blank field isn't evidence
    the two records differ, it's an absence of evidence either way."""
    return isinstance(value, str) and value.strip() != ""


def name_similarity(a: str, b: str) -> float:
    """
    Jaro-Winkler similarity between two names, 0.0 (nothing alike) to 1.0
    (identical). Jaro-Winkler is a character-based edit-distance metric
    tuned for short strings like names: it gives extra credit for shared
    *prefixes*, which fits how people commonly misspell names (the first
    couple of letters are usually right, and mistakes cluster later in the
    word, e.g. "Timothhy" vs "Timothy", "Jaon" vs "Jason"). Plain
    Levenshtein distance doesn't weight prefixes specially and can be a
    poorer fit for exactly this kind of typo.

    We lowercase both strings first, since character-distance metrics are
    case-sensitive by default and a mismatched capitalization convention
    between two source systems shouldn't look like a spelling difference.
    """
    if not _present(a) or not _present(b):
        return np.nan
    return JaroWinkler.normalized_similarity(a.lower(), b.lower())


def address_similarity(a: str, b: str) -> float:
    """
    Similarity between two address strings, 0.0 to 1.0.

    Addresses are multi-token strings (house number, street, city, state,
    zip) rather than single words, and the same address can appear with
    word order or formatting differences ("123 Main St, Springfield, IL"
    vs "Springfield IL, 123 Main St Apt 4B"). rapidfuzz's
    `token_sort_ratio` tokenizes both strings on whitespace, sorts the
    tokens alphabetically, then compares -- so it isn't thrown off by
    reordering the way a straight character-by-character comparison
    (including Jaro-Winkler) would be. It's a better fit here than the
    name_similarity metric above precisely because addresses are made of
    swappable chunks, while names generally aren't.

    Note this only measures *string* similarity -- it has no idea that an
    EMS "scene address" and an EHR "home address" can legitimately be two
    different real-world places for the same person (e.g. collapsed at a
    friend's house). A low score here doesn't necessarily mean "different
    person," just "different address on file" -- that ambiguity is exactly
    why this is one input to a later decision, not a decision by itself.
    """
    if not _present(a) or not _present(b):
        return np.nan
    return fuzz.token_sort_ratio(a.lower(), b.lower()) / 100.0


def compare_dob(a: str, b: str) -> tuple:
    """
    Compare two dates of birth (ISO format 'YYYY-MM-DD' strings) and
    return (score, match_type).

    Unlike name/address similarity, a DOB isn't well served by a generic
    string-distance metric: "1999-02-18" vs "1999-08-12" would score
    fairly high as a plain string (they share most characters and the
    same length) despite being a totally different birth date, while
    "1999-02-18" vs "1999-02-19" (one digit off, a plausible typo) would
    score similarly. What actually matters is which *components*
    (year/month/day) agree, since that tells us something about what kind
    of error likely happened:
      - exact match: identical date -> strong evidence.
      - month & day match, year differs: consistent with a misremembered
        or transposed birth year (the year-off-by-1-2 noise we inject).
      - year & day match, month differs / year & month match, day
        differs: consistent with a transposed month/day (the classic
        MM/DD vs DD/MM mixup we also inject) or a single mistyped digit.
      - year matches only, or nothing matches: weak or no evidence.

    Returns NaN/'missing' if either date is blank -- a missing DOB is not
    evidence of a mismatch.
    """
    if not _present(a) or not _present(b):
        return (np.nan, "missing")
    try:
        y1, m1, d1 = (int(x) for x in a.split("-"))
        y2, m2, d2 = (int(x) for x in b.split("-"))
    except ValueError:
        return (np.nan, "unparseable")

    if (y1, m1, d1) == (y2, m2, d2):
        return (1.0, "exact")
    if m1 == d2 and d1 == m2 and y1 == y2:
        # classic day/month transposition, e.g. 2003-05-07 vs 2003-07-05
        return (0.85, "month_day_transposed")
    if (m1, d1) == (m2, d2):
        return (0.7, "month_day_match_year_differs")
    if y1 == y2 and d1 == d2:
        return (0.6, "year_day_match_month_differs")
    if y1 == y2 and m1 == m2:
        return (0.4, "year_month_match_day_differs")
    if y1 == y2:
        return (0.25, "year_match_only")
    return (0.0, "no_match")


def phonetic_codes(name: str) -> tuple:
    """
    Return (soundex_code, nysiis_code) for a last name, or ('', '') if
    blank.

    Both algorithms collapse a name down to a short code representing
    roughly how it *sounds*, so names that are spelled differently but
    pronounced the same (or nearly so) get the same code -- catching
    cases that a character-edit-distance metric like Jaro-Winkler can
    under-score because the actual letters differ a lot even though the
    name would be spoken the same way ("Smith" vs "Smyth", "Meyer" vs
    "Meier"). Soundex is the older, simpler algorithm (keeps the first
    letter, encodes the rest into 3 digits by consonant sound group).
    NYSIIS is newer and generally considered more accurate for American
    names, using a richer set of phonetic replacement rules -- but the two
    don't always agree, which is exactly why comparing both surfaces more
    signal than trusting either alone.
    """
    if not _present(name):
        return ("", "")
    return (jellyfish.soundex(name), jellyfish.nysiis(name))


def add_similarity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of `df` with Tier 1's deterministic-match flag and all
    Tier 2 similarity feature columns added.
    """
    df = df.copy()

    df["tier1_deterministic_match"] = df.apply(deterministic_match, axis=1).astype(int)

    df["first_name_similarity"] = df.apply(
        lambda r: name_similarity(r["ems_first_name"], r["ehr_first_name"]), axis=1
    )
    df["last_name_similarity"] = df.apply(
        lambda r: name_similarity(r["ems_last_name"], r["ehr_last_name"]), axis=1
    )

    dob_results = df.apply(lambda r: compare_dob(r["ems_date_of_birth"], r["ehr_date_of_birth"]), axis=1)
    df["dob_similarity"] = dob_results.apply(lambda t: t[0])
    df["dob_match_type"] = dob_results.apply(lambda t: t[1])

    df["address_similarity"] = df.apply(
        lambda r: address_similarity(r["ems_address"], r["ehr_address"]), axis=1
    )

    ems_codes = df["ems_last_name"].apply(phonetic_codes)
    ehr_codes = df["ehr_last_name"].apply(phonetic_codes)
    df["ems_last_name_soundex"] = ems_codes.apply(lambda t: t[0])
    df["ehr_last_name_soundex"] = ehr_codes.apply(lambda t: t[0])
    df["ems_last_name_nysiis"] = ems_codes.apply(lambda t: t[1])
    df["ehr_last_name_nysiis"] = ehr_codes.apply(lambda t: t[1])

    def _code_match(ems_code, ehr_code):
        if ems_code == "" or ehr_code == "":
            return np.nan
        return int(ems_code == ehr_code)

    df["last_name_soundex_match"] = df.apply(
        lambda r: _code_match(r["ems_last_name_soundex"], r["ehr_last_name_soundex"]), axis=1
    )
    df["last_name_nysiis_match"] = df.apply(
        lambda r: _code_match(r["ems_last_name_nysiis"], r["ehr_last_name_nysiis"]), axis=1
    )

    return df


def main():
    df = pd.read_csv("data/synthetic_ems_ehr_pairs.csv", dtype=str, keep_default_na=False)
    df["is_match"] = df["is_match"].astype(int)

    featured = add_similarity_features(df)
    out_path = "data/synthetic_ems_ehr_pairs_features.csv"
    featured.to_csv(out_path, index=False)
    print(f"Wrote {len(featured)} pairs with similarity features to {out_path}\n")

    score_cols = ["first_name_similarity", "last_name_similarity", "dob_similarity", "address_similarity"]

    print("=== Mean similarity scores by ground-truth label ===")
    print(featured.groupby("is_match")[score_cols].mean().round(3).rename(
        index={1: "true matches", 0: "non-matches"}
    ))
    print()

    print("=== Sample TRUE MATCH pairs (Tier 1 did NOT resolve these) ===")
    sample_cols = [
        "ems_first_name", "ehr_first_name", "first_name_similarity",
        "ems_last_name", "ehr_last_name", "last_name_similarity",
        "last_name_soundex_match", "last_name_nysiis_match",
        "ems_date_of_birth", "ehr_date_of_birth", "dob_similarity", "dob_match_type",
        "address_similarity",
    ]
    undecided_true_matches = featured[(featured.is_match == 1) & (featured.tier1_deterministic_match == 0)]
    print(undecided_true_matches[sample_cols].head(6).to_string())
    print()

    print("=== Sample NON-MATCH pairs, for contrast ===")
    print(featured[featured.is_match == 0][sample_cols].head(6).to_string())


if __name__ == "__main__":
    main()
