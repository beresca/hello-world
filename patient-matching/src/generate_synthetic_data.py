"""
Generate synthetic EMS (ePCR) <-> Hospital (EHR/ADT) record pairs for
prototyping a patient-matching algorithm.

WHY SYNTHETIC DATA:
Real ambulance and hospital records contain PHI (Protected Health
Information) and can't be used for a personal prototype. Faker lets us
invent realistic-*looking* people (names, addresses, DOBs) with no
connection to real humans, so we can safely build and test matching logic.

WHAT THIS SCRIPT PRODUCES:
A CSV where each row is one "candidate pair": one EMS record sitting next
to one EHR record, plus a ground-truth `is_match` column saying whether
they really are the same person (1) or not (0). About half the rows are
matches, half are not. This mirrors how a real matching system works: you
take two record sources that don't share an ID, generate candidate pairs,
score how similar each pair is, and decide which pairs represent the same
person. The `is_match` column is the "answer key" we'll use later to check
whether our scoring logic actually works -- it should NEVER be fed into
the matching algorithm itself as an input, only used afterward to grade it.

HOW THE NOISE WORKS (see NOTES.md for the full reasoning):
We first generate a single "ground truth" person (clean name, DOB, sex,
address, phone). For a TRUE MATCH pair, both the EMS and EHR records are
derived from that same person, but the EMS side has realistic messiness
applied on top (typos, nicknames, a scene address instead of a home
address, missing fields, etc.) because field documentation is rushed and
often incomplete. For a NON-MATCH pair, the EMS record comes from one
person and the EHR record comes from a totally different person, with only
generic missingness applied (no identity-blurring noise -- they don't need
help looking different).
"""

import argparse
import random
from datetime import timedelta

import pandas as pd
from faker import Faker

# A small hand-picked map of common nicknames, used to simulate EMS crews
# (or the patient themselves, if conscious) giving a "go-by" name instead
# of the legal name that ends up on the hospital chart. Real record
# linkage systems maintain much bigger nickname dictionaries than this --
# this is just enough to demonstrate the concept.
NICKNAMES = {
    "robert": ["bob", "rob", "bobby"],
    "william": ["bill", "will", "billy"],
    "richard": ["rick", "rich", "dick"],
    "james": ["jim", "jimmy"],
    "john": ["jon", "jack", "johnny"],
    "michael": ["mike", "mikey"],
    "katherine": ["kathy", "kate", "katie"],
    "elizabeth": ["liz", "beth", "eliza"],
    "margaret": ["maggie", "meg", "peggy"],
    "jennifer": ["jen", "jenny"],
    "christopher": ["chris"],
    "matthew": ["matt"],
    "anthony": ["tony"],
    "patricia": ["pat", "patty"],
    "deborah": ["deb", "debbie"],
    "thomas": ["tom", "tommy"],
    "charles": ["charlie", "chuck"],
    "joseph": ["joe", "joey"],
    "daniel": ["dan", "danny"],
    "samuel": ["sam", "sammy"],
}


def make_typo(name: str, rng: random.Random) -> str:
    """
    Return `name` with one small random typo introduced.

    We simulate the handful of mistake types a human typing quickly (or a
    voice-to-text field system) tends to make: swapping two neighboring
    letters, dropping a letter, or duplicating a letter. Picking uniformly
    among these gives a mix of "close but not exact" strings, which is
    exactly the kind of mismatch fuzzy-matching tools like rapidfuzz are
    designed to catch.
    """
    if len(name) < 3:
        return name
    i = rng.randint(0, len(name) - 2)
    typo_kind = rng.choice(["swap", "drop", "duplicate"])
    if typo_kind == "swap":
        chars = list(name)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
        return "".join(chars)
    if typo_kind == "drop":
        return name[:i] + name[i + 1:]
    # duplicate a letter
    return name[:i] + name[i] + name[i:]


def maybe_nickname_or_typo(first_name: str, rng: random.Random) -> str:
    """
    With some probability, swap a first name for a nickname if we know one,
    otherwise fall back to a typo. This is only ever called for the EMS
    side of a TRUE MATCH pair -- it's what makes the "same person"
    correctly harder to match on name alone.
    """
    lower = first_name.lower()
    if lower in NICKNAMES and rng.random() < 0.6:
        nickname = rng.choice(NICKNAMES[lower])
        return nickname.capitalize()
    return make_typo(first_name, rng)


def corrupt_dob(dob, rng: random.Random) -> str:
    """
    Return a corrupted version of a date of birth string (YYYY-MM-DD).

    Two realistic failure modes are modeled:
      - transposed day/month (common when a date is read aloud or copied
        between MM/DD and DD/MM conventions)
      - a wrong birth year off by 1-2 years (misheard or misremembered,
        especially for an unconscious or disoriented patient)
    """
    year, month, day = dob.year, dob.month, dob.day
    kind = rng.choice(["transpose_day_month", "wrong_year"])
    if kind == "transpose_day_month" and day <= 12:
        month, day = day, month
    else:
        year += rng.choice([-2, -1, 1, 2])
    try:
        return f"{year:04d}-{month:02d}-{day:02d}"
    except ValueError:
        return f"{dob.year:04d}-{dob.month:02d}-{dob.day:02d}"


def build_person(fake: Faker) -> dict:
    """
    Generate one clean 'ground truth' person. This represents the actual
    real-world individual -- both the EMS and EHR records for a true-match
    pair are derived from this, and it's also what we draw from twice
    (independently) to build a non-match pair.
    """
    sex = random.choice(["M", "F"])
    first_name = fake.first_name_male() if sex == "M" else fake.first_name_female()
    return {
        "first_name": first_name,
        "last_name": fake.last_name(),
        "dob": fake.date_of_birth(minimum_age=1, maximum_age=95),
        "sex": sex,
        "home_address": fake.address().replace("\n", ", "),
        "phone": fake.phone_number(),
    }


def build_ehr_record(person: dict, mrn: str, fake: Faker, rng: random.Random) -> dict:
    """
    Build the hospital-side EHR/ADT record. This is treated as the hospital's
    own system of record, so it's generally clean and always carries an MRN
    (the hospital assigns this itself; it doesn't depend on EMS at all).
    """
    return {
        "mrn": mrn,
        "first_name": person["first_name"],
        "last_name": person["last_name"],
        "dob": person["dob"].isoformat(),
        "sex": person["sex"],
        "address": person["home_address"],
        "phone": person["phone"],
        "admit_timestamp": None,  # filled in by caller once we know the EMS timestamp
    }


def build_ems_record(
    person: dict,
    fake: Faker,
    rng: random.Random,
    apply_identity_noise: bool,
    known_mrn: str | None = None,
) -> dict:
    """
    Build the EMS-side ePCR record derived from `person`.

    apply_identity_noise=True is used only for the EMS half of a TRUE
    MATCH pair: it's what injects nicknames/typos/DOB corruption/scene
    address, i.e. noise that makes a genuinely-the-same-person pair look
    less obviously identical. Non-match pairs don't need this, since two
    different Faker identities are already different.

    Generic missingness (blank phone/address/DOB) is applied regardless of
    apply_identity_noise, because incomplete field documentation happens on
    every call, not just for "hard" matches.
    """
    first_name = person["first_name"]
    dob_str = person["dob"].isoformat()
    address = person["home_address"]

    if apply_identity_noise:
        if rng.random() < 0.35:
            first_name = maybe_nickname_or_typo(first_name, rng)
        if rng.random() < 0.25:
            dob_str = corrupt_dob(person["dob"], rng)
        if rng.random() < 0.40:
            # EMS captured the scene location, not the patient's home
            address = fake.address().replace("\n", ", ")

    # Generic real-world incompleteness, independent of whether this is a
    # true match -- EMS documentation is rushed regardless of who the
    # patient turns out to be.
    if rng.random() < 0.05:
        dob_str = ""
    if rng.random() < 0.10:
        address = ""
    phone = person["phone"]
    if rng.random() < 0.15:
        phone = ""

    sex = person["sex"]
    if rng.random() < 0.02:
        sex = "F" if sex == "M" else "M"  # rare data-entry error

    # EMS crews essentially never know the hospital's internal MRN. Only
    # occasionally (e.g. patient had an insurance/ID card on scene) does an
    # MRN-like value get written down at all, and even then it might not
    # actually be correct (simulated by the caller passing a wrong value
    # for non-match pairs).
    ems_mrn = ""
    if rng.random() < 0.10:
        ems_mrn = known_mrn if known_mrn is not None else ""

    return {
        "first_name": first_name,
        "last_name": person["last_name"],
        "dob": dob_str,
        "sex": sex,
        "address": address,
        "phone": phone,
        "mrn": ems_mrn,
        "destination_hospital": fake.company() + " Hospital",
    }


def generate_pairs(n_pairs: int, seed: int) -> pd.DataFrame:
    """
    Generate `n_pairs` EMS<->EHR candidate pairs, split roughly 50/50
    between true matches and non-matches, and return them as a DataFrame.
    """
    fake = Faker()
    Faker.seed(seed)
    rng = random.Random(seed)
    random.seed(seed)  # random.choice() calls inside build_person use the global RNG

    n_matches = n_pairs // 2
    n_non_matches = n_pairs - n_matches
    rows = []

    # --- True match pairs -------------------------------------------------
    for i in range(n_matches):
        person = build_person(fake)
        mrn = f"MRN{100000 + i}"
        ems_incident_time = fake.date_time_between(start_date="-2y", end_date="now")
        ehr_admit_time = ems_incident_time + timedelta(minutes=rng.randint(10, 90))

        ems = build_ems_record(
            person, fake, rng, apply_identity_noise=True, known_mrn=mrn
        )
        ehr = build_ehr_record(person, mrn, fake, rng)

        rows.append({
            "pair_id": f"P{i:05d}",
            "is_match": 1,
            "ems_person_id": f"person_{i:05d}",
            "ehr_person_id": f"person_{i:05d}",
            "ems_record_id": f"EMS{i:06d}",
            "ems_first_name": ems["first_name"],
            "ems_last_name": ems["last_name"],
            "ems_dob": ems["dob"],
            "ems_sex": ems["sex"],
            "ems_address": ems["address"],
            "ems_phone": ems["phone"],
            "ems_mrn": ems["mrn"],
            "ems_destination_hospital": ems["destination_hospital"],
            "ems_incident_timestamp": ems_incident_time.isoformat(sep=" "),
            "ehr_record_id": f"EHR{i:06d}",
            "ehr_mrn": ehr["mrn"],
            "ehr_first_name": ehr["first_name"],
            "ehr_last_name": ehr["last_name"],
            "ehr_dob": ehr["dob"],
            "ehr_sex": ehr["sex"],
            "ehr_address": ehr["address"],
            "ehr_phone": ehr["phone"],
            "ehr_admit_timestamp": ehr_admit_time.isoformat(sep=" "),
        })

    # --- Non-match pairs ----------------------------------------------------
    for j in range(n_non_matches):
        i = n_matches + j
        ems_person = build_person(fake)
        ehr_person = build_person(fake)
        ehr_mrn = f"MRN{200000 + j}"
        # An MRN-shaped value the EMS crew might have jotted down for the
        # WRONG patient (e.g. stale insurance card) -- deliberately does
        # not match ehr_mrn, since these two records are different people.
        wrong_mrn = f"MRN{300000 + j}"

        ems_incident_time = fake.date_time_between(start_date="-2y", end_date="now")
        # Admit time is unrelated to the EMS incident time here, since these
        # two records don't actually belong to the same real-world event.
        ehr_admit_time = fake.date_time_between(start_date="-2y", end_date="now")

        ems = build_ems_record(
            ems_person, fake, rng, apply_identity_noise=False, known_mrn=wrong_mrn
        )
        ehr = build_ehr_record(ehr_person, ehr_mrn, fake, rng)

        rows.append({
            "pair_id": f"P{i:05d}",
            "is_match": 0,
            "ems_person_id": f"person_{i:05d}a",
            "ehr_person_id": f"person_{i:05d}b",
            "ems_record_id": f"EMS{i:06d}",
            "ems_first_name": ems["first_name"],
            "ems_last_name": ems["last_name"],
            "ems_dob": ems["dob"],
            "ems_sex": ems["sex"],
            "ems_address": ems["address"],
            "ems_phone": ems["phone"],
            "ems_mrn": ems["mrn"],
            "ems_destination_hospital": ems["destination_hospital"],
            "ems_incident_timestamp": ems_incident_time.isoformat(sep=" "),
            "ehr_record_id": f"EHR{i:06d}",
            "ehr_mrn": ehr["mrn"],
            "ehr_first_name": ehr["first_name"],
            "ehr_last_name": ehr["last_name"],
            "ehr_dob": ehr["dob"],
            "ehr_sex": ehr["sex"],
            "ehr_address": ehr["address"],
            "ehr_phone": ehr["phone"],
            "ehr_admit_timestamp": ehr_admit_time.isoformat(sep=" "),
        })

    df = pd.DataFrame(rows)
    # Shuffle row order so matches/non-matches aren't grouped block-by-block
    # -- a matching algorithm (or a human skimming the CSV) shouldn't be
    # able to cheat off row position.
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-pairs", type=int, default=500, help="Total number of pairs to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed, for reproducible output")
    parser.add_argument(
        "--out",
        type=str,
        default="data/synthetic_ems_ehr_pairs.csv",
        help="Output CSV path (relative to project root)",
    )
    args = parser.parse_args()

    df = generate_pairs(args.n_pairs, args.seed)
    df.to_csv(args.out, index=False)

    print(f"Wrote {len(df)} pairs to {args.out}")
    print(df["is_match"].value_counts().rename({1: "true matches", 0: "non-matches"}))


if __name__ == "__main__":
    main()
