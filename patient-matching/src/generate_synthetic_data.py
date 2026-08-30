"""
Generate synthetic EMS (ePCR) <-> Hospital (EHR/ADT) record pairs for
prototyping a patient-matching algorithm.

GENERATOR VERSION: v2 (2026-08-30) -- true-match pairs now get typo noise
injected into last_name as well as first_name (previously only first_name
was noised, which made last_name unrealistically clean/always-exact). See
NOTES.md "Step 4.5" for why this changed and what it affects. Running this
script overwrites data/synthetic_ems_ehr_pairs.csv in place; NOTES.md
records the date of each regeneration so v1-era results can still be
understood if anything downstream needs to be compared against them.

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

FIELD NAMES FOLLOW USCDI:
The demographic fields on both sides (first/middle/last/previous name,
suffix, date of birth, birth sex, race, ethnicity, address, phone number,
email address) are named to match the USCDI (United States Core Data for
Interoperability) "Patient Demographics/Information" data class -- the
standardized field set real EMS and hospital systems are expected to
report. See NOTES.md for the full rationale.

The patient identifier (MRN) is deliberately named and treated as its own
concept, `patient_identifier_mrn`, rather than folded in among the
demographic fields above -- USCDI does not classify a medical record
number as a demographic attribute; it's a local/administrative identifier
issued by the hospital's system. Keeping it separate matters later: a
matching tier that checks "do these share a confirmed ID" is a completely
different kind of logic from one that checks "how similar do these
demographic fields look," and the field naming should make that
distinction obvious rather than blur it.

HOW THE NOISE WORKS (see NOTES.md for the full reasoning):
We first generate a single "ground truth" person (clean name, DOB, sex,
address, phone, race, ethnicity, email, etc). For a TRUE MATCH pair, both
the EMS and EHR records are derived from that same person, but the EMS
side has realistic messiness applied on top (typos/nicknames on
first_name, typos on last_name, a scene address instead of a home
address, missing fields, etc.) because field documentation is rushed and
often incomplete. Last_name gets the same typo technique as first_name
(swap/drop/duplicate a letter) but never the nickname substitution, since
nicknames are a first-name concept. For a NON-MATCH pair, the EMS record
comes from one person and the EHR record comes from a totally different
person, with only generic missingness applied (no identity-blurring noise
-- they don't need help looking different).

Some of the newer USCDI fields (previous name, email address) are
structurally absent on the EMS side regardless of match status: most
ePCR/NEMSIS-based ambulance documentation systems simply have no field to
capture a patient's email address or their prior legal name, so we always
leave those blank on the EMS side rather than modeling them as
"sometimes missing" -- that's a fact about what the form can capture, not
random noise.
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

# USCDI's Race and Ethnicity elements point to the OMB (Office of
# Management and Budget) standard categories used across US federal health
# data. These are illustrative population weights, not derived from any
# real dataset -- good enough for a synthetic prototype, not for anything
# claiming demographic accuracy.
RACE_CATEGORIES = [
    "White",
    "Black or African American",
    "Asian",
    "American Indian or Alaska Native",
    "Native Hawaiian or Other Pacific Islander",
    "Two or More Races",
]
RACE_WEIGHTS = [0.60, 0.13, 0.06, 0.01, 0.005, 0.10]

ETHNICITY_CATEGORIES = ["Not Hispanic or Latino", "Hispanic or Latino"]
ETHNICITY_WEIGHTS = [0.82, 0.18]

SUFFIXES = ["Jr.", "Sr.", "II", "III", "IV"]


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


def record_categorical(true_value: str, rng: random.Random, blank_prob: float, unknown_prob: float) -> str:
    """
    Simulate how a categorical field (race, ethnicity) actually ends up
    recorded, as opposed to the person's real/true value: sometimes it's
    left blank, sometimes the intake system records "Unknown" (the patient
    declined, or nobody asked), and most of the time it's captured
    correctly. `blank_prob` and `unknown_prob` differ between EMS and EHR
    because a structured hospital registration desk captures this far more
    reliably than a rushed field crew.
    """
    r = rng.random()
    if r < blank_prob:
        return ""
    if r < blank_prob + unknown_prob:
        return "Unknown"
    return true_value


def build_person(fake: Faker, rng: random.Random) -> dict:
    """
    Generate one clean 'ground truth' person. This represents the actual
    real-world individual -- both the EMS and EHR records for a true-match
    pair are derived from this, and it's also what we draw from twice
    (independently) to build a non-match pair.

    Fields here cover the full USCDI Patient Demographics/Information set
    we're targeting: first/middle/last/previous name, suffix, date of
    birth, birth sex, race, ethnicity, address, phone number, and email
    address. Not every person actually *has* a middle name, suffix, or
    previous name on file -- those are represented as empty strings when
    absent, same as a real record would.
    """
    birth_sex = rng.choice(["Male", "Female"])
    name_pool = fake.first_name_male() if birth_sex == "Male" else fake.first_name_female()
    first_name = name_pool

    # Most people do have a middle name recorded somewhere, but not all.
    middle_name = ""
    if rng.random() < 0.75:
        middle_name = fake.first_name_male() if birth_sex == "Male" else fake.first_name_female()

    # A legal name suffix (Jr., III, etc.) is uncommon.
    suffix = rng.choice(SUFFIXES) if rng.random() < 0.08 else ""

    # A previous/maiden name only exists for people who've legally changed
    # their last name (e.g. marriage) -- we assume the first name is
    # unchanged, which is the common real-world case.
    last_name = fake.last_name()
    previous_name = ""
    if rng.random() < 0.12:
        previous_name = f"{first_name} {fake.last_name()}"

    return {
        "first_name": first_name,
        "middle_name": middle_name,
        "last_name": last_name,
        "suffix": suffix,
        "previous_name": previous_name,
        "dob": fake.date_of_birth(minimum_age=1, maximum_age=95),
        "birth_sex": birth_sex,
        "race": rng.choices(RACE_CATEGORIES, weights=RACE_WEIGHTS, k=1)[0],
        "ethnicity": rng.choices(ETHNICITY_CATEGORIES, weights=ETHNICITY_WEIGHTS, k=1)[0],
        "home_address": fake.address().replace("\n", ", "),
        "phone": fake.phone_number(),
        "email": fake.email(),
    }


def build_ehr_record(person: dict, mrn: str, fake: Faker, rng: random.Random) -> dict:
    """
    Build the hospital-side EHR/ADT record. This is treated as the
    hospital's own system of record: a structured registration desk
    captures the full USCDI demographic set fairly reliably, and the
    hospital always assigns its own MRN (it doesn't depend on EMS at all).

    Small, realistic gaps remain even here -- a patient can decline to
    answer race/ethnicity questions, or not provide an email at
    registration -- but at much lower rates than on the EMS side.
    """
    email = person["email"]
    if rng.random() < 0.10:
        email = ""

    return {
        "patient_identifier_mrn": mrn,
        "first_name": person["first_name"],
        "middle_name": person["middle_name"],
        "last_name": person["last_name"],
        "suffix": person["suffix"],
        "previous_name": person["previous_name"],
        "dob": person["dob"].isoformat(),
        "birth_sex": person["birth_sex"],
        "race": record_categorical(person["race"], rng, blank_prob=0.02, unknown_prob=0.08),
        "ethnicity": record_categorical(person["ethnicity"], rng, blank_prob=0.02, unknown_prob=0.08),
        "address": person["home_address"],
        "phone": person["phone"],
        "email": email,
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
    MATCH pair: it's what injects nicknames/typos on first_name, typos on
    last_name, DOB corruption, and scene address, i.e. noise that makes a
    genuinely-the-same-person pair look less obviously identical.
    Non-match pairs don't need this, since two different Faker identities
    are already different.

    Generic missingness (blank phone/address/DOB, abbreviated middle name,
    dropped suffix, lower-quality race/ethnicity capture) is applied
    regardless of apply_identity_noise, because incomplete field
    documentation happens on every call, not just for "hard" matches.

    previous_name and email are always blank here -- see the module
    docstring: most ePCR/NEMSIS-based systems have no field to capture
    either one, so this isn't "sometimes missing," it's structurally
    absent from the form.
    """
    first_name = person["first_name"]
    last_name = person["last_name"]
    dob_str = person["dob"].isoformat()
    address = person["home_address"]

    if apply_identity_noise:
        if rng.random() < 0.35:
            first_name = maybe_nickname_or_typo(first_name, rng)
        if rng.random() < 0.35:
            # Same typo technique as first_name, minus the nickname
            # substitution (nicknames are a first-name concept -- nobody
            # has a "nickname" for a surname). Real-world last-name noise
            # like this comes from mishearing over radio, transposed
            # letters, or inconsistent hyphenation; make_typo's
            # swap/drop/duplicate operations are a reasonable stand-in for
            # that family of errors without inventing new logic.
            last_name = make_typo(last_name, rng)
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

    birth_sex = person["birth_sex"]
    if rng.random() < 0.02:
        birth_sex = "Female" if birth_sex == "Male" else "Male"  # rare data-entry error

    # Middle name: EMS often only captures an initial, or skips it
    # entirely, even when the person does have one on file elsewhere.
    middle_name = ""
    if person["middle_name"]:
        r = rng.random()
        if r < 0.45:
            middle_name = person["middle_name"]
        elif r < 0.70:
            middle_name = person["middle_name"][0] + "."
        # else: omitted entirely (remaining ~30%)

    # Suffix: frequently dropped in the rush of field documentation, even
    # when the person's legal name includes one.
    suffix = ""
    if person["suffix"] and rng.random() < 0.40:
        suffix = person["suffix"]

    # Race/ethnicity ARE standard NEMSIS/ePCR fields, but field crews
    # record "Unknown" or leave them blank far more often than a hospital
    # registration desk does.
    race = record_categorical(person["race"], rng, blank_prob=0.10, unknown_prob=0.25)
    ethnicity = record_categorical(person["ethnicity"], rng, blank_prob=0.10, unknown_prob=0.25)

    # EMS crews essentially never know the hospital's internal MRN. Only
    # occasionally (e.g. patient had an insurance/ID card on scene) does an
    # MRN-like value get written down at all, and even then it might not
    # actually be correct (simulated by the caller passing a wrong value
    # for non-match pairs).
    ems_mrn = ""
    if rng.random() < 0.10:
        ems_mrn = known_mrn if known_mrn is not None else ""

    return {
        "patient_identifier_mrn": ems_mrn,
        "first_name": first_name,
        "middle_name": middle_name,
        "last_name": last_name,
        "suffix": suffix,
        "previous_name": "",  # structurally absent on EMS forms -- see docstring
        "dob": dob_str,
        "birth_sex": birth_sex,
        "race": race,
        "ethnicity": ethnicity,
        "address": address,
        "phone": phone,
        "email": "",  # structurally absent on EMS forms -- see docstring
        "destination_hospital": fake.company() + " Hospital",
    }


def _pair_row(pair_id: str, is_match: int, ems_person_id: str, ehr_person_id: str,
              ems_record_id: str, ehr_record_id: str, ems: dict, ehr: dict,
              ems_incident_time, ehr_admit_time) -> dict:
    """Assemble one output row from an EMS record dict and an EHR record dict."""
    return {
        "pair_id": pair_id,
        "is_match": is_match,
        "ems_person_id": ems_person_id,
        "ehr_person_id": ehr_person_id,
        "ems_record_id": ems_record_id,
        "ems_patient_identifier_mrn": ems["patient_identifier_mrn"],
        "ems_first_name": ems["first_name"],
        "ems_middle_name": ems["middle_name"],
        "ems_last_name": ems["last_name"],
        "ems_suffix": ems["suffix"],
        "ems_previous_name": ems["previous_name"],
        "ems_date_of_birth": ems["dob"],
        "ems_birth_sex": ems["birth_sex"],
        "ems_race": ems["race"],
        "ems_ethnicity": ems["ethnicity"],
        "ems_address": ems["address"],
        "ems_phone_number": ems["phone"],
        "ems_email_address": ems["email"],
        "ems_destination_hospital": ems["destination_hospital"],
        "ems_incident_timestamp": ems_incident_time.isoformat(sep=" "),
        "ehr_record_id": ehr_record_id,
        "ehr_patient_identifier_mrn": ehr["patient_identifier_mrn"],
        "ehr_first_name": ehr["first_name"],
        "ehr_middle_name": ehr["middle_name"],
        "ehr_last_name": ehr["last_name"],
        "ehr_suffix": ehr["suffix"],
        "ehr_previous_name": ehr["previous_name"],
        "ehr_date_of_birth": ehr["dob"],
        "ehr_birth_sex": ehr["birth_sex"],
        "ehr_race": ehr["race"],
        "ehr_ethnicity": ehr["ethnicity"],
        "ehr_address": ehr["address"],
        "ehr_phone_number": ehr["phone"],
        "ehr_email_address": ehr["email"],
        "ehr_admit_timestamp": ehr_admit_time.isoformat(sep=" "),
    }


def generate_pairs(n_pairs: int, seed: int) -> pd.DataFrame:
    """
    Generate `n_pairs` EMS<->EHR candidate pairs, split roughly 50/50
    between true matches and non-matches, and return them as a DataFrame.
    """
    fake = Faker()
    Faker.seed(seed)
    rng = random.Random(seed)

    n_matches = n_pairs // 2
    n_non_matches = n_pairs - n_matches
    rows = []

    # --- True match pairs -------------------------------------------------
    for i in range(n_matches):
        person = build_person(fake, rng)
        mrn = f"MRN{100000 + i}"
        ems_incident_time = fake.date_time_between(start_date="-2y", end_date="now")
        ehr_admit_time = ems_incident_time + timedelta(minutes=rng.randint(10, 90))

        ems = build_ems_record(person, fake, rng, apply_identity_noise=True, known_mrn=mrn)
        ehr = build_ehr_record(person, mrn, fake, rng)

        rows.append(_pair_row(
            pair_id=f"P{i:05d}", is_match=1,
            ems_person_id=f"person_{i:05d}", ehr_person_id=f"person_{i:05d}",
            ems_record_id=f"EMS{i:06d}", ehr_record_id=f"EHR{i:06d}",
            ems=ems, ehr=ehr,
            ems_incident_time=ems_incident_time, ehr_admit_time=ehr_admit_time,
        ))

    # --- Non-match pairs ----------------------------------------------------
    for j in range(n_non_matches):
        i = n_matches + j
        ems_person = build_person(fake, rng)
        ehr_person = build_person(fake, rng)
        ehr_mrn = f"MRN{200000 + j}"
        # An MRN-shaped value the EMS crew might have jotted down for the
        # WRONG patient (e.g. stale insurance card) -- deliberately does
        # not match ehr_mrn, since these two records are different people.
        wrong_mrn = f"MRN{300000 + j}"

        ems_incident_time = fake.date_time_between(start_date="-2y", end_date="now")
        # Admit time is unrelated to the EMS incident time here, since these
        # two records don't actually belong to the same real-world event.
        ehr_admit_time = fake.date_time_between(start_date="-2y", end_date="now")

        ems = build_ems_record(ems_person, fake, rng, apply_identity_noise=False, known_mrn=wrong_mrn)
        ehr = build_ehr_record(ehr_person, ehr_mrn, fake, rng)

        rows.append(_pair_row(
            pair_id=f"P{i:05d}", is_match=0,
            ems_person_id=f"person_{i:05d}a", ehr_person_id=f"person_{i:05d}b",
            ems_record_id=f"EMS{i:06d}", ehr_record_id=f"EHR{i:06d}",
            ems=ems, ehr=ehr,
            ems_incident_time=ems_incident_time, ehr_admit_time=ehr_admit_time,
        ))

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
