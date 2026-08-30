# Patient Record Matching — Project Notes

## Goal

Build a prototype that matches EMS ambulance run records (ePCR — electronic
Patient Care Report) to hospital records (EHR/ADT — Electronic Health
Record / Admission-Discharge-Transfer feed) for the *same patient*, even
when:

- There's no shared ID number linking the two systems (EMS and hospital
  systems are usually completely separate databases).
- Names have typos, nicknames, or are transcribed slightly differently
  ("Jon" vs "John", "Kathy" vs "Katherine").
- Fields are missing or partially filled in (unconscious patients, rushed
  documentation in the field, etc.).
- Dates of birth, addresses, or phone numbers are formatted differently or
  have small entry errors.

This is a form of **record linkage** / **entity resolution** — the general
problem of figuring out when two records from different data sources refer
to the same real-world entity, without a perfect shared key. It shows up
constantly in healthcare, but also in things like deduplicating customer
databases or matching voter rolls.

We're building this as a learning project as much as a working prototype —
notes here should capture *why* we made a decision, not just *what* we
did, since that's the part that doesn't show up by reading the code later.

## Project structure

```
patient-matching/
├── data/           # synthetic/sample data lives here (never real PHI)
├── src/            # our actual Python code (matching logic, helpers)
├── tests/          # automated tests for that code
├── .venv/          # the virtual environment (not committed - see below)
├── requirements.txt
├── .gitignore
└── NOTES.md        # this file
```

## Step 1: Project setup (2026-08-30)

### Virtual environment

Created with `python3 -m venv .venv`. A virtual environment is an isolated
copy of Python + its own package folder, separate from whatever Python the
rest of the machine uses. Why bother: without one, `pip install` affects
the *entire machine's* Python, so two projects that need different
versions of the same library can silently break each other. The venv
keeps this project's dependencies self-contained and reproducible —
anyone (including future us) can recreate the exact same environment from
`requirements.txt` without guessing.

To use it in a terminal: `source .venv/bin/activate` (then `python`, `pip`,
etc. all refer to the venv's copies). To leave it: `deactivate`.

### Libraries installed

- **pandas** — the standard library for working with tabular data (rows and
  columns) in Python, similar in spirit to a spreadsheet or a SQL table but
  manipulated with code. We'll use it to load, clean, and inspect the EMS
  and hospital records.
- **rapidfuzz** — fast fuzzy string matching. Answers "how similar are
  these two strings?" (e.g. "Kathy Smith" vs "Katherine Smith") with a
  similarity score, using algorithms like Levenshtein distance. This is
  core to matching names/addresses that aren't typed identically.
- **Faker** — generates realistic-looking fake data (names, addresses,
  dates of birth, phone numbers). We'll use it to build a synthetic dataset
  to develop and test against, since we obviously can't use real patient
  data (PHI) for a prototype.
- **splink** — a purpose-built probabilistic record linkage library
  (originally from the UK Ministry of Justice). Instead of hand-writing
  matching rules, it learns statistically how much weight to give each
  field (name similarity, DOB match, address match, etc.) and produces a
  match *probability* for each candidate pair of records. This is the
  closest thing to "the real tool for this exact job."
- **scikit-learn** — general-purpose machine learning toolkit. We may use
  it for evaluation metrics (precision/recall on our matches) and possibly
  as an alternative/simpler classifier if we want to compare against
  splink's approach.

Exact installed versions are pinned in `requirements.txt` (via
`pip freeze`) so the environment is reproducible later.

### Version control

Initialized as a subfolder of the existing `hello-world` git repo, on
branch `claude/patient-record-matching-setup-wac2hx`. `.gitignore` excludes
the `.venv/` folder (rebuildable from `requirements.txt`, and large) and
the contents of `data/` (real-world record-linkage work involves PHI, so
we default to not committing data files even though this prototype only
uses synthetic data).

## Step 2: Synthetic dataset (2026-08-30)

### What it is

`src/generate_synthetic_data.py` generates `data/synthetic_ems_ehr_pairs.csv`
— 500 candidate pairs, each one row with an EMS (ePCR) record and an EHR
(ADT) record side by side, plus a ground-truth `is_match` column (1 = same
real person, 0 = different people). Split is exactly 250/250. This is our
"answer key": later, when we build the actual matching/scoring logic, we
run it on this file *without* letting it see `is_match`, then compare its
guesses against that column to measure precision/recall. `is_match` (and
the debug-only `ems_person_id`/`ehr_person_id` columns) must never be used
as an input feature to the matcher itself — that would be cheating, since
a real system will never have that column.

### How the data is generated

For each **true match**, one clean "ground truth" person is generated with
Faker, then two derived views are built from it: a clean-ish EHR record
(the hospital's own system of record, so it always has a valid MRN) and a
noisier EMS record. The EMS side gets several *independent, probabilistic*
noise types layered on — meaning not every true match is equally messy;
some pairs are nearly identical, some are quite corrupted, which mirrors
real variance in documentation quality:

- **Name**: nickname substitution (from a small hand-built map like
  Robert→Bob, Katherine→Kathy) or a random single-letter typo (swap/drop/
  duplicate) — 35% chance.
- **DOB**: transposed day/month, or the year off by 1-2 — 25% chance.
- **Address**: replaced with a random Faker address to represent the EMS
  crew logging *where they found the patient* (scene of the 911 call)
  rather than their home address — 40% chance.
- **Missing fields**: DOB (5%), address (10%), phone (15%) blanked out
  entirely, representing rushed or incomplete field documentation
  (unconscious patient, refused history, etc.).
- **Sex entry error**: rare (2%) flip of M/F, modeling a data-entry mistake.
- **MRN**: EMS crews essentially never know the hospital's internal MRN
  ahead of time, so it's blank ~90% of the time. The remaining ~10%
  represents a case where the crew captured it from an insurance/ID card
  on scene.

**Non-match pairs** pull two *independent* Faker people — no identity
noise is injected (they don't need help looking different) — but the same
generic missingness (blank phone/address/DOB) is still applied, so a
matcher can't cheat by using "has this field" as a proxy for "is a match."
For the small fraction of non-matches where an EMS-side MRN-like value
exists, it's deliberately a *different* value than the EHR's real MRN
(e.g. a stale insurance card), rather than just being blank.

Everything is seeded (`Faker.seed()` / `random.seed()`, default 42) so the
same command regenerates an identical file — useful for reproducibility
and for debugging.

To regenerate: `.venv/bin/python src/generate_synthetic_data.py --n-pairs 500`
(flags: `--n-pairs`, `--seed`, `--out`).

### Known limitations of this dataset

- **It's synthetic all the way down.** Faker produces plausible-looking
  but statistically simplistic names/addresses (e.g. no real-world name
  frequency distribution, no genuinely ambiguous common-surname clusters
  at scale). Real EMS/EHR data will have messier, less uniform noise than
  what we hand-coded here.
- **Noise types are our own guesses**, not measured from real ePCR/EHR
  error rates. The probabilities (35% typo, 25% DOB error, etc.) are
  illustrative, not calibrated to any real dataset.
- **Only one noise event per field per record**, not compounding
  multi-error chaos that real rushed documentation can produce.
- **No duplicate/multiple hospital visits** — every person appears exactly
  once as a match candidate; a real system also has to handle the same
  person showing up multiple times across many records.
- **Committed to git** as `data/synthetic_ems_ehr_pairs.csv` since it's
  fully synthetic (no real PHI) — `.gitignore` blocks everything else
  under `data/` by default, with a narrow exception for `synthetic_*.csv`
  files specifically, so a real data file dropped in `data/` won't
  accidentally get committed.

## Step 3: Tier 1 — deterministic matching (2026-08-30)

### What it is

`src/deterministic_matcher.py` implements the first, most conservative
matching tier: it only calls two records a match if they share the exact
same value on a strong identifier field (currently just `mrn` — our
dataset has no SSN field to also check, but the code is written so another
strong ID could be added as a second field pair without restructuring
anything). Both sides must have a *non-blank* value for the field before
comparing — a blank on one side matching a blank on the other is explicitly
NOT treated as a match, since "both unknown" isn't evidence of anything.
This tier makes no judgment calls about *how similar* two records look; it
either finds a confirmed shared ID or it doesn't.

### Results on the synthetic dataset

Run: `.venv/bin/python src/deterministic_matcher.py`

```
Total pairs evaluated:        500
True matches in dataset:      250
Non-matches in dataset:       250

Pairs flagged as a match:     23
  - correct (true positive):  23
  - wrong (false positive):   0

Recall on true matches:       9.2%
Precision:                    100.0%
```

**Interpretation:**
- **Precision is 100% and zero false positives were produced.** This
  matches the design goal — a tier based on confirmed shared IDs should
  essentially never be wrong, since it isn't guessing.
- **Recall is only 9.2%** — it catches 23 of the 250 true matches and
  correctly ignores every one of the other 227. That's expected, not a
  bug: our generator only gives the EMS side a captured MRN ~10% of the
  time (real ambulance crews usually don't know the hospital's internal
  MRN), so ~90% of true matches structurally *can't* be resolved this way
  — there's no shared ID to check in the first place.
- The other 227 true matches (and all 250 non-matches) are pairs this
  tier correctly declines to call either way — it isn't wrong about them,
  it's silent about them. That silence is exactly the gap the next tier
  (fuzzy matching on name/DOB/address with rapidfuzz) needs to close.

### Why this design choice matters

A tempting shortcut would be to also match on "close enough" IDs (e.g. MRN
off by one digit), but that would blur this tier's whole purpose. Tier 1
exists to be the one part of the system you can trust unconditionally; any
fuzziness belongs in a later, explicitly probabilistic tier where we can
also quantify the uncertainty.

## Step 3.5: Align schema to USCDI (2026-08-30)

### What changed and why

Before building the next matching tier, we renamed/restructured the
synthetic dataset's fields to match **USCDI (United States Core Data for
Interoperability)**'s *Patient Demographics/Information* data class — the
standardized field set real EMS and hospital systems are expected to
report. Reasoning: we're going to spend a lot more effort building
matching logic *on top of* this schema, so it's worth paying the
retrofit cost now, while it's cheap, rather than after Tier 2/3 logic is
already wired to old field names.

**Renamed** (old → new):
| Old | New | Why |
|---|---|---|
| `sex` | `birth_sex` | matches USCDI's Birth Sex element name; values changed from `M`/`F` to `Male`/`Female` |
| `dob` | `date_of_birth` | matches USCDI element name |
| `phone` | `phone_number` | matches USCDI element name |
| `mrn` | `patient_identifier_mrn` | see below — deliberately *not* named like a demographic field |

**Added** (new fields, not previously in the dataset): `middle_name`,
`suffix`, `previous_name`, `race`, `ethnicity`, `email_address` — all part
of USCDI's Patient Demographics/Information element list.

**Why the identifier is named/treated differently:** USCDI does not
classify a medical record number as a demographic attribute — it's a
local, hospital-issued administrative identifier, not something that
describes the patient. Naming it `patient_identifier_mrn` (rather than,
say, folding it in as just another demographic column) keeps that
distinction visible in the schema itself: Tier 1 works on the identifier
field, and every tier after this one will work on demographic fields —
two genuinely different kinds of evidence, and the column names now say
so rather than leaving it implicit.

**Race/ethnicity value sets** follow the OMB (Office of Management and
Budget) categories USCDI points to: `White`, `Black or African American`,
`Asian`, `American Indian or Alaska Native`, `Native Hawaiian or Other
Pacific Islander`, `Two or More Races` for race; `Hispanic or Latino` /
`Not Hispanic or Latino` for ethnicity. Population weights used to
generate these are illustrative guesses, not derived from real census
data — fine for a learning prototype, not something to cite.

**New fields follow the same "realistic capture gap" pattern as Step 2's
noise, not brand-new mechanics:**
- `previous_name` and `email_address` are **always blank on the EMS
  side** — not probabilistic noise, but a structural fact: ePCR/NEMSIS-
  based ambulance documentation generally has no field at all for a
  patient's prior legal name or email address. The hospital side
  populates both most of the time (with a small realistic gap — some
  patients decline to give an email at registration).
- `race`/`ethnicity` *are* real NEMSIS fields EMS crews do capture, but at
  lower reliability than a hospital registration desk: our generator
  blanks/marks them "Unknown" more often on the EMS side (blank 10% /
  "Unknown" 25%) than the EHR side (blank 2% / "Unknown" 8%).
- `middle_name`/`suffix` follow the existing "EMS abbreviates or drops
  what the hospital records in full" pattern already used for other
  fields.

**Address stayed a single free-text field** (not split into
street/city/state/zip components) — USCDI does define these as separate
sub-elements, but the task only asked to align the field list, and a
single string is enough for the similarity-scoring work ahead. Worth
revisiting if we ever need field-level (not whole-address) comparison.

### Re-verifying Tier 1 after the schema change

Ran `.venv/bin/python src/generate_synthetic_data.py --n-pairs 500` to
regenerate the dataset, then `.venv/bin/python src/deterministic_matcher.py`
(updated to check `ems_patient_identifier_mrn` /
`ehr_patient_identifier_mrn` instead of the old `ems_mrn`/`ehr_mrn`):

```
Pairs flagged as a match:     15
  - correct (true positive):  15
  - wrong (false positive):   0

Recall on true matches:       6.0%
Precision:                    100.0%
```

**Still 100% precision, zero false positives — Tier 1 passes.** Recall
moved from 9.2% (Step 3) to 6.0% here; this is expected sampling
variation, not a regression. The underlying rule generating an EMS-side
MRN is still "10% chance, and always correct when present" — unchanged
mechanically — but adding several new fields to `build_person`/
`build_ems_record` shifted *when* each pseudo-random draw happens in the
sequence, so the same seed produces a different specific set of "did this
particular record get an MRN" outcomes. Verified directly: of the 250 true
matches, exactly 15 got a non-blank EMS MRN, and all 15 equal the EHR MRN
(0 mismatches); of the 250 non-matches, 22 got a non-blank EMS MRN and
none accidentally collided with the EHR MRN (the ID numbering ranges —
100000s/200000s/300000s — make that structurally impossible, not just
unlikely). The mechanism is intact; only the specific random draws changed.

## Step 4: Similarity feature layer (2026-08-30)

### What it is

`src/similarity_features.py` computes, for every pair, a set of numeric
similarity scores comparing the EMS and EHR demographic fields — the
evidence a later decision layer (a threshold rule, then splink) will
reason over for the pairs Tier 1 can't resolve on its own. It writes a
new file, `data/synthetic_ems_ehr_pairs_features.csv`, containing the
original columns plus the scores below plus `tier1_deterministic_match`
(so the pairs Tier 1 already resolved are still identifiable, even though
we compute scores for them too — see "why compute for every pair" below).
We deliberately did **not** overwrite `synthetic_ems_ehr_pairs.csv` — that
file stays the canonical raw generator output other scripts (like
`deterministic_matcher.py`) depend on; this is a derived, feature-enriched
copy.

### The four techniques, and what each one measures

**1. Name similarity — Jaro-Winkler (`first_name_similarity`,
`last_name_similarity`)**
A character-based edit-distance score from 0 (nothing alike) to 1
(identical), computed via `rapidfuzz.distance.JaroWinkler`. Jaro-Winkler
gives extra credit for a shared *prefix*, which fits how people actually
misspell names — the first couple of letters are usually right, and the
mistake is further in ("Timothhy" vs "Timothy", "Jaon" vs "Jason"). We
compute it separately for first and last name (rather than one score on
the full name) so a typo in one doesn't get diluted by an exact match in
the other. Both strings are lowercased first so a capitalization
difference between systems doesn't register as a spelling difference.

**2. DOB similarity — custom component comparison (`dob_similarity`,
`dob_match_type`)**
A plain string-distance metric is a poor fit for dates: "1999-02-18" vs
"1999-08-12" looks fairly similar as a string despite being an entirely
different birthday. Instead, `compare_dob()` checks which parts
(year/month/day) actually agree and scores accordingly — exact match
(1.0), a day/month transposition like our generator's noise produces
(0.85), same month & day but a different year, i.e. a misremembered birth
year (0.7), and progressively weaker partial matches down to no shared
components (0.0). A blank DOB on either side returns `NaN`/`"missing"`,
never `0.0` — a missing field is an absence of evidence, not evidence of
a mismatch, and conflating the two would make the feature actively
misleading.

**3. Address similarity — token-based comparison (`address_similarity`)**
Uses `rapidfuzz.fuzz.token_sort_ratio`, which splits each address into
words, sorts them, and compares — unlike Jaro-Winkler, this isn't thrown
off by word reordering or extra tokens (apartment numbers, differently
ordered city/state). This is the right tool for a multi-token string like
an address, where Jaro-Winkler (built for short, single-token strings
like names) would be the wrong fit. Important caveat: a low score here
doesn't necessarily mean "different person" — the EMS scene address and
the EHR home address can legitimately differ for the same real person.
That ambiguity is exactly why this is one input among several, not a
standalone decision.

**4. Phonetic encoding — Soundex and NYSIIS on last name
(`last_name_soundex_match`, `last_name_nysiis_match`)**
Both algorithms collapse a name to a short code representing roughly how
it *sounds*, catching spelling variants that look very different
character-by-character but would be pronounced the same ("Smith" vs
"Smyth" both Soundex to `S530`). Soundex is the older, simpler algorithm;
NYSIIS is newer and generally more accurate for American names but uses
different rules, so the two don't always agree — comparing both surfaces
more signal than trusting either alone. Needed the `jellyfish` library for
this (added to `requirements.txt`) since rapidfuzz does string-distance
metrics, not phonetic encoding — a genuinely different technique family.

### Sanity-check results

Mean scores by ground truth label, across all 500 pairs:

| | first_name_sim | last_name_sim | dob_sim | address_sim | soundex_match | nysiis_match |
|---|---|---|---|---|---|---|
| **true matches** | 0.971 | 1.000 | 0.950 | 0.795 | 1.000 | 1.000 |
| **non-matches** | 0.398 | 0.375 | 0.008 | 0.377 | 0.004 | 0.004 |

Clear separation on every feature — exactly what we'd want going into a
threshold or probabilistic decision layer. Spot-checking individual true
matches Tier 1 did *not* resolve (no shared MRN) confirms the scores
behave sensibly even under the injected noise:

- `Marria` / `Maria` → 0.961 name similarity (a typo, correctly scored as
  "close").
- DOB `2005-11-15` / `2007-11-15` → 0.7, labeled
  `month_day_match_year_differs` (exactly the year-corruption noise from
  Step 2, correctly recognized as strong partial evidence rather than a
  flat "no match").
- DOB `1966-06-21` / `1965-06-21` → also 0.7, same reasoning.
- Exact matches on name/DOB/address (no noise landed on that particular
  pair) correctly score 1.0 across the board.

Non-match pairs show the opposite pattern: name similarities in the
0.0–0.6 range (some accidental partial overlap is expected — e.g. "Karen"
vs "Adam" scores 0.48 purely by character coincidence, which is a good
reminder that any single fuzzy score can be misleading on its own), DOBs
mostly `no_match` (0.0), and only 1 of 250 non-matches showing a
coincidental phonetic collision (a real phenomenon: two unrelated people
whose surnames happen to sound alike).

### A known limitation this surfaced

~~The `last_name_similarity`/phonetic features show **zero variation on
true matches in this dataset** — every one of the 250 true matches has an
exactly identical `last_name` on both sides, because the Step 2 noise
generator only ever injects typos/nicknames into `first_name`, never
`last_name`.~~ **RESOLVED 2026-08-30 — see Step 4.5 below.**

## Step 4.5: Add last-name noise to the generator (2026-08-30)

### What changed and why

The limitation logged above was a real gap: real EMS last-name data gets
garbled too (mishearing over radio, transposed letters, inconsistent
hyphenation), so a dataset where `last_name` is always byte-for-byte
identical between EMS and EHR was making our true matches unrealistically
easy on that field — and meant `last_name_similarity` and the phonetic
match features were never actually being exercised.

**Fix, in `src/generate_synthetic_data.py` (now `GENERATOR VERSION: v2`):**
`build_ems_record`'s identity-noise block now also runs `make_typo()` on
`last_name` with the same 35% probability first_name gets its
nickname-or-typo treatment. We reused the existing `make_typo()` function
(swap/drop/duplicate a letter) rather than writing new logic, per the
instruction to keep this a reuse, not a new noise mechanism — and
deliberately did **not** apply nickname substitution to `last_name`,
since "nicknames" are a first-name concept; nobody has a casual nickname
for their own surname the way "Bob" stands in for "Robert."

**Dataset regeneration:** this is v2 of the generator logic. Running
`generate_synthetic_data.py` overwrites `data/synthetic_ems_ehr_pairs.csv`
in place (same file path, same seed/CLI as before) — there is no
separately kept v1 file, but this note plus the git history for that path
is enough to reconstruct what v1 looked like if a future comparison ever
needs it (v1 = the commit at `eab6916`/`f70d11a`, before this change; v2 =
everything from this commit onward).

### Verification

Regenerated with `.venv/bin/python src/generate_synthetic_data.py
--n-pairs 500`, then re-ran `.venv/bin/python src/similarity_features.py`
to produce an updated `data/synthetic_ems_ehr_pairs_features.csv`.

Last-name mismatch rate on true matches is now **88/250 = 35.2%** —
right in line with the 35% probability parameter, matching first name's
noise rate as requested. Example typos produced: `Smth`/`Smith`,
`oRbles`/`Robles`, `Kellly`/`Kelly`, `iHll`/`Hill`.

Updated mean scores by ground truth label (all 500 pairs):

| | first_name_sim | last_name_sim | dob_sim | address_sim | soundex_match | nysiis_match |
|---|---|---|---|---|---|---|
| **true matches** | 0.978 | 0.979 | 0.936 | 0.720 | 0.932 | 0.808 |
| **non-matches** | 0.433 | 0.413 | 0.001 | 0.373 | 0.008 | 0.004 |

`last_name_similarity` on true matches now has real spread (mean 0.979,
std 0.072, min 0.0) instead of being pinned at a constant 1.0 — confirmed
88/250 true matches now score below 1.0 on it. The phonetic features also
now show genuine disagreement: 17/250 true matches (6.8%) have a Soundex
mismatch and 48/250 (19.2%) have a NYSIIS mismatch despite being the same
person, which is exactly the kind of case Tier 2's decision logic needs
to be able to tolerate rather than penalize too harshly.

**A genuinely interesting edge case surfaced by this fix:** one true
match, `Liu` → `iu` (the "drop first letter" typo), scored
`last_name_similarity = 0.0` — not a bug, but a real property of the
Jaro-Winkler algorithm: its comparison window shrinks with string length,
and for a 2-3 character string, dropping the first character leaves no
position where the algorithm considers the remaining letters "aligned,"
so it registers as completely dissimilar even though a human would
recognize it instantly as the same name missing a letter. Worth
remembering once we design Tier 2's decision logic: very short names are
a case where name-similarity scores alone can be misleadingly harsh, and
phonetic codes or a minimum-length-aware fallback might matter more there.

## Open questions / things to decide next

- Build Tier 2's actual decision logic next (a threshold or weighted
  combination over these similarity scores) to catch the true matches
  Tier 1 correctly leaves undecided — measure recall *and* watch for
  false positives, since this is where wrong-but-confident matches start
  becoming possible. Keep the short-name Jaro-Winkler edge case (above)
  in mind when designing thresholds.
- Should we calibrate the noise probabilities (or the race/ethnicity
  population weights) against any published research, or is illustrative
  noise good enough for a learning prototype?
- Do we eventually want a version of this generator that creates messier,
  larger-scale data (e.g. thousands of records, multiple visits per
  person) to stress-test performance, once the core matching logic works?
- If we ever need field-level address comparison (street vs. city vs.
  state), we'll need to split `address` into USCDI's component
  sub-elements rather than comparing it as one free-text string.
