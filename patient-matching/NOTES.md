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

## Step 5: Tier 3 — probabilistic matching with Splink (2026-08-30)

### Where this lives

- `src/probabilistic_matcher.py` — the whole pipeline: reshape data,
  define the model, train it, predict, save results.
- `data/splink_trained_model.json` — the trained model's full
  configuration (every comparison, every level, every learned m/u
  probability). This is what "the config" means below — open this file to
  see exactly what the model learned, or hand it to
  `Linker(df, settings="data/splink_trained_model.json", db_api=...)` to
  reload a working model without retraining.
- `data/splink_predictions_full_cross_join.csv` — every EMS record scored
  against every EHR record (500 × 500 = 250,000 comparisons). **Not
  committed to git** (~56MB, would permanently bloat this repo) — it's
  fully reproducible by re-running `src/probabilistic_matcher.py` with the
  same generator output, so regenerate it locally if you need it.
- `data/splink_predictions_designed_pairs.csv` — the same predictions,
  filtered down to just our original 500 designed (250/250) pairs, for an
  apples-to-apples comparison with Steps 3 and 4's evaluations.

### Plain-language: what Fellegi-Sunter is actually doing

Tier 1 makes no judgment call — it just confirms a shared ID. Tier 2
computed similarity scores per field, but never combined them or learned
which fields to trust more. Tier 3 is where that combination actually
happens, using a real statistical model instead of a hand-picked rule like
"if three out of four fields look similar, call it a match."

For every field we compare, and every distinct **level of agreement** we
define for it (e.g. for first name: exact match / very similar / somewhat
similar / not similar), Fellegi-Sunter estimates two numbers from the
data itself:

- **m probability** — among pairs that really ARE the same person, how
  often do we see this level of agreement? (E.g., what fraction of true
  matches have an *exact* first-name match?)
- **u probability** — among pairs that are NOT the same person, how often
  does this level of agreement happen purely *by coincidence*? (What
  fraction of random, unrelated pairs happen to share the exact same
  first name?)

The ratio **m / u** is the actual evidentiary value of that observation —
how much more likely that agreement pattern is under "same person" than
under "random pair." This is the answer to "how does it decide some
fields matter more": **a field matters more exactly when its m/u ratio is
bigger** — when agreement is common among true matches but rare by
coincidence. A field where both true matches AND random pairs agree
often (like sharing the same U.S. state) has an m/u ratio near 1 and
barely moves the needle, even though it "feels" like agreement. Nothing
about this is guessed by a person — it's measured directly from how
discriminating each field's agreement pattern turns out to be in the
actual data.

To combine evidence across several fields into one number, Splink takes
`log2(m/u)` for each observed level (called the **match weight** for that
level — positive means "evidence for a match," negative means "evidence
against"), and **adds them up** across all the fields being compared,
starting from a prior weight that reflects how likely any two random
records are to match before looking at anything. This addition is only
valid under an assumption — that, conditional on the true match status,
the fields' agreement patterns don't influence each other (a person's
name doesn't affect how likely their DOB is to match, given that they are
or aren't the same person). That's a simplification, but it's the same
one that makes naive Bayes classifiers work reasonably well in practice.
The final summed weight converts back into a probability via a standard
log-odds transform: `probability = 2^weight / (1 + 2^weight)`.

**How m and u actually get learned, without being told the right
answer:** `u` is the easy half — since real matches are rare (in our
data, 250 out of 250,000 possible comparisons), a big *random* sample of
pairs is overwhelmingly non-matches by default, so you can estimate u
directly by sampling and counting, no labels needed
(`linker.training.estimate_u_using_random_sampling`). `m` is harder,
since you don't know in advance which pairs are the true matches (we
happen to, because we generated this data — a real deployment wouldn't).
Splink uses **Expectation-Maximisation (EM)**: start from a rough guess,
use it to estimate how likely each pair in a training subset is to be a
match, recompute the agreement statistics weighted toward the pairs it
currently believes are matches, and repeat until the numbers stop moving.
It bootstraps m from its own evolving belief about which pairs look like
matches — never from ground truth. We ran EM twice, with different
**blocking rules** restricting each run to a plausible-candidate subset
(once restricted to pairs sharing an exact first+last name, once to pairs
sharing an exact date of birth) — a field can't teach the model anything
about its own discriminating power during a round where it was the thing
forced to match by the blocking rule, so running it twice with
complementary blocking rules means every field gets a clean training
round.

### What the model actually learned (real numbers from this dataset)

| Comparison | Level | m | u | weight = log₂(m/u) |
|---|---|---|---|---|
| address | exact match | 0.507 | 0.0005 | **+9.88** |
| date_of_birth | exact match | 0.722 | 0.0008 | **+9.83** |
| last_name_soundex | exact match | 0.974 | 0.0055 | +7.48 |
| last_name | exact match | 0.636 | 0.0032 | +7.62 |
| first_name | exact match | 0.658 | 0.0047 | +7.13 |
| date_of_birth | within 10 years | 0.019 | 0.178 | **−3.27** |
| last_name | "all other comparisons" (no real similarity) | 0.008 | 0.978 | −7.02 |

An exact address or DOB match carries almost as much weight as name and
phonetic evidence *combined* — because a coincidental exact address or
DOB match between two random people is extremely rare (u ≈ 0.0005-0.0008),
so seeing one is powerful evidence, even though address/DOB "feel" like
weaker identifiers than a name. Note one field can carry both positive
*and* negative weights at different levels: "DOB within 10 years" is
actually evidence **against** a match (−3.27) despite sounding like
partial agreement, because true matches are usually either exact or
wildly different on DOB (see Step 4's bimodal DOB noise), so landing in
this vague middle band is itself informative — mostly seen among
non-matches. This is exactly the kind of nuance a hand-picked scoring
rule (like "add 0.5 points for a loose match") would miss, and it's
learned automatically here.

**A build note on data reshaping:** Splink expects one table per data
source (all EMS records, all EHR records) and generates its own candidate
pairs — it can't consume Step 4's already-paired CSV directly, since its
comparisons run on individual per-record columns (`l.first_name` vs
`r.first_name`), not on a number that was already computed by comparing
two records together. That also means Step 4's *similarity scores*
(`first_name_similarity`, etc.) couldn't be handed to Splink directly —
those are pairwise results, and Splink needs to do the pairwise
comparison itself so it can learn from it. What we could and did reuse
directly were the Soundex/NYSIIS codes, since a phonetic code is a
genuine per-record attribute (a property of one name), not a comparison
between two.

**A build note on evaluation scope:** rather than scoring only our
original 500 designed pairs, we scored the full 500×500 = 250,000
EMS-to-EHR cross-join, since that's a more honest picture of the real
problem (an ambulance record has exactly one correct hospital record
among many candidates, not a coin flip) — and then filtered the results
back down to the 500 designed pairs for the label-by-label comparison
below, to stay comparable with every earlier step.

### Results: does the score distribution actually separate the two groups?

```
=== Match probability distribution on the 500 designed pairs ===
              count   mean     std     min  25%  50%  75%     max
non-matches   250.0  0.000  0.0001  0.0000  0.0  0.0  0.0  0.0014
true matches  250.0  0.992  0.0858  0.0178  1.0  1.0  1.0  1.0000
```

**Clean separation, with zero overlap**: every non-match scores at or
below 0.0014; every true match scores at or above 0.0178. Looking at the
underlying `match_weight` (the log-odds score before it gets squashed
into a 0-1 probability, which is more informative here since probability
saturates to 0/1 for anything with a strong signal either way) makes the
gap even clearer: **every non-match sits at −9.4 or lower, every true
match sits at −5.8 or higher** — see the published chart:
https://claude.ai/code/artifact/894dd4f6-6c00-466b-9177-9db36fe9d181,
which bins all 500 pairs by match weight and shows the two populations
as visibly separate humps with an empty valley between them.

The two lowest-scoring true matches (weight −5.8 and −3.96, probability
0.018 and 0.060) are worth naming specifically, because they're not
random noise in the result — they're the same kind of hard case Step 4.5
already flagged: pair `P00234` has "Liu" typo'd to "iu" (dropping the
first letter), which both tanks the Jaro-Winkler name score *and* breaks
the Soundex/NYSIIS phonetic code (a dropped leading letter changes how a
name sounds, not just how it's spelled), stacked with a completely
different (scene) address — so almost every field the model can see
disagrees, and only an exact DOB match argues for "same person." Given
only the fields Splink can see, uncertainty here is the *correct*
response, not a bug — we only know it's actually a match because we
generated the data and kept the answer key.

## Step 6: address-weight diagnostic + evaluation harness (2026-08-30)

### Diagnostic: is the +9.88 address weight trustworthy?

Prompted by the address comparison's learned weight (+9.88, nearly as
strong as DOB's +9.83) looking suspiciously high given that EMS addresses
are supposed to be an unreliable field. Checked both halves of the m/u
ratio directly against the data rather than trusting the trained number
at face value:

**True-match address agreement** (from `data/synthetic_ems_ehr_pairs.csv`,
the Step 2 generator's own output): of 250 true matches, 218 have an
address on both sides (32 have at least one side blank). Of those 218:
**121 (55.5%) are identical, 97 (44.5%) are mismatched.** This lines up
with Step 2's design (40% chance of a scene address, plus some
independent blank-address noise) and looks realistic — no problem here.

**Non-match coincidental address agreement**: checked at two scales.
Among our 250 designed non-match pairs, **0 of 233** non-blank pairs
share an identical address. Among the *entire* 250,000-pair cross-join
(249,750 genuine non-matches), **0 of 225,282** non-blank comparisons
share an identical address — a full-population check, not just a sample.
**The true coincidence rate in this dataset is exactly zero.**

That's the mechanism, but it doesn't fully explain the numbers: Splink's
trained u-probability for an exact address match is **0.000537**, not
literally 0 (a literal 0 would make the weight infinite, which is one
reason model-fitting code generally avoids landing on it). Tracing why:
`linker.training.estimate_u_using_random_sampling()` estimates u by
randomly sampling pairs from the *entire* cross-join and assuming they're
all non-matches — Splink's own docs are explicit that this is an
approximation ("the validity of the u values rests on the assumption
that the resultant pairwise comparisons are non-matches... for large
datasets, this is typically true"). Our dataset doesn't fully satisfy
that assumption: true matches are 250 of 250,000 pairs (0.1%), and since
55.5% of those true matches happen to share an identical address, some of
them inevitably get swept into the "assumed non-match" sample used to
estimate u. The arithmetic is consistent with exactly this: 250 × 0.555 ≈
139 true-match "leaks" spread across a ~250,000-row sample works out to
≈0.00056 — very close to the actual trained value of 0.000537. In other
words, **the learned u for address is essentially measuring diluted
true-match agreement, not genuine non-match coincidence** — because
genuine non-match coincidence is, in this dataset, really zero.

**Verdict: the address weight is not fully trustworthy, and the reason
is a real gap in the synthetic generator, not a modeling mistake.** Two
compounding synthetic-data artifacts inflate it:
1. Faker draws an effectively unique, unrelated full street address for
   every independent fake identity — there's no mechanism for two
   *different* people to plausibly share an address (no simulated family
   members at the same home, no shared apartment complex, no generic
   downtown/PO-box address a call center might reuse). Real-world
   non-match address collisions are not this rare.
2. Because true-match density is unusually high in this small, curated
   dataset (0.1%) compared to a real record-linkage deployment (often
   far below 0.01%), Splink's u-sampling approximation leaks more
   true-match signal into the "assumed non-match" pool than it would at
   real-world scale.
Both effects push the u-probability artificially low, which inflates
`log2(m/u)` for address independent of whether address is actually that
strong a real-world signal. **Follow-up:** add deliberate non-match
address collisions to the Step 2 generator (shared family address,
common apartment-complex address, a small pool of reused generic
addresses) before trusting this weight for anything beyond this
prototype.

### Evaluation harness

`src/evaluate_matching.py` re-runs Step 5's training/prediction pipeline
in memory (the 56MB full cross-join file isn't committed to git, so this
regenerates it fresh in under a second rather than depending on a stale
local file) and reports:

**1. Precision/recall/F1 across thresholds — full 250,000-pair
cross-join** (the realistic, heavily imbalanced population, not the
curated 500-pair set):

| threshold | tp | fp | fn | precision | recall |
|---|---|---|---|---|---|
| 0.001 | 250 | 861 | 0 | 0.2250 | 1.0000 |
| 0.05  | 249 | 86  | 1 | 0.7433 | 0.9960 |
| 0.5   | 248 | 22  | 2 | 0.9185 | 0.9920 |
| 0.9   | 248 | 10  | 2 | 0.9612 | 0.9920 |
| 0.99  | 245 | 2   | 5 | 0.9919 | 0.9800 |

This is the number that actually matters for picking a threshold, and
it's a meaningfully different (harder) picture than Step 5's headline
result: at threshold 0.5, precision on the full population is **0.9185**,
not the 1.0 the curated 500-pair evaluation showed. Confusion matrix at
0.5:

```
                     predicted match   predicted non-match
actual match                    248                     2
actual non-match                 22               249,728
```

**What the 22 false positives actually look like** (inspected directly,
not just counted): the single highest-scoring false positive (probability
0.993) is "James Smith" vs. "James Smith" — two genuinely different
people in the synthetic data who happen to share a common first and last
name, with completely different addresses and one missing DOB. The model
currently gives an exact name match the same weight regardless of how
common the name is — Splink supports **term-frequency adjustments** for
exactly this (a rarer name should count as stronger evidence than a
common one), and `cl.NameComparison` even sets up the metadata for it,
but we never actually estimated/enabled term frequency tables in Tier 3.
**This is a concrete, evidence-based argument for adding term-frequency
adjustment in a future revision**, not a hypothetical one — it's the
single largest error class the full-population evaluation surfaced.

**2. Confusion matrix on the 500 designed pairs** (for continuity with
Steps 3-5): precision 1.0000, recall 0.9920 (248/250, same 2 misses as
above — the curated set simply never generates a "James Smith" style
coincidence, since its 250 non-match identities were drawn independently
and none happened to collide).

**3. Breakdown by EMS-side MRN presence** (500 designed pairs, threshold
0.5) — checking whether Tier 3 is equally reliable regardless of whether
Tier 1 also had a shot at a given pair:

| group | n | true matches | precision | recall |
|---|---|---|---|---|
| MRN absent on EMS side | 469 | 239 | 1.0000 | 0.9916 |
| MRN present on EMS side | 31 | 11 | 1.0000 | 1.0000 |

No meaningful difference — Tier 3 doesn't lean on MRN presence to perform
well, which is the right result (MRN, when present, is Tier 1's job
entirely; Tier 3 needs to carry pairs where it's absent, which is 94% of
true matches per Step 3).

**4. Breakdown by last-name length** (longer of the two sides, 500
designed pairs, threshold 0.5) — checking whether the short-name
Jaro-Winkler weakness (Step 4.5's "Liu"→"iu", Step 5's lowest-scoring
true match) still shows up as a measurable recall gap now that Soundex is
in the model:

| last name length | n | true matches | precision | recall | missed |
|---|---|---|---|---|---|
| ≤4 chars | 39 | 38 | 1.0000 | 0.9737 | 1 |
| 5-7 chars | 328 | 164 | 1.0000 | 0.9939 | 1 |
| 8+ chars | 133 | 48 | 1.0000 | 1.0000 | 0 |

**The gap is still there, and it's exactly where expected**: the ≤4-char
bucket has the lowest recall (97.4%, missing 1 of 38 — this is "Liu"→"iu"
itself), and 5-7 chars misses one more (the "Berry"→"erry" case from
Step 5). Adding Soundex/NYSIIS to the model did **not** close this gap,
because the specific failure mode is a dropped *first* letter, which
breaks the phonetic code the same way it breaks Jaro-Winkler (a name's
sound, not just its spelling, changes when its first letter is missing).
Both pairs still cleared the classification threshold overall only
because DOB and other fields carried enough weight to compensate — see
the caveat below about why that safety margin may not be real-world
representative.

### Caveat: this separation reflects synthetic noise levels, not real-world performance

Step 5 reported zero overlap between true-match and non-match probability
distributions on the 500 designed pairs, and it's worth being explicit
about what that does and doesn't mean now that this step has looked more
closely. **That clean separation is a property of how much noise Step
2's generator currently injects, not a validated real-world performance
number.** Two results from this step make that concrete:
- The full 250,000-pair evaluation (which Step 5 didn't originally run)
  already surfaces a real 0.9185 precision at a plausible threshold, not
  1.0 — imperfection was there all along, just invisible in the curated
  500-pair comparison, which was built for clarity, not statistical
  representativeness.
- The last-name-length breakdown shows the two hardest cases still barely
  clear the bar, papered over by DOB and other fields being clean on
  those specific pairs. A dataset with harder, more compounding noise
  (multiple noisy fields stacking on the same pair more often, a wider
  variety of typo/OCR/mishearing patterns, realistic address collisions
  per the diagnostic above) would likely show a real, currently-invisible
  precision/recall trade-off region rather than the near-total separation
  reported so far.

**Bottom line: treat every precision/recall number in this project so far
as "how this pipeline performs on deliberately illustrative synthetic
noise," not as a production-readiness claim.** Before trusting these
numbers for anything beyond learning the mechanics, the generator needs a
revision pass adding: realistic non-match address collisions (per the
diagnostic above), last-name noise that also affects the first letter
more often (to stress-test the phonetic-code weakness directly), and
compounding multi-field noise on a higher fraction of true matches.

## Step 7: operating thresholds — auto-match / review / reject (2026-08-30)

### The decision

`src/threshold_analysis.py` computes precision and recall across every
possible `match_probability` cutoff (via
`sklearn.metrics.precision_recall_curve`) on the full, realistic 250,000-
pair cross-join, and sets a **three-way policy** rather than a single
threshold:

```
match_probability >= 0.995              -> AUTO-MATCH (no human review)
0.015 <= match_probability < 0.995      -> MANUAL REVIEW QUEUE
match_probability < 0.015               -> AUTO-REJECT (treated as non-match)
```

Full precision/recall-vs-threshold chart (log-odds x-axis, both cutoffs
marked): https://claude.ai/code/artifact/a2b0421c-3985-4588-8fde-c41a70ca55e2

**A single cutoff forces every pair into "confident yes" or "confident
no," with no room for genuine uncertainty.** Given the framing from Step
6 — a false-positive *merge* (combining two different patients' records)
is the costly error, meaningfully worse than a record that just needs a
person to glance at it — the review band exists to absorb exactly that
uncertainty instead of forcing a bad guess either direction.

### Why 0.995 and 0.015, specifically

Both numbers come directly from the full 250,000-pair evaluation, not
round-number guessing:

- **Auto-match threshold = 0.995** is the *lowest* probability in the
  entire evaluation that produces **zero observed false positives** —
  the most conservative cut point the data actually supports. It
  auto-matches 243 of the 250 true matches with a perfect 243/243
  precision record; the other 7 true matches (2.8%) fall just short and
  go to manual review instead of being auto-matched. Lowering this
  threshold would auto-match more true matches automatically, but the
  curve shows exactly what that costs: at 0.9, precision drops to
  0.9612 (10 false merges among 258 auto-matched pairs); at 0.5, precision
  is only 0.9185 (22 false merges). Given the stated priority, trading
  a small amount of automation for a verified-zero false-merge rate on
  this evaluation is the deliberate choice being made here — see the
  caveat below on what "verified-zero" should and shouldn't be taken to
  mean.
- **Review floor = 0.015** is set just below 0.0178 — the lowest
  probability any true match ever received in this entire evaluation
  (the "Liu"→"iu" pair tracked since Step 4.5). Anything scoring below
  this is auto-rejected with no human ever seeing it, so this floor is
  chosen specifically so that **zero true matches are ever silently
  discarded** — every real match in this evaluation clears it into at
  least the review queue.

### What this policy produces on the full 250,000-pair evaluation

| Band | n | true matches | non-matches |
|---|---|---|---|
| Auto-match (≥0.995) | 243 | 243 | 0 |
| Review queue [0.015, 0.995) | 312 | 7 | 305 |
| Auto-reject (<0.015) | 249,445 | 0 | 249,445 |

- Auto-match precision: **100%** (243/243) on this evaluation.
- 97.2% of all true matches get auto-matched; the remaining 2.8% go to
  review, none are silently lost.
- The review queue is small relative to the whole population — **0.125%
  of all 250,000 candidate pairs** (312 of them) — and is where a human
  actually has to look, at a ratio of roughly 1 real match for every 44
  non-matches in that queue.

### This is a business tradeoff — the deliberate alternative framing

Two things a reader should be able to redo with different priorities:

- **Lowering the auto-match threshold to 0.9** would auto-match all 250
  true matches with zero misses at the auto-match stage, but would also
  auto-match **10 real non-matches** as if they were confirmed — the
  exact false-merge risk this recommendation exists to avoid. That's a
  legitimate choice if false merges are cheaper to detect/undo downstream
  than this project assumes, or if review-team capacity is the binding
  constraint instead.
- **Raising the review floor** would shrink the review queue further
  (less staff time spent on obvious non-matches) but risks pushing a
  genuine match below the floor where nobody ever looks at it again —
  a silent miss rather than a caught, correctable one.

Whether spending 2.8% of true matches' worth of manual review time to
guarantee zero known false merges is the right trade is a call about
review-team capacity and the real-world cost of a bad merge versus a
missed one. That's not something this analysis can settle — it's flagged
here explicitly so the choice was made on purpose, not defaulted into.

### Caveat this decision inherits from Step 6

The "zero false positives" and "100% precision" figures above describe
this specific synthetic evaluation, not a real-world guarantee — Step 6
already flagged that the near-total separation in this dataset likely
reflects insufficient/unrealistic noise (no non-match address collisions,
no common-name term-frequency weighting, limited compounding multi-field
noise) rather than genuine 100% reliability. **Once the generator
revisions listed in Step 6 land, re-run `threshold_analysis.py` and
expect these exact numbers to move** — probably requiring a stricter
(higher) auto-match threshold to keep the same zero-false-positive
guarantee against harder, more realistic noise. Treat 0.995/0.015 as a
first, defensible operating point on the evidence available today, not a
constant to hard-code and forget.

## Open questions / things to decide next

- Re-run `src/threshold_analysis.py` after the Step 6 generator revisions
  (realistic address collisions, term-frequency adjustment, more
  compounding noise) land, and expect the 0.995/0.015 operating
  thresholds to need re-justifying against a harder evaluation.
- Add term-frequency adjustments to Tier 3's name comparisons (Splink
  supports this; `cl.NameComparison` already sets up the metadata) — the
  full-population evaluation's worst false positive ("James Smith" vs.
  "James Smith") is a direct, concrete case for it.
- Revise the Step 2 generator to add: realistic non-match address
  collisions (shared family/apartment/generic addresses), last-name noise
  that more often corrupts the first letter (to properly stress-test the
  Soundex/NYSIIS weakness), and a higher rate of multiple noisy fields
  compounding on the same true-match pair.
- `last_name`'s "Jaro-Winkler distance >= 0.7" level never got enough
  training examples in either EM round to learn an m probability (Splink
  falls back to a default) — a third EM round with a different blocking
  rule, or a bigger dataset, would likely resolve this. Logged, not fixed,
  since the final separation was already clean without it.
- The `address` comparison ended up as a plain exact-match after checking
  the real Levenshtein-distance distribution on true matches: it's sharply
  bimodal (identical, or a totally different scene address) with nothing
  in between, so multi-threshold levels had nothing to train on. Revisit
  if we ever add address noise that's a genuine partial edit (e.g. a
  typo'd street number) rather than a wholesale different address.
- Should we calibrate the noise probabilities (or the race/ethnicity
  population weights) against any published research, or is illustrative
  noise good enough for a learning prototype?
- Do we eventually want a version of this generator that creates messier,
  larger-scale data (e.g. thousands of records, multiple visits per
  person) to stress-test performance, once the core matching logic works?
- If we ever need field-level address comparison (street vs. city vs.
  state), we'll need to split `address` into USCDI's component
  sub-elements rather than comparing it as one free-text string.
