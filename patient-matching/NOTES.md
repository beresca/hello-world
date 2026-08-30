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

## Open questions / things to decide next

- Build Tier 2 (fuzzy matching with rapidfuzz on name/DOB/address) next,
  to catch the 227 true matches Tier 1 correctly leaves undecided — measure
  its recall *and* watch for false positives, since fuzzy scoring is where
  wrong-but-confident matches start becoming possible.
- Should we calibrate the noise probabilities against any published
  research on ePCR data quality, or is illustrative noise good enough for
  a learning prototype?
- Do we eventually want a version of this generator that creates messier,
  larger-scale data (e.g. thousands of records, multiple visits per
  person) to stress-test performance, once the core matching logic works?
