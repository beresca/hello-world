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

## Open questions / things to decide next

- What fields will our synthetic EMS vs. hospital records actually share?
  (Likely candidates: name, DOB, sex, address, phone, incident/admission
  date-time.) Need to decide the "messiness" we simulate (typos, missing
  fields, nicknames) so it resembles real-world data quality issues.
- Do we build our own scoring/matching logic first (using rapidfuzz) to
  understand the mechanics, before bringing in splink's probabilistic
  model? (Leaning toward yes — build intuition manually first, then let
  splink automate/improve it.)
- How will we evaluate whether a match is "correct"? Since we're
  generating the synthetic data ourselves, we can keep a hidden "ground
  truth" ID to check our matching logic against — need to design that in
  from the start.
