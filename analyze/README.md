# analyze

Fetches trial data from Firestore (public REST API) and/or local CSV files,
merges them, and produces a Markdown statistics report.

## Setup

```bash
cd analyze
uv sync
```

## Usage

```bash
uv run main.py [--csv-dir DIR] [--out FILE] [--min-trials N] [--no-firebase]
```

| Flag | Default | Description |
|---|---|---|
| `--csv-dir` | `csv/` | Directory of local CSV export files |
| `--out` | `report.md` | Output Markdown report |
| `--min-trials` | `10` | Discard participants with fewer trials |
| `--no-firebase` | off | Skip Firestore; use only local CSVs |

**Firestore requirement:** your security rules must include `allow read: true`
on the `trials` collection. No credentials are needed.

### Examples

```bash
# Firestore + any CSVs in csv/
uv run main.py

# Local CSVs only
uv run main.py --no-firebase

# Custom CSV folder, lower participant threshold
uv run main.py --csv-dir data/exports --min-trials 5
```

## Data sources and deduplication

Both sources are merged on `(participant_id, trial_nr)`. If the same trial
appears in both Firestore and a local CSV, the Firestore copy is kept.

## Output (`report.md`)

Six sections:

1. **Dataset Overview** — participant count, total trials, overall accuracy with 95 % CI, median RT, trial counts per source.
2. **Per-Participant Descriptives** — trial count and accuracy per participant.
3. **Accuracy by Difficulty Level** — accuracy + CI for each of the five difficulty levels.
4. **Zahlen vs. Georgisch Descriptives** — accuracy, CI, and median RT per stimulus type.
5. **Zahlen vs. Georgisch Inferential Statistics** — paired t-test, Cohen's d, observed power.
6. **Accuracy by Difficulty × Type** — cross-tabulation.

### Statistical approach

- **Confidence intervals:** Wilson score intervals — correct for binary (0/1) data; the normal approximation breaks down near 0 % or 100 %.
- **Inferential test:** McNemar's exact test — the standard test for paired binary outcomes. A t-test would be wrong here because trial outcomes are not continuous.
- **Pairing:** within each participant, Zahlen and Georgisch trials are paired by trial-number order. The experiment design interleaves both conditions at each difficulty level, so this is a natural within-subject pairing.
- **Effect size:** odds ratio of discordant pairs (b / c), where b = trials Zahlen correct & Georgisch wrong, c = the reverse.
- **Works with any N ≥ 1 participants**, as long as there are at least 2 paired trials total.
