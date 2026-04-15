"""
Fetch trial data from Firestore (public REST API) and/or local CSV files,
run descriptive + inferential stats, and write a Markdown report.

Usage:
    uv run main.py [--csv-dir DIR] [--out FILE] [--min-trials N] [--no-firebase]

Firestore rules must allow public reads on the 'trials' collection.
Local CSVs in --csv-dir are merged with remote data; duplicates
(same participant_id + trial_nr) are dropped, keeping the remote copy.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from scipy import stats


FIREBASE_PROJECT = "tpr-game"
FIRESTORE_BASE   = (
    f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT}"
    "/databases/(default)/documents"
)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Analyse experiment data.")
    p.add_argument("--csv-dir",    default="csv",    help="Directory of local CSV files (default: csv/).")
    p.add_argument("--out",        default="report.md", help="Output Markdown file (default: report.md).")
    p.add_argument("--min-trials", type=int, default=10,
                   help="Discard participants with fewer trials (default: 10).")
    p.add_argument("--no-firebase", action="store_true",
                   help="Skip Firestore fetch; use only local CSVs.")
    return p.parse_args()


# ── FIRESTORE REST ─────────────────────────────────────────────────────────────

def _parse_value(v: dict):
    """Unwrap a Firestore typed value into a plain Python value."""
    if "stringValue"  in v: return v["stringValue"]
    if "integerValue" in v: return int(v["integerValue"])
    if "doubleValue"  in v: return float(v["doubleValue"])
    if "booleanValue" in v: return v["booleanValue"]
    if "nullValue"    in v: return None
    if "arrayValue"   in v:
        items = v["arrayValue"].get("values", [])
        return [_parse_value(i) for i in items]
    if "mapValue"     in v:
        return {k: _parse_value(fv) for k, fv in v["mapValue"].get("fields", {}).items()}
    return None


def fetch_firestore(collection: str = "trials") -> pd.DataFrame:
    """Page through an entire Firestore collection via the REST API."""
    url   = f"{FIRESTORE_BASE}/{collection}"
    rows  = []
    token = None

    while True:
        params: dict = {"pageSize": 300}
        if token:
            params["pageToken"] = token

        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 403:
            print(
                "ERROR: Firestore returned 403. "
                "Make sure your security rules allow `allow read: true` on the trials collection.",
                file=sys.stderr,
            )
            sys.exit(1)
        resp.raise_for_status()
        body = resp.json()

        for doc in body.get("documents", []):
            row = {k: _parse_value(v) for k, v in doc.get("fields", {}).items()}
            row["_source"] = "firestore"
            rows.append(row)

        token = body.get("nextPageToken")
        if not token:
            break

    return pd.DataFrame(rows)


# ── LOCAL CSVs ─────────────────────────────────────────────────────────────────

def load_csv_dir(csv_dir: str) -> pd.DataFrame:
    path  = Path(csv_dir)
    files = sorted(path.glob("*.csv"))
    if not files:
        return pd.DataFrame()

    frames = []
    for f in files:
        df = pd.read_csv(f, dtype=str, encoding="utf-8-sig")
        df["_source"] = f.name
        frames.append(df)
        print(f"  Loaded {len(df)} rows from {f.name}")

    return pd.concat(frames, ignore_index=True)


# ── MERGE ──────────────────────────────────────────────────────────────────────

def merge_sources(remote: pd.DataFrame, local: pd.DataFrame) -> pd.DataFrame:
    """
    Combine remote and local frames.
    If the same (participant_id, trial_nr) appears in both, keep the remote copy.
    """
    combined = pd.concat([remote, local], ignore_index=True)
    # Sort so remote rows come first, then drop duplicates keeping first
    combined["_is_remote"] = combined["_source"] == "firestore"
    combined = combined.sort_values("_is_remote", ascending=False)
    combined = combined.drop_duplicates(subset=["participant_id", "trial_nr"], keep="first")
    combined = combined.drop(columns=["_is_remote"])
    return combined.reset_index(drop=True)


# ── CLEANING ───────────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame, min_trials: int) -> pd.DataFrame:
    required = {"participant_id", "type", "difficulty", "correct", "reaction_time_ms"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.copy()
    for col in ("correct", "reaction_time_ms", "difficulty", "trial_nr"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["correct", "reaction_time_ms", "participant_id"])

    counts   = df.groupby("participant_id")["correct"].count()
    keep     = counts[counts >= min_trials].index
    excluded = counts[counts < min_trials]

    if len(excluded):
        print(f"Excluding {len(excluded)} participant(s) with < {min_trials} trials:")
        for pid, n in excluded.items():
            print(f"  {pid}: {n} trial(s)")

    return df[df["participant_id"].isin(keep)].copy()


# ── STATS HELPERS ──────────────────────────────────────────────────────────────

def pct(x): return x.mean() * 100

def wilson_ci(correct_series):
    """Wilson score 95 % CI for a proportion. Better than normal approx for binary data."""
    n = len(correct_series)
    if n == 0:
        return float("nan"), float("nan")
    p = correct_series.mean()
    z = 1.96
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - margin, centre + margin

def build_mcnemar_pairs(df: pd.DataFrame):
    """
    For each participant, zip their Zahlen and Georgisch trials sorted by trial_nr.
    Returns two arrays of equal length: z_correct, g_correct (both 0/1 integers).

    The experiment interleaves conditions at matched difficulty levels, so ordering
    by trial_nr produces natural within-difficulty pairs.
    """
    z_all, g_all = [], []
    for pid, grp in df.groupby("participant_id"):
        z = grp[grp["type"] == "Zahlen"].sort_values("trial_nr")["correct"].astype(int).tolist()
        g = grp[grp["type"] == "Georgisch"].sort_values("trial_nr")["correct"].astype(int).tolist()
        n_pairs = min(len(z), len(g))
        z_all.extend(z[:n_pairs])
        g_all.extend(g[:n_pairs])
    return np.array(z_all), np.array(g_all)

def mcnemar_test(z: np.ndarray, g: np.ndarray):
    """
    McNemar's exact test for paired binary outcomes.
    b = Zahlen correct & Georgisch wrong  (Zahlen wins)
    c = Zahlen wrong  & Georgisch correct (Georgisch wins)
    H0: b == c  (no condition effect)
    Uses exact two-sided binomial test on discordant pairs.
    Effect size: odds ratio = b/c.
    """
    b = int(((z == 1) & (g == 0)).sum())
    c = int(((z == 0) & (g == 1)).sum())
    n_discord = b + c

    if n_discord == 0:
        return b, c, float("nan"), float("nan")

    # Exact two-sided binomial: under H0, b ~ Bin(b+c, 0.5)
    result = stats.binomtest(b, n_discord, p=0.5, alternative="two-sided")
    p_val  = result.pvalue
    odds_ratio = b / c if c > 0 else float("inf")
    return b, c, odds_ratio, p_val


# ── REPORT ─────────────────────────────────────────────────────────────────────

def fmt_pct(v):  return f"{v:.1f}%"
def fmt_p(p):
    if np.isnan(p): return "n/a"
    if p < 0.001:   return "< 0.001"
    return f"{p:.3f}"

def build_report(df: pd.DataFrame, min_trials: int) -> str:
    lines = []
    def h(level, text): lines.append(f"\n{'#' * level} {text}\n")
    def row(*cols):     lines.append("| " + " | ".join(str(c) for c in cols) + " |")
    def sep(n):         lines.append("|" + "|".join(["---"] * n) + "|")

    n_participants = df["participant_id"].nunique()
    n_trials       = len(df)
    overall_acc    = pct(df["correct"])
    median_rt      = df["reaction_time_ms"].median()

    sources = df["_source"].value_counts().to_dict() if "_source" in df.columns else {}

    lines.append("# Memory Experiment — Analysis Report")
    lines.append(f"\n_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_\n")

    # 1. Overview
    h(2, "1. Dataset Overview")
    row("Metric", "Value")
    sep(2)
    row(f"Participants (≥ {min_trials} trials)", n_participants)
    row("Total trials", n_trials)
    lo, hi = wilson_ci(df["correct"])
    row("Overall accuracy", f"{fmt_pct(overall_acc)} (95 % CI {fmt_pct(lo*100)}–{fmt_pct(hi*100)})")
    row("Median reaction time", f"{median_rt:.0f} ms")
    if sources:
        for src, cnt in sorted(sources.items()):
            row(f"Source: {src}", f"{cnt} trials")

    # 2. Per-participant
    h(2, "2. Per-Participant Descriptives")
    pp = (
        df.groupby("participant_id")["correct"]
        .agg(trials="count", accuracy=pct)
        .reset_index()
        .sort_values("accuracy", ascending=False)
    )
    row("Participant ID", "Trials", "Accuracy")
    sep(3)
    for _, r in pp.iterrows():
        row(r["participant_id"], int(r["trials"]), fmt_pct(r["accuracy"]))

    # 3. By difficulty
    h(2, "3. Accuracy by Difficulty Level")
    row("Level", "Trials", "Accuracy", "95 % CI (Wilson)")
    sep(4)
    for d in sorted(df["difficulty"].dropna().unique()):
        sub = df[df["difficulty"] == d]["correct"]
        lo_d, hi_d = wilson_ci(sub)
        row(int(d), len(sub), fmt_pct(pct(sub)), f"{fmt_pct(lo_d*100)}–{fmt_pct(hi_d*100)}")

    # 4. Zahlen vs Georgisch descriptives
    h(2, "4. Zahlen vs. Georgisch — Descriptives")
    row("Type", "Trials", "Accuracy", "95 % CI (Wilson)", "Median RT (ms)")
    sep(5)
    for label in ["Zahlen", "Georgisch"]:
        sub = df[df["type"] == label]
        lo_t, hi_t = wilson_ci(sub["correct"])
        row(label, len(sub), fmt_pct(pct(sub["correct"])),
            f"{fmt_pct(lo_t*100)}–{fmt_pct(hi_t*100)}",
            f"{sub['reaction_time_ms'].median():.0f}")

    # 5. McNemar's test
    h(2, "5. Zahlen vs. Georgisch — Inferential Statistics")
    lines.append(
        "**Test:** McNemar's exact test on paired binary trial outcomes "
        "(correct/incorrect per trial). Trials are paired within each participant "
        "by matching Zahlen and Georgisch trials in trial-number order, which "
        "aligns them by difficulty level as the experiment interleaves conditions. "
        "Effect size is the odds ratio of discordant pairs (b/c).\n"
    )

    z_arr, g_arr = build_mcnemar_pairs(df)
    n_pairs = len(z_arr)

    if n_pairs < 2:
        lines.append("> Not enough paired trials to run McNemar's test.\n")
    else:
        b, c, odds_ratio, p_val = mcnemar_test(z_arr, g_arr)
        n_discord = b + c

        row("Statistic", "Value")
        sep(2)
        row("Total pairs", n_pairs)
        row("Concordant (both correct)", int(((z_arr==1)&(g_arr==1)).sum()))
        row("Concordant (both wrong)",   int(((z_arr==0)&(g_arr==0)).sum()))
        row("Zahlen ✓ / Georgisch ✗ (b)", b)
        row("Zahlen ✗ / Georgisch ✓ (c)", c)
        row("Discordant pairs (b + c)", n_discord)
        row("Odds ratio (b / c)", f"{odds_ratio:.2f}" if not np.isinf(odds_ratio) else "∞")
        row("p (exact binomial, two-tailed)", fmt_p(p_val))

        sig = (not np.isnan(p_val)) and p_val < 0.05
        lines.append(
            f"\n**Interpretation:** {'Statistically significant' if sig else 'No significant'} "
            f"difference between conditions at α = 0.05 (p = {fmt_p(p_val)}). "
        )
        if n_discord > 0:
            lines.append(
                f"Of {n_discord} trials where conditions disagreed, "
                f"{b} favoured Zahlen and {c} favoured Georgisch. "
            )
        if not np.isinf(odds_ratio) and not np.isnan(odds_ratio) and c > 0:
            direction = "Zahlen" if odds_ratio > 1 else "Georgisch"
            lines.append(f"Odds ratio = {odds_ratio:.2f} ({direction} easier).")
        lines.append("")

    # 6. Difficulty × type cross-tab
    h(2, "6. Accuracy by Difficulty × Type")
    row("Level", "Zahlen", "Georgisch")
    sep(3)
    for d in sorted(df["difficulty"].dropna().unique()):
        cells = []
        for label in ["Zahlen", "Georgisch"]:
            sub = df[(df["difficulty"] == d) & (df["type"] == label)]["correct"]
            cells.append(fmt_pct(pct(sub)) if len(sub) else "—")
        row(int(d), *cells)

    return "\n".join(lines) + "\n"


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    frames = []

    if not args.no_firebase:
        print("Fetching data from Firestore …")
        remote = fetch_firestore("trials")
        print(f"  {len(remote)} documents retrieved.")
        frames.append(remote)

    csv_dir = Path(args.csv_dir)
    if csv_dir.exists():
        print(f"Loading CSVs from {csv_dir}/ …")
        local = load_csv_dir(args.csv_dir)
        if not local.empty:
            frames.append(local)

    if not frames:
        print("No data found. Pass --csv-dir or ensure Firestore is reachable.", file=sys.stderr)
        sys.exit(1)

    if len(frames) == 2:
        df_raw = merge_sources(frames[0], frames[1])
        print(f"  {len(df_raw)} trials after merging (duplicates removed).")
    else:
        df_raw = frames[0]

    df = clean(df_raw, args.min_trials)
    print(f"  {len(df)} trials kept ({df['participant_id'].nunique()} participant(s)).")

    report = build_report(df, args.min_trials)
    out = Path(args.out)
    out.write_text(report, encoding="utf-8")
    print(f"Report written to {out.resolve()}")


if __name__ == "__main__":
    main()
