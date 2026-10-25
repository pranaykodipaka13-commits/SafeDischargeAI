"""
Reproducible, versioned evaluation run for SafeDischargeAI.

Each run writes to a NEW folder:
    evaluation/runs/<UTC timestamp>_<git commit>[_dirty]_n<num_sentences>/

It never overwrites an existing run and never writes to data/processed/.

Outputs in the run folder:
    notes.csv               one row per note (summary text + hashes + counts)
    entries.csv             one row per verifier entry, keyed by note_id
    summary_screening.csv   counts of flags (screening statistics ONLY)
    run_manifest.json       code version, input hash, parameters, environment

IMPORTANT: flag counts are screening statistics. They are NOT accuracy,
precision, recall or a measure of clinical safety. Those require independent
human labels.

Usage (from the repository root):
    python -m evaluation.run_evaluation --input data/processed/test.csv
    python -m evaluation.run_evaluation --limit 20      # quick trial
"""
import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from summarizer import split_sentences, textrank_summarize
from verifier import verify_summary

DEFAULT_INPUT = Path("data/processed/test.csv")
DEFAULT_RUNS_DIR = Path("evaluation/runs")

CAVEAT = (
    "Flag counts are rule-based screening statistics. They are not accuracy, "
    "precision, recall, or evidence of clinical safety."
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text, length=16):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:length]


def _git(args, cwd):
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, cwd=cwd, check=True
    )
    return result.stdout


def git_info(cwd="."):
    """Return commit, whether tracked files are modified, and a diff hash."""
    try:
        commit = _git(["rev-parse", "HEAD"], cwd).strip()
        dirty = bool(
            _git(["status", "--porcelain", "--untracked-files=no"], cwd).strip()
        )
        diff_hash = sha256_text(_git(["diff", "HEAD"], cwd), 64) if dirty else None
        return {"commit": commit, "dirty": dirty, "tracked_diff_sha256": diff_hash}
    except Exception:
        return {"commit": "unknown", "dirty": None, "tracked_diff_sha256": None}


def _load_input(input_csv, id_column, note_column, limit):
    df = pd.read_csv(input_csv)

    for column in (id_column, note_column):
        if column not in df.columns:
            raise ValueError(
                f"Column '{column}' not found. Columns: {list(df.columns)}"
            )

    if df[id_column].isna().any():
        raise ValueError(f"Column '{id_column}' contains missing values.")

    if df[id_column].duplicated().any():
        raise ValueError(f"Column '{id_column}' must be unique (stable note IDs).")

    return df.head(limit) if limit else df


def run_evaluation(
    input_csv=DEFAULT_INPUT,
    runs_dir=DEFAULT_RUNS_DIR,
    num_sentences=8,
    limit=None,
    id_column="patient_id",
    note_column="note",
    now=None,
    repo_root=".",
):
    """Run summarizer + verifier on every note and save a versioned run."""
    input_csv = Path(input_csv)
    df = _load_input(input_csv, id_column, note_column, limit)

    git = git_info(repo_root)
    created = now or datetime.now(timezone.utc)
    short = git["commit"][:7] if git["commit"] != "unknown" else "nogit"
    run_name = (
        f"{created.strftime('%Y%m%dT%H%M%SZ')}_{short}"
        f"{'_dirty' if git['dirty'] else ''}_n{num_sentences}"
    )

    run_dir = Path(runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)  # never overwrite a run

    note_rows, entry_rows = [], []

    for _, row in df.iterrows():
        note_id = row[id_column]
        note = str(row[note_column])
        summary = textrank_summarize(note, num_sentences=num_sentences)
        summary_hash = sha256_text(summary)
        result = verify_summary(note, summary)

        missing = (
            int((result["Verification"] == "POSSIBLY MISSING").sum())
            if len(result) else 0
        )

        note_rows.append({
            "note_id": note_id,
            "source_sha": sha256_text(" ".join(note.split()).lower()),
            "summary_sha": summary_hash,
            "n_source_sentences": len(split_sentences(note)),
            "n_summary_sentences": len(split_sentences(summary)),
            "n_entries": len(result),
            "n_possible_missing": missing,
            "summary": summary,
        })

        for _, entry in result.iterrows():
            entry_rows.append({
                "note_id": note_id,
                "summary_sha": summary_hash,
                "Term": entry["Term"],
                "Category": entry["Category"],
                "Source Status": entry["Source Status"],
                "Summary Status": entry["Summary Status"],
                "Verification": entry["Verification"],
                "Priority": entry["Priority"],
            })

    notes = pd.DataFrame(note_rows)
    entries = pd.DataFrame(
        entry_rows,
        columns=[
            "note_id", "summary_sha", "Term", "Category", "Source Status",
            "Summary Status", "Verification", "Priority",
        ],
    )

    screening = pd.DataFrame([
        {"Metric": "Notes evaluated", "Value": len(notes)},
        {"Metric": "Notes with at least one POSSIBLY MISSING flag",
         "Value": int((notes["n_possible_missing"] > 0).sum())},
        {"Metric": "POSSIBLY MISSING flags",
         "Value": int(notes["n_possible_missing"].sum())},
        {"Metric": "Verification entries", "Value": len(entries)},
    ])

    notes.to_csv(run_dir / "notes.csv", index=False)
    entries.to_csv(run_dir / "entries.csv", index=False)
    screening.to_csv(run_dir / "summary_screening.csv", index=False)

    manifest = {
        "run_name": run_name,
        "created_utc": created.isoformat(),
        "git": git,
        "parameters": {
            "num_sentences": num_sentences,
            "limit": limit,
            "id_column": id_column,
            "note_column": note_column,
        },
        "input": {
            "path": str(input_csv),
            "sha256": sha256_file(input_csv),
            "notes_evaluated": len(notes),
        },
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
        },
        "caveat": CAVEAT,
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    parser.add_argument("--num-sentences", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    run_dir = run_evaluation(
        input_csv=args.input,
        runs_dir=args.runs_dir,
        num_sentences=args.num_sentences,
        limit=args.limit,
    )
    print(f"Run saved to: {run_dir}")
    print(CAVEAT)


if __name__ == "__main__":
    main()