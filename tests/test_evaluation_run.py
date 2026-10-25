import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from evaluation.run_evaluation import run_evaluation, sha256_file

NOTES = [
    "Patient presented with abdominal pain and fever. CT scan showed a 4 cm "
    "lesion in the left kidney. History of diabetes. He received antibiotics. "
    "Discharged in stable condition.",
    "Patient denied chest pain. No fracture was seen. Biopsy was performed. "
    "Follow-up in two weeks.",
    "Routine visit with no complaints.",
]


@pytest.fixture
def input_csv(tmp_path):
    path = tmp_path / "test.csv"
    pd.DataFrame({"patient_id": [101, 102, 103], "note": NOTES}).to_csv(
        path, index=False
    )
    return path


def _run(input_csv, tmp_path, second=0, **kwargs):
    return run_evaluation(
        input_csv=input_csv,
        runs_dir=tmp_path / "runs",
        now=datetime(2026, 10, 25, 12, 0, second, tzinfo=timezone.utc),
        repo_root=tmp_path,  # not a git repo -> commit "unknown"
        **kwargs,
    )


def test_run_creates_expected_files_and_manifest(input_csv, tmp_path):
    run_dir = _run(input_csv, tmp_path)

    for name in ("notes.csv", "entries.csv", "summary_screening.csv",
                 "run_manifest.json"):
        assert (run_dir / name).exists()

    manifest = json.loads((run_dir / "run_manifest.json").read_text("utf-8"))
    assert manifest["input"]["sha256"] == sha256_file(input_csv)
    assert manifest["input"]["notes_evaluated"] == 3
    assert manifest["parameters"]["num_sentences"] == 8
    assert "not accuracy" in manifest["caveat"]


def test_run_never_overwrites_an_existing_run(input_csv, tmp_path):
    _run(input_csv, tmp_path)

    with pytest.raises(FileExistsError):
        _run(input_csv, tmp_path)  # identical timestamp -> same folder name


def test_separate_runs_are_identical_for_identical_inputs(input_csv, tmp_path):
    first = pd.read_csv(_run(input_csv, tmp_path, second=0) / "notes.csv")
    second = pd.read_csv(_run(input_csv, tmp_path, second=1) / "notes.csv")

    assert first["summary_sha"].tolist() == second["summary_sha"].tolist()


def test_input_file_is_not_modified(input_csv, tmp_path):
    before = sha256_file(input_csv)
    _run(input_csv, tmp_path)

    assert sha256_file(input_csv) == before


def test_entries_are_keyed_by_note_id_and_counts_are_consistent(
    input_csv, tmp_path
):
    run_dir = _run(input_csv, tmp_path)
    notes = pd.read_csv(run_dir / "notes.csv")
    entries = pd.read_csv(run_dir / "entries.csv")

    assert set(entries["note_id"]) <= set(notes["note_id"])
    assert notes["n_entries"].sum() == len(entries)
    assert notes["n_possible_missing"].sum() == int(
        (entries["Verification"] == "POSSIBLY MISSING").sum()
    )


def test_limit_restricts_number_of_notes(input_csv, tmp_path):
    run_dir = _run(input_csv, tmp_path, limit=2)

    assert len(pd.read_csv(run_dir / "notes.csv")) == 2


def test_duplicate_note_ids_are_rejected(tmp_path):
    path = tmp_path / "dup.csv"
    pd.DataFrame({"patient_id": [1, 1], "note": ["a b.", "c d."]}).to_csv(
        path, index=False
    )

    with pytest.raises(ValueError, match="unique"):
        _run(path, tmp_path)


def test_missing_note_column_is_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"patient_id": [1], "text": ["a b."]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="not found"):
        _run(path, tmp_path)