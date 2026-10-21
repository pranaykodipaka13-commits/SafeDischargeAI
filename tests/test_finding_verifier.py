import pytest

from verifier import verify_summary


def finding_row(note, summary):
    df = verify_summary(note, summary)
    rows = df[df["Category"] == "Clinical Findings"]
    assert len(rows) == 1, (
        f"Expected one clinical finding row, got {len(rows)}"
    )
    return rows.iloc[0]


def test_missing_kidney_lesion_is_flagged():
    note = (
        "CT scan showed a 4 cm lesion in the left kidney. "
        "The patient remained stable."
    )
    summary = "The patient remained stable."

    row = finding_row(note, summary)

    assert row["Verification"] == "POSSIBLY MISSING"
    assert row["Priority"] == "High"


def test_preserved_kidney_lesion_is_not_flagged_missing():
    note = "CT scan showed a 4 cm lesion in the left kidney."
    summary = (
        "CT scan showed a 4 cm lesion in the left kidney "
        "requiring further evaluation."
    )

    row = finding_row(note, summary)

    assert row["Verification"] == "TERM FOUND - REVIEW CONTEXT"


def test_missing_lesion_size_is_flagged_for_review():
    note = "CT scan showed a 4 cm lesion in the left kidney."
    summary = "CT scan showed a lesion in the left kidney."

    row = finding_row(note, summary)

    assert row["Verification"] == "POSSIBLY MISSING"
