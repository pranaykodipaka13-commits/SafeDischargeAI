
from verifier import verify_summary


def tenderness_row(note, summary):
    df = verify_summary(note, summary)

    rows = df[
        (df["Category"] == "Clinical Findings")
        & (df["Term"] == "left abdominal tenderness")
    ]

    assert len(rows) == 1
    return rows.iloc[0]


def test_location_specific_tenderness_is_preserved():
    note = (
        "Physical examination revealed tenderness "
        "in the left abdominal region."
    )
    summary = "Left abdominal tenderness was noted."

    row = tenderness_row(note, summary)

    assert row["Verification"] == "TERM FOUND - REVIEW CONTEXT"


def test_missing_tenderness_location_is_flagged():
    note = (
        "Physical examination revealed tenderness "
        "in the left abdominal region."
    )
    summary = "Tenderness was noted."

    row = tenderness_row(note, summary)

    assert row["Verification"] == "POSSIBLY MISSING"
    assert row["Priority"] == "High"


def test_opposite_tenderness_location_is_flagged():
    note = (
        "Physical examination revealed tenderness "
        "in the left abdominal region."
    )
    summary = "Right abdominal tenderness was noted."

    row = tenderness_row(note, summary)

    assert row["Verification"] == "CHECK STATUS MISMATCH"
