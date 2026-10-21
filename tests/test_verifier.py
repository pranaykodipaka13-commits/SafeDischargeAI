
import pytest

from verifier import (
    term_pattern,
    find_term_status,
    find_occurrence_status,
    find_source_mentions,
    verify_summary,
)

AFFIRMED = "AFFIRMED_OR_MENTIONED"


def matches(term, text):
    return term_pattern(term).search(text) is not None


def label_for(note, summary, term):
    df = verify_summary(note, summary)
    row = df[df["Term"] == term]
    assert len(row) == 1, (
        f"expected exactly one row for {term!r}, got {len(row)}"
    )
    return row.iloc[0]["Verification"]


# ---------------------------------------------------------------
# 1. Term matching
# ---------------------------------------------------------------
@pytest.mark.parametrize(
    "term,text,expected",
    [
        ("headache", "severe headache", True),
        ("headache", "recurrent headaches", True),
        ("pain", "severe pain", True),
        ("pain", "chronic pains", True),
        ("pain", "painless", False),
        ("pain", "Spain", False),
        ("pain", "paint", False),
        ("fracture", "mild fractures", True),
        ("MRI", "follow-up MRIs", True),
        ("MRI", "mri of the brain", True),
        ("biopsy", "two biopsies were taken", True),
        ("surgery", "previous surgeries", True),
        ("cough", "persistent coughs", True),
        ("CT scan", "a CT scan was done", True),
        ("CT scan", "a CT scanner was used", False),
        ("heart failure", "history of heart failure", True),
        ("breathing difficulties", "breathing difficulties noted", True),
        ("diabetes", "diabetes mellitus", True),
    ],
)
def test_term_matching(term, text, expected):
    assert matches(term, text) is expected


# ---------------------------------------------------------------
# 2. Multiword term robustness
# ---------------------------------------------------------------
@pytest.mark.parametrize(
    "term,text",
    [
        ("CT scan", "two CT scans were done"),
        ("saline infusion", "repeated saline infusions"),
        ("heart failure", "signs of heart  failure"),
        ("heart failure", "signs of heart\nfailure"),
    ],
)
def test_multiword_robustness(term, text):
    assert matches(term, text) is True


# ---------------------------------------------------------------
# 3. Status: negation, uncertainty, affirmation
# ---------------------------------------------------------------
@pytest.mark.parametrize(
    "sentence,term,category,expected",
    [
        ("No evidence of infection.", "infection", "Diagnoses", "NEGATED"),
        ("The patient denies pain.", "pain", "Symptoms", "NEGATED"),
        ("No fracture was identified.", "fracture", "Diagnoses", "NEGATED"),
        ("Possible pneumonia.", "pneumonia", "Diagnoses", "UNCERTAIN"),
        ("The patient has pneumonia.", "pneumonia", "Diagnoses", AFFIRMED),
        ("The patient is stable.", "pain", "Symptoms", "NOT_FOUND"),
        ("No surgery was performed.", "surgery", "Procedures", "MENTIONED"),
        ("She takes insulin daily.", "insulin", "Medications", "MENTIONED"),
    ],
)
def test_status_basic(sentence, term, category, expected):
    assert find_term_status(sentence, term, category) == expected


@pytest.mark.parametrize(
    "sentence,term,expected",
    [
        ("The patient denied pain.", "pain", "NEGATED"),
        ("The patient is pain-free.", "pain", "NEGATED"),
        ("No fever, but pneumonia was confirmed.", "pneumonia", AFFIRMED),
        ("Pneumonia was ruled out.", "pneumonia", "NEGATED"),
    ],
)
def test_status_known_defects(sentence, term, expected):
    assert find_term_status(sentence, term, "Diagnoses") == expected


# ---------------------------------------------------------------
# 4. Occurrence-level behavior and overlap suppression
# ---------------------------------------------------------------
def test_second_occurrence_has_own_status():
    note = "No pain at rest, but severe pain on exertion."
    start = note.rfind("pain")
    assert (
        find_occurrence_status(note, "pain", start, "Symptoms")
        == AFFIRMED
    )


def test_longer_phrase_suppresses_shorter_in_source():
    mentions = find_source_mentions("She had abdominal pain.")
    assert [m["Term"] for m in mentions] == ["abdominal pain"]


# ---------------------------------------------------------------
# 5. Summary verification: labels and priorities
# ---------------------------------------------------------------
def test_possibly_missing_high_priority():
    df = verify_summary(
        "The patient has pneumonia.",
        "The patient is stable.",
    )
    row = df[df["Term"] == "pneumonia"].iloc[0]
    assert row["Verification"] == "POSSIBLY MISSING"
    assert row["Priority"] == "High"


def test_negated_source_absent_from_summary():
    assert (
        label_for("No evidence of infection.", "Stable.", "infection")
        == "NEGATED IN SOURCE - REVIEW"
    )


def test_uncertain_source_absent_from_summary():
    assert (
        label_for("Possible pneumonia.", "Stable.", "pneumonia")
        == "UNCERTAIN IN SOURCE - REVIEW"
    )


def test_status_mismatch_negated_source_affirmed_summary():
    assert (
        label_for("No fever.", "Fever was noted.", "fever")
        == "CHECK STATUS MISMATCH"
    )


def test_status_mismatch_affirmed_source_negated_summary():
    assert (
        label_for(
            "The patient has pneumonia.",
            "No pneumonia was seen.",
            "pneumonia",
        )
        == "CHECK STATUS MISMATCH"
    )


def test_term_found_when_same_status():
    assert (
        label_for(
            "The patient has pneumonia.",
            "The patient has pneumonia.",
            "pneumonia",
        )
        == "TERM FOUND - REVIEW CONTEXT"
    )


def test_plural_in_summary_counts_as_found():
    assert (
        label_for(
            "She has a fracture.",
            "She has fractures.",
            "fracture",
        )
        == "TERM FOUND - REVIEW CONTEXT"
    )


def test_painless_does_not_satisfy_pain():
    assert (
        label_for(
            "The pain was severe.",
            "The patient was painless.",
            "pain",
        )
        == "POSSIBLY MISSING"
    )


def test_no_configured_terms_gives_empty_frame():
    df = verify_summary("The patient is well.", "Well.")
    assert len(df) == 0
    assert "Verification" in df.columns


# ---------------------------------------------------------------
# 6. Summary verification: occurrence-level behavior
# ---------------------------------------------------------------
def test_affirmed_later_occurrence_is_found_in_summary():
    note = "No fever on admission. Fever developed on day three."
    summary = "Fever developed on day three."
    assert (
        label_for(note, summary, "fever")
        == "TERM FOUND - REVIEW CONTEXT"
    )


# This must pass: the word "pain" inside "abdominal pain"
# must not automatically satisfy a separate standalone "pain"
# term when the source also mentions chest pain.
def test_pain_not_satisfied_by_abdominal_pain_in_summary():
    note = "She reported chest pain. She also had abdominal pain."
    summary = "She also had abdominal pain."
    assert label_for(note, summary, "pain") == "POSSIBLY MISSING"


# Both spellings should be treated as equivalent.
def test_anemia_anaemia_variants():
    assert (
        label_for("She has anemia.", "She has anaemia.", "anemia")
        == "TERM FOUND - REVIEW CONTEXT"
    )

def test_pain_is_satisfied_by_neck_pain_in_summary():
    note = "The patient reported neck pain."
    summary = "The patient continued to report neck pain."

    assert label_for(note, summary, "pain") == "TERM FOUND - REVIEW CONTEXT"