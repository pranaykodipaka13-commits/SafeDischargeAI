import pytest

from summarizer import textrank_summarize
from verifier import verify_summary


CLINICAL_NOTE = (
    "A 58-year-old patient presented with severe abdominal pain, "
    "fever of 38.5 C, nausea, and vomiting for two days. "
    "Medical history includes type 2 diabetes mellitus and hypertension. "
    "CT scan of the abdomen revealed a 4 cm lesion in the left kidney. "
    "The patient had no breathing difficulties. "
    "The patient received intravenous fluids, pain medication, and antibiotics."
)


def test_three_sentence_summary_preserves_symptoms_and_history():
    summary = textrank_summarize(CLINICAL_NOTE, num_sentences=3)

    assert len(summary.split(". ")) <= 3
    assert "abdominal pain" in summary.lower()
    assert "diabetes" in summary.lower()
    assert "hypertension" in summary.lower()


def test_verifier_flags_omitted_kidney_lesion():
    summary = (
        "A 58-year-old patient presented with severe abdominal pain, "
        "fever of 38.5 C, nausea, and vomiting for two days. "
        "Medical history includes type 2 diabetes mellitus and hypertension. "
        "The patient received intravenous fluids, pain medication, and antibiotics."
    )

    result = verify_summary(CLINICAL_NOTE, summary)

    assert "4 cm lesion in the left kidney" in result["Term"].values
    finding = result[
        result["Term"] == "4 cm lesion in the left kidney"
    ]

    assert finding["Verification"].str.contains(
        "POSSIBLY MISSING", case=False
    ).any()


LONG_NOTE = (
    "A 62-year-old man presented with abdominal pain and headache for five days. "
    "His past medical history includes hypertension and diabetes. "
    "He denied fever or cough. "
    "Laboratory tests showed hemoglobin of 5.1 g/dL consistent with severe anemia. "
    "CT scan of the neck revealed severe tracheal narrowing. "
    "MRI of the spine showed spinal cord compression with spinal canal stenosis. "
    "Biopsy of the mass demonstrated abnormal cells suspicious for malignancy. "
    "A 4 cm lesion was seen in the left kidney. "
    "He was started on antibiotics and insulin. "
    "He underwent surgery on day three. "
    "Postoperative course was uneventful. "
    "Blood pressure was stable. "
    "The patient was seen by physiotherapy. "
    "Diet was advanced as tolerated. "
    "He was discharged in stable condition. "
    "Follow-up in two weeks was advised. "
)


def _sentence_count(text):
    from summarizer import split_sentences

    return len(split_sentences(text))


@pytest.mark.parametrize("requested", [1, 2, 3, 5, 8, 12])
def test_summary_returns_requested_number_of_sentences(requested):
    summary = textrank_summarize(LONG_NOTE, num_sentences=requested)

    assert _sentence_count(summary) == requested


def test_default_summary_uses_eight_sentences_for_long_note():
    summary = textrank_summarize(LONG_NOTE)

    assert _sentence_count(summary) == 8


def test_summary_sentences_come_from_source_in_original_order():
    from summarizer import split_sentences

    source = split_sentences(LONG_NOTE)
    chosen = split_sentences(textrank_summarize(LONG_NOTE, num_sentences=8))

    assert all(sentence in source for sentence in chosen)
    positions = [source.index(sentence) for sentence in chosen]
    assert positions == sorted(positions)
    assert len(set(positions)) == len(positions)


def test_larger_budget_retains_measured_finding_dropped_at_small_budget():
    summary = textrank_summarize(LONG_NOTE, num_sentences=8)

    assert "4 cm lesion" in summary
    flags = verify_summary(LONG_NOTE, summary)
    lesion = flags[flags["Category"] == "Clinical Findings"]
    assert not lesion["Verification"].str.contains("POSSIBLY MISSING").any()


def test_short_note_is_returned_whole():
    note = "Patient presented with fever. Discharged in stable condition."

    assert textrank_summarize(note, num_sentences=8) == note