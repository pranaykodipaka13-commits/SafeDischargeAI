
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
