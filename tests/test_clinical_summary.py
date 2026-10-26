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



def test_flags_missing_spinal_canal_stenosis():
    source = (
        "MRI revealed severe spinal cord compression and "
        "spinal canal stenosis."
    )
    summary = "The patient had severe spinal cord compression."

    result = verify_summary(source, summary)

    stenosis = result[
        result["Term"].str.contains(
            "spinal canal stenosis", case=False, na=False
        )
    ]
    assert not stenosis.empty
    assert stenosis["Verification"].str.contains(
        "POSSIBLY MISSING", case=False
    ).any()


def test_flags_missing_tracheal_lumen_measurement():
    source = (
        "CT scan showed compromised tracheal lumen "
        "measuring 6 mm."
    )
    summary = "CT scan showed an abnormality."

    result = verify_summary(source, summary)

    airway = result[
        result["Term"].str.contains(
            "tracheal lumen", case=False, na=False
        )
    ]
    assert not airway.empty
    assert airway["Verification"].str.contains(
        "POSSIBLY MISSING", case=False
    ).any()


def test_flags_missing_intraoperative_bile_duct_complication():
    source = (
        "During surgery, the intrapancreatic bile duct was "
        "inadvertently removed from the main pancreatic duct. "
        "The patient developed postoperative pancreatitis."
    )
    summary = (
        "The patient developed postoperative pancreatitis."
    )

    result = verify_summary(source, summary)

    complication = result[
        result["Term"].str.contains(
            "bile duct", case=False, na=False
        )
    ]
    assert not complication.empty
    assert complication["Verification"].str.contains(
        "POSSIBLY MISSING", case=False
    ).any()

def test_accepts_preserved_spinal_canal_stenosis():
    source = (
        "MRI revealed severe spinal cord compression and "
        "spinal canal stenosis."
    )
    summary = (
        "MRI showed severe spinal cord compression and "
        "spinal canal stenosis."
    )
    result = verify_summary(source, summary)

    stenosis = result[
        result["Term"].str.contains(
            "spinal canal stenosis", case=False, na=False
        )
    ]

    assert not stenosis.empty
    assert not stenosis["Verification"].str.contains(
        "POSSIBLY MISSING", case=False
    ).any()


def test_accepts_preserved_tracheal_lumen_measurement():
    source = (
        "CT scan showed compromised tracheal lumen "
        "measuring 6 mm."
    )
    summary = (
        "CT scan showed a compromised tracheal lumen "
        "measuring 6 mm."
    )
    result = verify_summary(source, summary)

    airway = result[
        result["Term"].str.contains(
            "tracheal lumen", case=False, na=False
        )
    ]

    assert not airway.empty
    assert not airway["Verification"].str.contains(
        "POSSIBLY MISSING", case=False
    ).any()


def test_accepts_preserved_intraoperative_bile_duct_complication():
    source = (
        "During surgery, the intrapancreatic bile duct was "
        "inadvertently removed from the main pancreatic duct. "
        "The patient developed postoperative pancreatitis."
    )
    summary = (
        "During surgery, the intrapancreatic bile duct was "
        "inadvertently removed from the main pancreatic duct. "
        "The patient developed postoperative pancreatitis."
    )
    result = verify_summary(source, summary)

    complication = result[
        result["Term"].str.contains(
            "bile duct", case=False, na=False
        )
    ]

    assert not complication.empty
    assert not complication["Verification"].str.contains(
        "POSSIBLY MISSING", case=False
    ).any()


def test_summary_preserves_primary_gunshot_injury_and_pelvic_fractures():
    note = (
        "Discharge Summary: The patient is a 33-year-old man with a "
        "gunshot injury to the pelvis, resulting in comminuted left "
        "pelvic fractures and an open wound in the left lower quadrant. "
        "The surgery was successful. At 4-month follow-up, the ureter "
        "was completely healed, and the uretero-rectal fistula was resolved. "
        "The patient underwent removal of the nephrostomy tube, and the "
        "double J stent was kept in place to prevent ureteric strictures."
    )

    summary = textrank_summarize(note, num_sentences=5).lower()

    assert "gunshot injury" in summary
    assert "pelvic fractures" in summary
    assert "open wound" in summary


def test_summary_preserves_primary_injury_in_long_trauma_note():
    import pandas as pd

    dataset = pd.read_csv("data/processed/test.csv")
    note = dataset.iloc[177]["note"]

    summary = textrank_summarize(note, num_sentences=8).lower()

    assert "gunshot injury" in summary
    assert "pelvic fractures" in summary
    assert "open wound" in summary

def test_cancer_flag_accepts_adenocarcinoma_in_summary():
    source = (
        "The patient was diagnosed with gallbladder cancer. "
        "Pathology showed moderately differentiated adenocarcinoma."
    )
    summary = (
        "Pathology showed moderately differentiated adenocarcinoma."
    )
    result = verify_summary(source, summary)
    cancer = result[
        (result["Term"].str.lower() == "cancer")
        & (result["Category"] == "Diagnoses")
    ]
    assert not cancer.empty
    assert not cancer["Verification"].str.contains(
        "POSSIBLY MISSING", case=False
    ).any()


def test_infective_endocarditis_covers_infection_flag():
    source = "The patient was diagnosed with infection."
    summary = "The patient was diagnosed with infective endocarditis."
    result = verify_summary(source, summary)
    infection = result[
        (result["Term"].str.lower() == "infection")
        & (result["Category"] == "Diagnoses")
    ]
    assert not infection.empty
    assert not infection["Verification"].str.contains(
        "POSSIBLY MISSING", case=False
    ).any()


def test_future_stent_removal_does_not_trigger_bile_duct_injury():
    source = (
        "A temporary plastic biliary stent was placed into the "
        "common bile duct, to be removed in three months."
    )
    summary = (
        "A temporary plastic biliary stent was placed into the "
        "common bile duct, to be removed in three months."
    )
    result = verify_summary(source, summary)
    complication = result[
        result["Term"].str.contains(
            "bile duct removal or injury complication",
            case=False,
            na=False,
        )
    ]
    assert complication.empty

def test_low_risk_covid_screening_is_not_confirmed_infection():
    source = (
        "The patient was considered at low risk of COVID-19 infection "
        "during screening."
    )
    summary = "The patient underwent routine screening."
    result = verify_summary(source, summary)

    infection = result[
        (result["Term"].str.lower() == "infection")
        & (result["Category"] == "Diagnoses")
    ]

    if not infection.empty:
        assert not (
            infection["Source Status"]
            .astype(str)
            .str.upper()
            .eq("AFFIRMED_OR_MENTIONED")
        ).any()
        assert not infection["Verification"].str.contains(
            "POSSIBLY MISSING", case=False
        ).any()
