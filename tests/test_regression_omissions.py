"""Regression tests for real clinical omissions.

Copy to tests/ in SafeDischargeAI and run:
    python -m pytest tests/test_regression_omissions.py -v
"""
import pytest
from nltk.tokenize import sent_tokenize

from summarizer import textrank_summarize

ID3_NOTE = """Discharge Summary:
Hospital Course:
The patient, a 32-year-old male, with a diagnosis of hypothyroidism was admitted with extreme tiredness, concentration difficulties, and muscle aches. He was found to have severe hypothyroidism with enlarged thyroid lobes and isthmus with an irregular structure. The patient was started on a medication of levothyroxine that was later increased when he presented after five months with further enlargement of the right thyroid lobe. The patient continued with the medication for a year with no significant changes in the thyroid ultrasound or FNA tests. However, the patient presented with neck area pain, breathing difficulties, and tender thyroid. Laboratory tests revealed hypoparathyroidism and hyperphosphataemia. The CT scan showed enlarged thyroid lobes and isthmus with a compromised tracheal lumen of 6 mm. The patient was treated with betamethasone, calcium gluconate bolus injections, saline infusion, and alphacalcidol. Subacute isthmectomy was performed, and histopathology showed extensive inflammation, fibrosis, and extra thyroidal engagement consistent with RT, but no sign of IgG4-related disease. CT scan of the thorax and MRI of the abdomen showed no abnormalities, and S-IgG4 was negative.
Diagnosis:
Chronic lymphocytic thyroiditis, hypothyroidism, hypoparathyroidism, and hyperphosphataemia.
Treatment:
Betamethasone, calcium gluconate bolus injections, saline infusion, alphacalcidol, levothyroxine, and calcium carbonate.
Risk Factors:
None.
Follow-up:
The patient was discharged in a stable condition and was advised to continue with the prescribed medications. Follow-up appointments were scheduled at the clinic.
"""


def test_id3_keeps_severe_hypothyroidism_and_thyroid_enlargement():
    summary = textrank_summarize(ID3_NOTE).lower()
    assert "severe hypothyroidism" in summary
    assert "enlarged thyroid lobes" in summary


def test_id3_keeps_tracheal_narrowing_measurement():
    summary = textrank_summarize(ID3_NOTE).lower()
    assert "6 mm" in summary


def test_id3_keeps_presenting_symptoms():
    summary = textrank_summarize(ID3_NOTE).lower()
    assert "tiredness" in summary or "neck area pain" in summary


@pytest.mark.parametrize("n", [3, 5, 8])
def test_id3_sentence_count_is_preserved(n):
    summary = textrank_summarize(ID3_NOTE, num_sentences=n)
    assert len(sent_tokenize(summary)) == n


ID31_NOTE = """Discharge Summary

Patient Name: [REDACTED]

Medical Record Number: [REDACTED]

Admission Date: [REDACTED]
Discharge Date: [REDACTED]

Admitting Diagnosis: Chronic otitis media with cholesteatoma, status post-radical mastoidectomy

Brief Hospital Course:

The patient was admitted to our clinic with complaints of right ear discharge, hearing impairment, occasional dizziness, and pain in the face and ear. The patient had previously undergone radical mastoidectomy for chronic otitis media with cholesteatoma at an external facility. About two months postoperatively, a mass was observed at the entrance of the right external auditory canal. The patient visited our clinic about ten months postoperatively, and a mass-like lesion was found at the posterosuperior wall of the right external auditory canal entrance.

Temporal bone computed tomography and cranial contrast MRI were performed. A defect measuring 13mm at the tegmen of the right temporal bone and a 12mm defect at the lateral segment of the right tegmen tympani were identified through CT and MRI scans, respectively. A nodular signal about 15 x 13 x 10 mm that was isointense to the brain parenchyma in all sequences was detected filling the right middle ear and proximal segment lodge of the external ear canal. The mass was determined to be dural prolapsus and focal herniation of brain parenchyma. The neurosurgery clinic was consulted, and surgery was scheduled. The operation was accompanied by neurosurgeons. After ensuring the operation site through a transmastoid approach under general anaesthesia, the patient was referred to the neurosurgery team.

The neurosurgery team excised the herniated fibrotic glial tissue, which extended to the external ear canal by using bipolar cautery. They performed duraplasty with galea after determining bone borders. A barrier was created with fibrin glue. Otorhinolaryngology team was consulted once more due to the absence of perioperative cerebrospinal fluid (CSF). Grafts were taken from conchal cartilage and temporalis muscle fascia. The cavity was obliterated with cartilage grafts, and a flat plane was created by laying the temporalis muscle fascia graft on the cartilage grafts. Durability was improved with fibrin glue.

No complications such as postoperative otorrhoea, meningitis, or epilepsy were reported. The patient had no complaints during the follow-up visits at weeks one and months one, three, six, and fifteen postoperatively. There was no pathology at the herniated segment through the defective area.

Discharge Diagnosis: Dural prolapsus and focal herniation of brain parenchyma in the right ear canal

Discharge Condition: Stable

Discharge Medications: N/A

Discharge Instructions: The patient should be mindful of ear hygiene and avoid any activities that could cause ear trauma. The patient should inform their physician if any ear discharge, pain, or dizziness occurs.

Follow-up Appointments: The patient should follow-up with their physician as directed.
"""

ID8_NOTE = """Hospital Course Summary:

The 14-year-old female patient presented with complaints of continuous lower abdominal pain for four months. Physical examination revealed normal external genitalia and clinical abdominal examination was unremarkable. Hemogram and other biochemical parameters were within normal range. Trans-abdominal ultrasound revealed two uterine horns and minimal endometrial fluid collection in both horns. The right horn was seen to communicate caudally with a large, ovoid fluid collection with internal echoes. The left kidney was present while the right kidney was absent. Translabial ultrasound revealed the lower limit of pelvic fluid collection, which was seen to end 1.5 cm above introitus, with findings pointing towards vaginal location of the collection. MRI confirmed the diagnosis of OHVIRA syndrome. Local examination under anesthesia revealed a bulge in the medial wall of the patent left hemivagina caused by the distended right hemivagina.
"""


def test_id31_keeps_operative_details():
    summary = textrank_summarize(ID31_NOTE).lower()
    assert "excised" in summary
    assert "duraplasty" in summary
    assert "graft" in summary


def test_id31_keeps_imaging_defect_measurements():
    summary = textrank_summarize(ID31_NOTE).lower()
    assert "13mm" in summary
    assert "12mm" in summary
    assert "15 x 13 x 10 mm" in summary


def test_id31_keeps_final_diagnosis():
    summary = textrank_summarize(ID31_NOTE).lower()
    assert "dural prolapsus" in summary


@pytest.mark.parametrize("n", [5, 8])
def test_id31_sentence_count_is_preserved(n):
    summary = textrank_summarize(ID31_NOTE, num_sentences=n)
    assert len(sent_tokenize(summary)) == n


@pytest.mark.parametrize("n", [5, 8])
def test_id8_keeps_four_month_abdominal_pain(n):
    # Guard: ID 8 was a mislabeled review item, not a real omission.
    summary = textrank_summarize(ID8_NOTE, num_sentences=n).lower()
    assert "lower abdominal pain for four months" in summary


# ---- Findings from the 500-note diff review -------------------------------
from summarizer import sentence_importance, symptom_hits  # noqa: E402


def test_abnormal_lab_sentence_is_high_priority():
    # Row 212: the elevated D-dimer / neutrophilia sentence was dropped.
    lab = ("Laboratory tests were significant for elevated D-dimer, white "
           "blood cell count with neutrophilia and lymphopenia, C-reactive "
           "protein, and procalcitonin.")
    assert sentence_importance(lab) >= 8


def test_normal_lab_mention_is_not_boosted_as_abnormal():
    normal = "Laboratory tests, including creatinine, were within normal limits."
    assert sentence_importance(normal) < 8


def test_symptom_slot_ignores_outcomes_and_treatments():
    # Row 38: "relief from pain" is an outcome, not a presenting symptom.
    assert symptom_hits("15 days after the procedure, the patient reported "
                        "relief from pain in the face.") == 0
    assert symptom_hits("The patient received pain medication.") == 0
    assert symptom_hits("Chief complaint: pain and torpor of the tongue.") == 1


def test_bladder_incontinence_counts_as_symptom():
    # Row 22: bladder incontinence was lost from the summary.
    assert symptom_hits("The patient presented with low back pain, lower limb "
                        "weakness, and bladder incontinence.") == 3


def test_ground_glass_ct_finding_is_prioritized():
    # Row 7: "opacities" (plural) never matched the "opacity" keyword.
    ct = ("On the 8th day of hospitalization, the native thoraco-pulmonary CT "
          "examination showed unsystematized central and peripheral pulmonary "
          "infiltrates and ground-glass opacities with approximately 50-60% "
          "degree of damage.")
    discharge = ("The patient was discharged the next day with the following "
                 "recommendations: continuing voriconazole therapy for 5 weeks "
                 "more and maintaining physical distance.")
    assert sentence_importance(ct) > sentence_importance(discharge) + 2


def test_diagnosis_statements_are_prioritized():
    # Rows 13 and 25: "diagnosed with ..." sentences were being dropped.
    dx = ("The patient's mother and daughter were also diagnosed with "
          "long QT syndrome.")
    assert sentence_importance(dx) >= 7
    neu = ("The patient was admitted to the intensive care unit with a "
           "diagnosis of high-risk febrile neutropenia.")
    assert sentence_importance(neu) >= 8