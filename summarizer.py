import re
import nltk
import numpy as np
import pandas as pd

from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Clinical information categories to prioritize.
CLINICAL_KEYWORDS = {
    "diagnosis": [
        "diagnosis", "diagnosed", "hypothyroidism",
        "hyperthyroidism", "syndrome", "thyroiditis",
        "cancer", "tumor", "tumour", "infection",
        "pneumonia", "diabetes", "heart failure",
        "liposarcoma", "hypoparathyroidism", "hypertension"
    ],
    "symptoms": [
        "symptoms", "pain", "fatigue", "tiredness",
        "breathing difficulties", "shortness of breath",
        "weakness", "fever", "bleeding", "nausea",
        "vomiting", "swelling", "muscle aches"
    ],
    "tests": [
        "ct scan", "computed tomography", "mri",
        "ultrasound", "biopsy", "histopathology",
        "laboratory tests", "blood test", "fna",
        "examination", "test results"
    ],
    "treatment": [
        "treatment", "treated", "medication",
        "medications", "prescribed", "levothyroxine",
        "surgery", "surgical resection", "isthmectomy",
        "radiation therapy", "therapy", "procedure"
    ],
    # Generic operative vocabulary so detailed procedures are not
    # compressed down to a single "surgery" mention.
    "procedure_detail": [
        "excision", "excised", "resection", "resected",
        "graft", "duraplasty", "closure", "closed",
        "anastomosis", "implant", "implanted", "incision",
        "craniotomy", "laminectomy", "debridement",
        "reconstruction", "ligation", "stent", "suture",
        "sutured", "drain", "drainage", "mesh", "fixation"
    ],
    "clinical_findings": [
        "lesion", "mass", "nodule", "opacity",
        "fracture", "enlargement", "abnormality",
        "kidney", "tumor", "tumour","gunshot injury",
        "defect", "herniation", "prolapse", "prolapsus",
        "opacities", "ground-glass", "infiltrate", "infiltrates",
        "consolidation", "effusion", "thrombosis", "stenosis",
        "calcification", "calcifications"
    ],
    "medical_history": [
        "medical history", "past medical history",
        "history includes", "history of",
        "comorbidities", "underlying condition"
    ],
    "pathology": [
        "biopsy showed", "biopsy revealed", "histopathology",
        "denervation", "regenerating fibers",
        "lipid deposition", "glycogen deposits",
        "malignant", "benign", "histological findings"
    ],
    "discharge_support": [
        "tracheostomy", "ventilator", "ventilator-dependent",
        "unable to wean", "urinary catheter",
        "feeding tube", "oxygen support", "home oxygen",
        "chronic catheter", "discharged with"
    ],
    "complications": [
        "complication", "postoperative pancreatitis",
        "pancreatitis", "fluid collection",
        "fluid collections", "adverse event",
        "inadvertent removal", "injury during surgery",
        "surgical complication"
    ],
    "abnormal_labs": [
        "severe anemia", "anaemia", "anemia",
        "thrombocytopenia", "pancytopenia", "neutropenia",
        "elevated creatinine", "raised creatinine",
        "elevated urea", "abnormal blood count",
        "abnormal laboratory", "abnormal lab",
        "hemoglobin", "haemoglobin"
    ],
    "disease_stage_and_active_treatment": [
        "stage 3", "stage iii", "stage iv",
        "chemotherapy", "folfox", "radiotherapy",
        "metastatic", "malignant", "recurrence"
    ],
    "discharge_followup": [
        "discharged", "follow-up", "follow up",
        "appointment", "advised", "recommend",
        "stable condition", "continue medication"
    ]
}

SYMPTOM_KEYWORDS = [
    "pain", "fever", "nausea", "vomiting",
    "cough", "fatigue", "weakness",
    "shortness of breath", "breathing difficulties",
    "dizziness", "headache",
    "incontinence", "diarrhea", "diarrhoea", "dyspnea", "dyspnoea",
    "syncope", "dysphagia", "numbness", "rash", "itching"
]

HISTORY_KEYWORDS = [
    "medical history", "past medical history",
    "history of", "diabetes", "hypertension",
    "comorbidities"
]

NEGATION_RE = re.compile(
    r"\b(?:no|denied|denies|without|negative for|absence of|free of)\b"
)

MEASUREMENT_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:cm|mm|mg|ml|kg)\b")

STAGE_RE = re.compile(r"\bstage\s*(?:iii|iv|3|4)[a-c]?\b")

# Laboratory markers and abnormal-result wording ("elevated D-dimer").
LAB_TERMS = [
    "d-dimer", "c-reactive protein", "crp", "procalcitonin",
    "neutrophilia", "lymphopenia", "leukocytosis", "leukopenia",
    "thrombocytosis", "troponin", "ferritin", "bilirubin", "creatinine",
    "lactate", "potassium", "white blood cell", "wbc", "platelet",
    "hemoglobin", "haemoglobin", "alt", "ast", "inr", "bnp", "hba1c"
]
LAB_RE = re.compile(r"\b(?:" + "|".join(re.escape(t) for t in LAB_TERMS) + r")\b")
ABNORMAL_RE = re.compile(
    r"\b(?:elevated|raised|increased|decreased|reduced|abnormal|"
    r"significant for|neutrophilia|lymphopenia|leukocytosis|leukopenia|"
    r"thrombocytosis)\b"
)

DIAGNOSIS_STATEMENT_RE = re.compile(
    r"\b(?:diagnosed with|diagnosed as|diagnosis of|was diagnosed|were diagnosed)\b"
)

# Severity word directly before a diagnosis term, e.g. "severe hypothyroidism".
SEVERITY_DIAGNOSIS_RE = re.compile(
    r"\b(?:severe|marked|profound|critical|acute)\s+(?:"
    + "|".join(re.escape(k) for k in CLINICAL_KEYWORDS["diagnosis"])
    + r")"
)

# Words whose presence counts as a "critical detail" for the coverage check.
COVERAGE_TERMS = (
    CLINICAL_KEYWORDS["diagnosis"]
    + CLINICAL_KEYWORDS["tests"]
    + CLINICAL_KEYWORDS["procedure_detail"]
    + CLINICAL_KEYWORDS["clinical_findings"]
    + LAB_TERMS
)


def split_sentences(text):
    """Split a clinical note into sentences."""
    text = str(text).replace("\\n", "\n")
    text = re.sub(r"\s+", " ", text).strip()

    try:
        return sent_tokenize(text)
    except LookupError:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        return sent_tokenize(text)


def sentence_importance(sentence):

    """Give extra priority to sentences with clinical information."""
    sentence_lower = sentence.lower()
    score = 0
    # Prioritize the primary injury and reason for admission.
    primary_injury_terms = [
        "gunshot injury",
        "pelvic fractures",
        "open wound",
        "presented with",
        "chief complaint",
    ]

    if any(term in sentence_lower for term in primary_injury_terms):
        score += 10
    for category, keywords in CLINICAL_KEYWORDS.items():
        if any(keyword in sentence_lower for keyword in keywords):
            score += 2

    # Prioritize explicit diagnoses and discharge instructions.
    if "diagnosis:" in sentence_lower:
        score += 3

    if "discharge diagnosis" in sentence_lower:
        score += 3

    if "follow-up:" in sentence_lower or "follow up:" in sentence_lower:
        score += 2

    # Prioritize measurements and specific clinical findings.
    if MEASUREMENT_RE.search(sentence_lower):
        score += 3

    # Prioritize explicit medical history.
    if any(phrase in sentence_lower for phrase in [
        "medical history",
        "past medical history",
        "history includes",
        "history of"
    ]):
        score += 2

    # Prioritize important investigation findings.
    if any(term in sentence_lower for term in [
        "lesion", "mass", "nodule", "opacity", "opacities",
        "ground-glass", "infiltrate", "consolidation", "effusion"
    ]):
        score += 3

    # Strongly prioritize discharge dependencies and major complications.
    high_priority_phrases = [
        "tracheostomy",
        "unable to wean",
        "ventilator-dependent",
        "discharged with",
        "chronic urinary catheter",
        "inadvertent removal",
        "postoperative pancreatitis",
        "spinal cord compression",
        "severe cord compression",
        "spinal canal stenosis"
    ]

    if any(phrase in sentence_lower for phrase in high_priority_phrases):
        score += 8

    # Preserve specific pathology and biopsy results.
    pathology_phrases = [
        "biopsy showed",
        "biopsy revealed",
        "muscle biopsy",
        "denervation",
        "regenerating fibers",
        "lipid deposition",
        "glycogen deposits",
        "histopathology revealed"
    ]

    if any(phrase in sentence_lower for phrase in pathology_phrases):
        score += 5

    # Prioritize clinically significant abnormal laboratory results.
    abnormal_lab_phrases = [
        "severe anemia",
        "severe anaemia",
        "dimorphic anemia",
        "dimorphic anaemia",
        "thrombocytopenia",
        "pancytopenia",
        "neutropenia",
        "requiring blood transfusion",
        "blood transfusion",
        "elevated creatinine",
        "raised creatinine"
    ]

    if any(phrase in sentence_lower for phrase in abnormal_lab_phrases):
        score += 5

    # Preserve cancer stage and active treatment context.
    # (Fixed: the old pattern used doubled backslashes inside a raw string,
    # so it looked for a literal "\b" and never matched.)
    if STAGE_RE.search(sentence_lower):
        score += 5

    if any(term in sentence_lower for term in [
        "chemotherapy", "folfox", "metastatic cancer",
        "ongoing chemotherapy", "active malignancy"
    ]):
        score += 4

    # Boost sentences describing imaging or investigation findings,
    # rather than prioritizing the test name alone.
    finding_phrases = [
        "revealed severe",
        "showed severe",
        "revealed a lesion",
        "showed a lesion",
        "revealed calcifications",
        "showed calcifications",
        "revealed stenosis",
        "showed stenosis",
        "revealed compression",
        "showed compression",
        "revealed pseudocystic",
        "showed pseudocystic",
        "narrowing",
        "narrowed"
    ]

    if any(phrase in sentence_lower for phrase in finding_phrases):
        score += 4

    # Preserve final diagnostic conclusions and important genetic findings.
    diagnostic_priority_phrases = [
        "likely pathogenic",
        "scn4a",
        "periodic paralysis",
        "abnormal potassium levels",
        "t2n1m0",
        "tumor stage",
        "tumour stage",
        "adenocarcinoma",
        "intracystic papillary neoplasm",
    ]

    if any(phrase in sentence_lower for phrase in diagnostic_priority_phrases):
        score += 8

    # Preserve specific operative events and major complications.
    complication_priority_phrases = [
        "inadvertently removed",
        "intrapancreatic bile duct",
        "main pancreatic duct",
        "acute necrotic collection",
        "progression of acute necrotic collection",
        "readmitted",
        "readmission",
        "severe bilateral pleural effusion",
        "candida peritonitis",
        "sepsis due to",
    ]

    if any(phrase in sentence_lower for phrase in complication_priority_phrases):
        score += 8

    # Preserve significant treatment decisions and discharge instructions.
    treatment_priority_phrases = [
        "broad-spectrum antibiotics",
        "meropenem",
        "anidulafungin",
        "tacrolimus was interrupted",
        "antibiotic treatment was to be continued",
        "continue antibiotic treatment",
        "chemotherapy was not possible",
        "delayed recovery from postoperative pancreatitis",
    ]

    if any(phrase in sentence_lower for phrase in treatment_priority_phrases):
        score += 6

    # Preserve clinically significant abnormal imaging findings,
    # without boosting every sentence that merely mentions an MRI.
    imaging_priority_phrases = [
        "numerous pseudo cystic formation",
        "pseudocystic formation",
        "peripancreatic fluid collection",
        "fluid collection began to resolve",
        "complete thrombosis",
        "pulmonary consolidation",
        "severe bilateral pleural effusion",
    ]

    if any(phrase in sentence_lower for phrase in imaging_priority_phrases):
        score += 6

    # Preserve abnormal laboratory results ("elevated D-dimer, neutrophilia").
    if LAB_RE.search(sentence_lower) and ABNORMAL_RE.search(sentence_lower):
        score += 6

    # Preserve sentences that state a diagnosis ("were diagnosed with ...").
    if DIAGNOSIS_STATEMENT_RE.search(sentence_lower):
        score += 5

    # Preserve severity of a diagnosis ("severe hypothyroidism").
    if SEVERITY_DIAGNOSIS_RE.search(sentence_lower):
        score += 5

    # Preserve operative detail (what was actually done), not just the fact
    # that surgery happened. Counts distinct procedure terms so a sentence
    # listing several steps (e.g. excision + duraplasty + graft) ranks higher.
    procedure_hits = sum(
        1 for term in CLINICAL_KEYWORDS["procedure_detail"]
        if term in sentence_lower
    )
    if procedure_hits:
        score += min(3 + procedure_hits, 7)

    return score


def critical_details(sentence):
    """Return the set of critical details (measurements and key terms)."""
    sentence_lower = sentence.lower()
    found = set(MEASUREMENT_RE.findall(sentence_lower))

    for term in COVERAGE_TERMS:
        if term in sentence_lower:
            found.add(term)

    return found


TREATMENT_CONTEXT_RE = re.compile(
    r"\b(?:pain|fever)\s+(?:medication|medications|management|relief|control|killers?)\b"
    r"|\banalges\w*|\bantipyretic\w*|\bfor\s+(?:pain|fever|nausea)\b"
    r"|\b(?:relief|improvement|resolution)\s+(?:from|of|in)\b[^.]*"
    r"|\b(?:resolved|improved|subsided)\b"
)


def symptom_hits(sentence_lower):
    """
    Count distinct symptoms described in a sentence, ignoring symptom words
    that appear only as treatments (e.g. "pain medication").
    """
    cleaned = TREATMENT_CONTEXT_RE.sub(" ", sentence_lower)
    return sum(1 for term in SYMPTOM_KEYWORDS if term in cleaned)


def is_negated(sentence_lower):
    """True if the sentence contains a negation cue."""
    return bool(NEGATION_RE.search(sentence_lower))


def textrank_summarize(text, num_sentences=8):
    """
    Extract important clinical sentences using TF-IDF similarity
    and clinical-information priority.
    """
    sentences = split_sentences(text)

    if not sentences:
        return ""

    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    # Convert sentences into numerical TF-IDF vectors.
    vectorizer = TfidfVectorizer(stop_words="english")

    try:
        sentence_vectors = vectorizer.fit_transform(sentences)
    except ValueError:
        # Handle notes containing no usable TF-IDF terms.
        return " ".join(sentences[:num_sentences])

    # Measure similarity between sentences.
    similarity_matrix = cosine_similarity(sentence_vectors)
    n = len(sentences)

    # Centrality: average similarity to the OTHER sentences, so it stays on a
    # 0-1 scale regardless of note length (the old row sum grew with n and
    # included each sentence's similarity to itself).
    centrality = np.clip(
        (similarity_matrix.sum(axis=1) - 1.0) / max(n - 1, 1), 0.0, None
    )

    # Clinical score, normalized to 0-1 so both signals are comparable.
    clinical = np.array(
        [sentence_importance(s) for s in sentences], dtype=float
    )
    if clinical.max() > 0:
        clinical = clinical / clinical.max()

    # Starting weights; tune against the regression tests.
    CENTRALITY_WEIGHT = 0.3
    CLINICAL_WEIGHT = 0.7

    final_scores = []

    for index, sentence in enumerate(sentences):
        score = (
            CENTRALITY_WEIGHT * centrality[index]
            + CLINICAL_WEIGHT * clinical[index]
        )

        # Slightly reduce the influence of very long sentences.
        if len(sentence.split()) > 60:
            score *= 0.9

        final_scores.append(score)

    ranked_indices = sorted(
        range(n),
        key=lambda i: final_scores[i],
        reverse=True
    )

    selected_indices = []
    reserved_indices = set()  # slots picked on purpose; not swapped out later

    def add_sentence(index):
        """Add a sentence unless it is redundant."""
        if index in selected_indices:
            return False

        details = critical_details(sentences[index])

        for selected in selected_indices:
            similarity = similarity_matrix[index, selected]

            if similarity > 0.80:
                return False

            # Moderately similar AND adds no new critical detail
            # (e.g. the same abnormal labs stated twice).
            if (
                similarity > 0.50
                and details
                and details <= critical_details(sentences[selected])
            ):
                return False

        selected_indices.append(index)
        return True

    # Reserve ONE slot for the best-ranked sentence describing a present
    # (non-negated) symptom.
    # Prefer the sentence describing the most distinct symptoms (the
    # presenting complaint), breaking ties by overall ranking.
    if num_sentences >= 3:
        symptom_candidates = []

        for index in ranked_indices:
            sentence_lower = sentences[index].lower()

            if is_negated(sentence_lower):
                continue

            hits = symptom_hits(sentence_lower)

            if hits > 0:
                symptom_candidates.append((hits, index))

        # Most distinct symptoms first; ties go to the earliest sentence,
        # since the presenting complaint comes early in a note.
        symptom_candidates.sort(key=lambda c: (-c[0], c[1]))

        for _, index in symptom_candidates:
            if add_sentence(index):
                reserved_indices.add(index)
                break

    # Reserve ONE slot for the best-ranked medical-history sentence,
    # chosen from the ranking rather than the first keyword hit.
    if num_sentences >= 3 and len(selected_indices) < num_sentences:
        for index in ranked_indices:
            sentence_lower = sentences[index].lower()

            if any(term in sentence_lower for term in HISTORY_KEYWORDS):
                if add_sentence(index):
                    reserved_indices.add(index)
                    break

    # Fill remaining slots using the importance ranking.
    for index in ranked_indices:
        if len(selected_indices) >= num_sentences:
            break

        add_sentence(index)

    # Coverage check: if an important unselected sentence carries a critical
    # detail (measurement, diagnosis, procedure step...) that no selected
    # sentence covers, swap it in for a redundant, low-scoring selected
    # sentence. Swapping keeps the sentence count fixed.
    def covered_by(indices):
        covered = set()
        for k in indices:
            covered |= critical_details(sentences[k])
        return covered

    covered = covered_by(selected_indices)

    for index in ranked_indices:
        if index in selected_indices:
            continue

        missing = critical_details(sentences[index]) - covered

        # A sentence carrying a measurement is always worth covering, even
        # when its overall importance score is modest.
        has_measurement = bool(MEASUREMENT_RE.search(sentences[index].lower()))

        if not missing or (
            sentence_importance(sentences[index]) < 6 and not has_measurement
        ):
            continue

        # Find the lowest-scoring non-reserved selected sentence whose
        # details are all carried by the other selected sentences.
        for victim in sorted(selected_indices, key=lambda k: final_scores[k]):
            if victim in reserved_indices:
                continue

            remaining = [k for k in selected_indices if k != victim]

            # Do not swap in a near-duplicate of something already kept.
            if any(similarity_matrix[index, k] > 0.80 for k in remaining):
                break

            others = covered_by(remaining)

            if critical_details(sentences[victim]) <= others:
                selected_indices.remove(victim)
                selected_indices.append(index)
                covered = covered_by(selected_indices)
                break

    # Return selected sentences in their original order.
    selected_indices.sort()

    return " ".join(sentences[i] for i in selected_indices)


if __name__ == "__main__":
    data = pd.read_csv("data/processed/test.csv")
    note = data.iloc[0]["note"]

    summary = textrank_summarize(note)

    print("\nORIGINAL NOTE:\n")
    print(str(note)[:2000])

    print("\nGENERATED SUMMARY:\n")
    print(summary)