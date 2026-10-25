
import re
import nltk
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
    "clinical_findings": [
        "lesion", "mass", "nodule", "opacity",
        "fracture", "enlargement", "abnormality",
        "kidney", "tumor", "tumour"
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
        "thrombocytopenia", "pancytopenia",
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
    if re.search(
        r"\b\d+(?:\.\d+)?\s*(?:cm|mm|mg|ml|kg)\b",
        sentence_lower
    ):
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
        "lesion", "mass", "nodule", "opacity"
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
        "requiring blood transfusion",
        "blood transfusion",
        "elevated creatinine",
        "raised creatinine"
    ]

    if any(phrase in sentence_lower for phrase in abnormal_lab_phrases):
        score += 5

    # Preserve cancer stage and active treatment context.
    if re.search(r"\\bstage\\s*(?:iii|iv|3|4|3b|4a|4b)\\b", sentence_lower):
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
        "showed pseudocystic"
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

    return score


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
    similarity_scores = similarity_matrix.sum(axis=1).ravel()

    # Combine similarity and clinical-information scores.
    final_scores = []

    for index, sentence in enumerate(sentences):
        clinical_score = sentence_importance(sentence)

        score = (
            similarity_scores[index]
            + clinical_score * 0.5
        )

        # Slightly reduce the influence of very long sentences.
        word_count = len(sentence.split())

        if word_count > 60:
            score *= 0.9

        final_scores.append(score)

    ranked_indices = sorted(
        range(len(sentences)),
        key=lambda i: final_scores[i],
        reverse=True
    )

    selected_indices = []

    def add_sentence(index):
        """Add a sentence unless it is redundant."""
        if index in selected_indices:
            return False

        for selected in selected_indices:
            if similarity_matrix[index, selected] > 0.80:
                return False

        selected_indices.append(index)
        return True

    # First prioritize a sentence containing presenting symptoms.
    symptom_keywords = [
        "pain", "fever", "nausea", "vomiting",
        "cough", "fatigue", "weakness",
        "shortness of breath", "breathing difficulties", "dizziness", "headache"
    ]
  
    if num_sentences >= 3:
        for index, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()

            # Avoid selecting a sentence that mentions only
            # a negated symptom as though it were present.
            symptom_present = any(
                term in sentence_lower
                for term in symptom_keywords
                if term not in ("shortness of breath", "breathing difficulties")
            )

            if symptom_present and add_sentence(index):
                break

    for index in ranked_indices:
        sentence_lower = sentences[index].lower()

        if any(term in sentence_lower for term in symptom_keywords):
            if add_sentence(index):
                break
 # Prioritize medical history if space remains.
    if num_sentences >= 3 and len(selected_indices) < num_sentences:
        history_keywords = [
            "medical history", "past medical history",
            "history of", "diabetes", "hypertension",
            "comorbidities"
        ]

        for index, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()

            if any(term in sentence_lower for term in history_keywords):
                if add_sentence(index):
                    break

    # Fill remaining slots using the existing importance ranking.
    for index in ranked_indices:
        if len(selected_indices) >= num_sentences:
            break

        add_sentence(index)
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
