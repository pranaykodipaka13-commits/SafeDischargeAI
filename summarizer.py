
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
