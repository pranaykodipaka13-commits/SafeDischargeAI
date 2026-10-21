
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
        "liposarcoma", "hypoparathyroidism"
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
        nltk.download("punkt")
        nltk.download("punkt_tab")
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
    sentence_vectors = vectorizer.fit_transform(sentences)

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

        # Give a small preference to shorter sentences so that
        # long, repetitive sentences do not dominate unnecessarily.
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

    for index in ranked_indices:
        is_redundant = False

        for selected in selected_indices:
            if similarity_matrix[index, selected] > 0.80:
                is_redundant = True
                break

        if not is_redundant:
            selected_indices.append(index)

        if len(selected_indices) >= num_sentences:
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
