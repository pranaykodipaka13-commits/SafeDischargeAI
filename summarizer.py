
import re
import nltk
import pandas as pd

from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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


def textrank_summarize(text, num_sentences=5):
    """Select important, non-redundant sentences from a clinical note."""
    sentences = split_sentences(text)

    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    vectorizer = TfidfVectorizer(stop_words="english")
    sentence_vectors = vectorizer.fit_transform(sentences)

    similarity_matrix = cosine_similarity(sentence_vectors)
    scores = similarity_matrix.sum(axis=1).ravel()

    selected_indices = []

    # Select important sentences while reducing redundancy.
    ranked_indices = scores.argsort()[::-1]

    for index in ranked_indices:
        is_redundant = False

        for selected in selected_indices:
            if similarity_matrix[index, selected] > 0.75:
                is_redundant = True
                break

        if not is_redundant:
            selected_indices.append(index)

        if len(selected_indices) == num_sentences:
            break

    # Restore the original order of the selected sentences.
    selected_indices.sort()

    return " ".join(sentences[i] for i in selected_indices)


if __name__ == "__main__":
    data = pd.read_csv("data/processed/test.csv")

    note = data.iloc[0]["note"]

    summary = textrank_summarize(note, num_sentences=5)

    print("\nORIGINAL NOTE:\n")
    print(note[:1500])

    print("\nGENERATED SUMMARY:\n")
    print(summary)
