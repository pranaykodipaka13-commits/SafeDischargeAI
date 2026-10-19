
import re
import nltk
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
        return sent_tokenize(text)


def textrank_summarize(text, num_sentences=5):
    """Select important sentences from a clinical note."""
    sentences = split_sentences(text)

    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    vectorizer = TfidfVectorizer(stop_words="english")
    sentence_vectors = vectorizer.fit_transform(sentences)

    similarity_matrix = cosine_similarity(sentence_vectors)

    scores = similarity_matrix.sum(axis=1).ravel()

    # Select the highest-scoring sentences.
    selected_indices = sorted(
        scores.argsort()[-num_sentences:]
    )

    return " ".join(sentences[i] for i in selected_indices)


if __name__ == "__main__":
    import pandas as pd

    data = pd.read_csv("data/processed/test.csv")
    note = data.iloc[0]["note"]

    summary = textrank_summarize(note, num_sentences=5)

    print("\nORIGINAL NOTE:\n")
    print(note[:1500])

    print("\nGENERATED SUMMARY:\n")
    print(summary)
