
import pandas as pd
import re
from collections import Counter

DATA_PATH = "data/processed/test.csv"

data = pd.read_csv(DATA_PATH)

print("Total test notes:", len(data))
print("Available columns:", list(data.columns))

# Inspect the first five note lengths.
print("\n========== NOTE LENGTHS ==========")
for i, note in enumerate(data["note"].head(5), start=1):
    words = len(str(note).split())
    print(f"Note {i}: {words} words")

# Find frequent words across the test notes.
# This is exploratory text analysis, not clinical entity recognition.
stop_words = {
    "the", "and", "was", "were", "with", "for", "from",
    "that", "this", "patient", "has", "had", "have", "been",
    "are", "is", "in", "on", "to", "of", "a", "an", "as",
    "by", "at", "or", "but", "he", "she", "it", "his",
    "her", "their", "they", "after", "before", "during",
    "not", "no", "also", "which", "who", "when", "then",
}

counter = Counter()

for note in data["note"].dropna():
    words = re.findall(r"\b[a-zA-Z][a-zA-Z-]{2,}\b", note.lower())
    counter.update(
        word for word in words if word not in stop_words
    )

print("\n========== 50 MOST FREQUENT WORDS ==========")
for word, count in counter.most_common(50):
    print(f"{word:<25} {count}")

# Save the frequency list for further inspection.
frequency_df = pd.DataFrame(
    counter.most_common(),
    columns=["Word", "Frequency"],
)
frequency_df.to_csv(
    "data/processed/word_frequencies.csv",
    index=False,
)

print("\nSaved: data/processed/word_frequencies.csv")
