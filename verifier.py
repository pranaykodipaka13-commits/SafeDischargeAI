
import re
import pandas as pd
from summarizer import textrank_summarize

CLINICAL_TERMS = {
    "Diagnoses": [
        "hypothyroidism",
        "hyperthyroidism",
        "hypoparathyroidism",
        "hyperparathyroidism",
        "hyperphosphataemia",
        "hypertension",
        "diabetes",
        "pneumonia",
        "infection",
        "anemia",
        "anaemia",
        "chronic lymphocytic thyroiditis",
        "thyroiditis",
        "cancer",
        "fracture",
        "kidney disease",
        "heart failure",
    ],
    "Symptoms": [
        "pain",
        "fever",
        "tiredness",
        "fatigue",
        "breathing difficulties",
        "shortness of breath",
        "muscle aches",
        "headache",
        "nausea",
        "vomiting",
        "weakness",
        "swelling",
        "bleeding",
        "cough",
        "abdominal pain",
    ],
    "Procedures": [
        "surgery",
        "biopsy",
        "isthmectomy",
        "catheter ablation",
        "tracheostomy",
        "CT scan",
        "MRI",
        "ultrasound",
        "FNA",
    ],
    "Medications": [
        "levothyroxine",
        "betamethasone",
        "calcium gluconate",
        "alphacalcidol",
        "calcium carbonate",
        "saline infusion",
        "antibiotics",
        "insulin",
        "aspirin",
        "paracetamol",
    ],
}

NEGATION_PATTERNS = [
    r"\bno evidence of\b",
    r"\bno sign of\b",
    r"\bno history of\b",
    r"\bdenies\b",
    r"\bwithout\b",
    r"\bnegative for\b",
    r"\bnot present\b",
]

UNCERTAINTY_PATTERNS = [
    r"\brule out\b",
    r"\bpossible\b",
    r"\bpossibly\b",
    r"\bsuspected\b",
    r"\bmay have\b",
]


def find_term_status(text, term):
    text = str(text).lower()
    term = term.lower()

    for match in re.finditer(re.escape(term), text):
        start = max(0, match.start() - 100)
        end = min(len(text), match.end() + 50)
        context = text[start:end]

        if any(re.search(p, context) for p in UNCERTAINTY_PATTERNS):
            return "UNCERTAIN"

        if any(re.search(p, context) for p in NEGATION_PATTERNS):
            return "NEGATED"

        return "AFFIRMED_OR_MENTIONED"

    return "NOT_FOUND"


def verify_summary(note, summary):
    results = []

    for category, terms in CLINICAL_TERMS.items():
        for term in terms:
            source_status = find_term_status(note, term)

            if source_status == "NOT_FOUND":
                continue

            summary_status = find_term_status(summary, term)

            if summary_status == "NOT_FOUND":
                verification = "POSSIBLY MISSING"
            elif source_status != summary_status:
                verification = "CHECK STATUS MISMATCH"
            else:
                verification = "TERM FOUND - REVIEW CONTEXT"

            results.append({
                "Category": category,
                "Clinical term": term,
                "Source status": source_status,
                "Summary status": summary_status,
                "Verification": verification,
            })

    return pd.DataFrame(results)


if __name__ == "__main__":
    data = pd.read_csv("data/processed/test.csv")
    note = data.iloc[0]["note"]
    summary = textrank_summarize(note, num_sentences=5)

    print("\nGENERATED SUMMARY:\n")
    print(summary)

    results = verify_summary(note, summary)

    print("\nCLINICAL INFORMATION VERIFICATION:\n")
    if results.empty:
        print("No configured clinical terms found.")
    else:
        print(results.to_string(index=False))
        print("\nVerification counts:")
        print(results["Verification"].value_counts().to_string())
