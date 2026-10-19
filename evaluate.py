
import pandas as pd
from summarizer import textrank_summarize
from verifier import verify_summary


def main():
    data = pd.read_csv("data/processed/test.csv")

    total_terms = 0
    missing_terms = 0
    notes_with_terms = 0
    flagged_notes = 0
    note_results = []

    print(f"Evaluating {len(data)} test notes...")

    for index, row in data.iterrows():
        note = str(row["note"])
        summary = textrank_summarize(note, num_sentences=5)
        results = verify_summary(note, summary)

        # Count configured terms detected by the verifier.
        total = len(results)
        missing = 0

        if total > 0:
            missing = int(
                (results["Verification"] == "POSSIBLY MISSING").sum()
            )
            notes_with_terms += 1

        total_terms += total
        missing_terms += missing

        if missing > 0:
            flagged_notes += 1

        note_results.append({
            "Test row": index,
            "Terms checked": total,
            "Possible omissions": missing,
            "Has configured terms": total > 0,
            "Flagged for review": missing > 0,
        })

        if (index + 1) % 50 == 0:
            print(f"Processed {index + 1} of {len(data)} notes...")

    output = pd.DataFrame(note_results)
    output.to_csv(
        "data/processed/evaluation_results.csv",
        index=False,
    )

    print("\n========== EVALUATION RESULTS ==========")
    print(f"Test notes processed: {len(data)}")
    print(f"Notes with configured terms: {notes_with_terms}")
    print(f"Notes without configured terms: {len(data) - notes_with_terms}")
    print(f"Term occurrences checked: {total_terms}")
    print(f"Possible omissions: {missing_terms}")
    print(f"Notes flagged for review: {flagged_notes}")

    if total_terms > 0:
        coverage = (total_terms - missing_terms) / total_terms * 100
        omission_rate = missing_terms / total_terms * 100

        print(f"Preliminary term coverage: {coverage:.2f}%")
        print(f"Preliminary omission rate: {omission_rate:.2f}%")
    else:
        print("No configured terms were detected.")

    print("\nSaved all note results to:")
    print("data/processed/evaluation_results.csv")


if __name__ == "__main__":
    main()
