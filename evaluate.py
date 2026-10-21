
import pandas as pd
from summarizer import textrank_summarize
from verifier import verify_summary


def main():
    data = pd.read_csv("data/processed/test.csv")

    detailed_results = []

    print(f"Evaluating {len(data)} test notes...")

    for index, row in data.iterrows():
        note = str(row["note"])
        summary = textrank_summarize(note, num_sentences=8)
        results = verify_summary(note, summary)

        if len(results) > 0:
            missing = int(
                (results["Verification"] == "POSSIBLY MISSING").sum()
            )
        else:
            missing = 0

        detailed_results.append({
            "Test row": index,
            "Source note": note,
            "Generated summary": summary,
            "Terms checked": len(results),
            "Possible omissions": missing,
            "Verification details": (
                results.to_json(orient="records", force_ascii=False)
            ),
        })

        if (index + 1) % 50 == 0:
            print(f"Processed {index + 1} of {len(data)} notes...")

    output = pd.DataFrame(detailed_results)

    output.to_csv(
        "data/processed/detailed_evaluation_results.csv",
        index=False
    )

    print("\nDetailed evaluation completed.")
    print("Saved to: data/processed/detailed_evaluation_results.csv")

    flagged = output[output["Possible omissions"] > 0]

    print(f"Notes flagged for review: {len(flagged)}")

    if len(flagged) > 0:
        print("\nFirst 3 flagged notes:")
        for _, row in flagged.head(3).iterrows():
            print("\n" + "=" * 70)
            print(f"Test row: {row['Test row']}")
            print("\nSOURCE NOTE:")
            print(str(row["Source note"])[:1500])
            print("\nGENERATED SUMMARY:")
            print(row["Generated summary"])
            print("\nVERIFICATION DETAILS:")
            print(row["Verification details"])


if __name__ == "__main__":
    main()
