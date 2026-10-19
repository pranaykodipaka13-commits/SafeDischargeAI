
import pandas as pd

path = "data/processed/evaluation_results.csv"
results = pd.read_csv(path)

print("\n========== EVALUATION FILE ==========")
print("Rows:", len(results))
print("Columns:", list(results.columns))

print("\n========== FIRST 20 ROWS ==========")
print(results.head(20).to_string(index=False))

print("\n========== SUMMARY STATISTICS ==========")
print(results.describe().to_string())

print("\n========== NOTES WITH MOST POSSIBLE OMISSIONS ==========")
print(
    results.sort_values(
        "Possible omissions",
        ascending=False
    ).head(15).to_string(index=False)
)

print("\n========== ZERO-OMISSION NOTES ==========")
print(
    (results["Possible omissions"] == 0).sum()
)
