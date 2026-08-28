"""Load and prepare the Asclepius synthetic clinical notes dataset."""

from argparse import ArgumentParser
from pathlib import Path

from datasets import load_dataset
from sklearn.model_selection import train_test_split

DATASET_ID = "starmpcc/Asclepius-Synthetic-Clinical-Notes"


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        help="Optional directory for the Hugging Face dataset cache.",
    )
    args = parser.parse_args()

    # Load dataset
    dataset = load_dataset(DATASET_ID, cache_dir=args.cache_dir)

    print(f"Dataset: {DATASET_ID}")
    print(f"Splits: {list(dataset.keys())}")

    for split_name, split in dataset.items():
        print(f"{split_name}: {len(split)} rows")
        print(f"Columns: {split.column_names}")

    # Convert train split to DataFrame
    df = dataset["train"].to_pandas()

    print("\nDataset shape:")
    print(df.shape)

    # Show task distribution
    print("\nTask distribution:")
    print(df["task"].value_counts())

    # Filter summarization records
    print("\nFiltering summarization data...")

    summary_df = df[df["task"] == "Summarization"].copy()

    print("\nSummarization dataset:")
    print(f"Number of samples: {len(summary_df)}")
    print(f"Columns: {list(summary_df.columns)}")

    # Select 5,000 samples
    print("\nCreating project subset...")

    project_df = summary_df.sample(
        n=5000,
        random_state=42
    ).reset_index(drop=True)

    print(f"Project dataset size: {len(project_df)}")

    # Create 80/10/10 split
    print("\nCreating train/validation/test splits...")

    train_df, temp_df = train_test_split(
        project_df,
        test_size=0.20,
        random_state=42
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42
    )

    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

    # Create output directory
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save splits
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "validation.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print("\nSaved processed datasets:")
    print(f"Train: {output_dir / 'train.csv'}")
    print(f"Validation: {output_dir / 'validation.csv'}")
    print(f"Test: {output_dir / 'test.csv'}")


if __name__ == "__main__":
    main()