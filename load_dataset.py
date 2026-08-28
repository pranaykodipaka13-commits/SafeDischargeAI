"""Load and inspect the Asclepius synthetic clinical notes dataset."""

from argparse import ArgumentParser

from datasets import load_dataset

DATASET_ID = "starmpcc/Asclepius-Synthetic-Clinical-Notes"


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        help="Optional directory for the Hugging Face dataset cache.",
    )
    args = parser.parse_args()

    dataset = load_dataset(DATASET_ID, cache_dir=args.cache_dir)
    print(f"Dataset: {DATASET_ID}")
    print(f"Splits: {list(dataset.keys())}")
    for split_name, split in dataset.items():
        print(f"{split_name}: {len(split)} rows")
        print(f"Columns: {split.column_names}")


if __name__ == "__main__":
    main()
