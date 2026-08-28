# Asclepius Dataset Loader

Loads the Hugging Face dataset `starmpcc/Asclepius-Synthetic-Clinical-Notes` and prints its available splits, row counts, and columns.

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

## Run

```powershell
py load_dataset.py
```

To use a custom Hugging Face cache directory:

```powershell
py load_dataset.py --cache-dir .\huggingface_cache
```

The downloaded dataset is cached locally and ignored by Git.
