import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from summarizer import textrank_summarize

CSV = Path(__file__).resolve().parent.parent / "data" / "processed" / "test.csv"


def load_note(row):
    return str(pd.read_csv(CSV).iloc[row]["note"])


def test_row7_summary_keeps_ct_findings():
    note = load_note(7)
    # Guard: make sure we loaded the right note
    assert "ground-glass" in note.lower()

    summary = textrank_summarize(note, num_sentences=8).lower()
    assert "ground-glass" in summary, "CT finding (ground-glass opacities) was dropped"
    assert "infiltrate" in summary, "CT finding (pulmonary infiltrates) was dropped"


def test_row45_summary_keeps_ulcer_finding():
    note = load_note(45)
    assert "ulcer" in note.lower()
    summary = textrank_summarize(note, num_sentences=8).lower()
    assert "ulcer" in summary, "Clinical exam finding (ulcer) was dropped"


def test_row212_summary_keeps_lab_findings():
    note = load_note(212)
    assert "d-dimer" in note.lower()
    summary = textrank_summarize(note, num_sentences=8).lower()
    assert "d-dimer" in summary, "Abnormal lab results were dropped"