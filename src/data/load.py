"""Utilities for loading project datasets."""

from pathlib import Path

import pandas as pd

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def load_raw_dataset(filename: str = "elo7_recruitment_dataset.csv") -> pd.DataFrame:
    """Load the raw Elo7 recruitment dataset as a DataFrame."""
    return pd.read_csv(RAW_DATA_DIR / filename, parse_dates=["creation_date"])
