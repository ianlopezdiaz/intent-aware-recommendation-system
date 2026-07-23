"""Utilities for loading project datasets."""

from pathlib import Path

import pandas as pd

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def load_raw_dataset(filename: str = "elo7_recruitment_dataset.csv") -> pd.DataFrame:
    """Load the raw Elo7 recruitment dataset as a DataFrame."""
    return pd.read_csv(RAW_DATA_DIR / filename, parse_dates=["creation_date"])


def load_processed_dataset(filename: str = "01_data.parquet") -> pd.DataFrame:
    """Load a cleaned dataset persisted by an earlier notebook.

    Parameters
    ----------
    filename : str, default "01_data.parquet"
        Name of the parquet file under `data/processed/`, e.g. the cleaned
        dataset saved by notebook 01.

    Returns
    -------
    pandas.DataFrame
        The persisted dataset, with dtypes preserved as saved.
    """
    return pd.read_parquet(PROCESSED_DATA_DIR / filename)
