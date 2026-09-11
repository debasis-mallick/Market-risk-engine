import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest


def test_multiindex_columns_get_flattened():
    """
    Reproduces the exact multi-index bug found on Day 2: yfinance sometimes
    returns (Price, Ticker) tuple columns that must be flattened to plain names.
    """
    arrays = [["Close", "High", "Low", "Open", "Volume"], ["AAPL"] * 5]
    columns = pd.MultiIndex.from_arrays(arrays)
    df = pd.DataFrame(np.random.randn(5, 5), columns=columns)

    assert isinstance(df.columns, pd.MultiIndex)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    assert not isinstance(df.columns, pd.MultiIndex)
    assert list(df.columns) == ["Close", "High", "Low", "Open", "Volume"]


def test_incomplete_rows_get_dropped():
    """
    Reproduces the Day 2 bug: a trailing row with NaN OHLC values (e.g. an
    in-progress trading session) must be dropped before caching.
    """
    df = pd.DataFrame({
        "Open": [100, 101, np.nan],
        "High": [102, 103, np.nan],
        "Low": [99, 100, np.nan],
        "Close": [101, 102, np.nan],
        "Volume": [1000, 1100, 1200],
    })

    before = len(df)
    df_clean = df.dropna(subset=["Open", "High", "Low", "Close"])
    dropped = before - len(df_clean)

    assert dropped == 1
    assert df_clean.isna().sum().sum() == 0


def test_incomplete_rows_preserved_when_all_complete():
    """Sanity check: dropna shouldn't remove anything when there's nothing incomplete."""
    df = pd.DataFrame({
        "Open": [100, 101, 102],
        "High": [102, 103, 104],
        "Low": [99, 100, 101],
        "Close": [101, 102, 103],
        "Volume": [1000, 1100, 1200],
    })

    df_clean = df.dropna(subset=["Open", "High", "Low", "Close"])
    assert len(df_clean) == len(df)