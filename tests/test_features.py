import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from features.build_features import add_returns, add_rolling_volatility, add_rolling_stats, build_feature_pipeline


@pytest.fixture
def sample_price_df():
    """A small, deterministic price series for testing — not real market data."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100))
    return pd.DataFrame({"Close": prices}, index=dates)


def test_add_returns_creates_expected_columns(sample_price_df):
    result = add_returns(sample_price_df)
    assert "return" in result.columns
    assert "log_return" in result.columns


def test_add_returns_first_row_is_nan(sample_price_df):
    """First row has no prior price, so return should be NaN."""
    result = add_returns(sample_price_df)
    assert pd.isna(result["return"].iloc[0])
    assert pd.isna(result["log_return"].iloc[0])


def test_log_return_matches_manual_calculation(sample_price_df):
    result = add_returns(sample_price_df)
    expected = np.log(sample_price_df["Close"].iloc[1] / sample_price_df["Close"].iloc[0])
    assert np.isclose(result["log_return"].iloc[1], expected)


def test_rolling_volatility_creates_expected_columns(sample_price_df):
    df = add_returns(sample_price_df)
    result = add_rolling_volatility(df, windows=[5, 21])
    assert "rolling_vol_5d" in result.columns
    assert "rolling_vol_21d" in result.columns


def test_build_feature_pipeline_drops_all_nans(sample_price_df):
    """The full pipeline should never return rows with missing values."""
    result = build_feature_pipeline(sample_price_df)
    assert result.isna().sum().sum() == 0


def test_build_feature_pipeline_produces_fewer_rows_than_input(sample_price_df):
    """Rolling windows consume initial rows — output should be shorter than input."""
    result = build_feature_pipeline(sample_price_df)
    assert len(result) < len(sample_price_df)

