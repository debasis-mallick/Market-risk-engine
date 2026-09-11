import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from models.anomaly_isolation_forest import fit_isolation_forest, add_anomaly_scores, FEATURE_COLUMNS


@pytest.fixture
def sample_features_df():
    """Synthetic feature data with the same columns the model expects."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "log_return": np.random.randn(n) * 0.01,
        "rolling_vol_5d": np.abs(np.random.randn(n) * 0.01) + 0.01,
        "rolling_vol_21d": np.abs(np.random.randn(n) * 0.01) + 0.01,
        "rolling_vol_63d": np.abs(np.random.randn(n) * 0.01) + 0.01,
        "zscore_5d": np.random.randn(n),
        "zscore_21d": np.random.randn(n),
    })
    return df


def test_fit_isolation_forest_returns_model_and_features(sample_features_df):
    model, X = fit_isolation_forest(sample_features_df)
    assert model is not None
    assert list(X.columns) == FEATURE_COLUMNS


def test_anomaly_scores_added_correctly(sample_features_df):
    model, X = fit_isolation_forest(sample_features_df)
    result = add_anomaly_scores(sample_features_df, model, X)
    assert "anomaly_score" in result.columns
    assert "is_anomaly" in result.columns


def test_anomaly_labels_are_valid_values(sample_features_df):
    """is_anomaly should only ever be -1 (anomaly) or 1 (normal), per sklearn's convention."""
    model, X = fit_isolation_forest(sample_features_df)
    result = add_anomaly_scores(sample_features_df, model, X)
    assert set(result["is_anomaly"].unique()).issubset({-1, 1})


def test_contamination_affects_anomaly_count(sample_features_df):
    """Higher contamination should flag more (or equal) anomalies, never fewer."""
    model_low, X_low = fit_isolation_forest(sample_features_df, contamination=0.05)
    result_low = add_anomaly_scores(sample_features_df, model_low, X_low)
    n_low = (result_low["is_anomaly"] == -1).sum()

    model_high, X_high = fit_isolation_forest(sample_features_df, contamination=0.20)
    result_high = add_anomaly_scores(sample_features_df, model_high, X_high)
    n_high = (result_high["is_anomaly"] == -1).sum()

    assert n_high >= n_low