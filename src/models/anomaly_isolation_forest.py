import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


FEATURE_COLUMNS = [
    "log_return",
    "rolling_vol_5d",
    "rolling_vol_21d",
    "rolling_vol_63d",
    "zscore_5d",
    "zscore_21d",
]


def fit_isolation_forest(df: pd.DataFrame, contamination: float = 0.05, random_state: int = 42) -> tuple:
    """
    Fit an Isolation Forest on engineered features to detect anomalous trading days.

    contamination: expected proportion of anomalies in the data (0.05 = assume ~5% of days are anomalous)
    Returns the fitted model and the feature matrix used.
    """
    X = df[FEATURE_COLUMNS].copy()

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X)

    return model, X


def add_anomaly_scores(df: pd.DataFrame, model: IsolationForest, X: pd.DataFrame) -> pd.DataFrame:
    """Add anomaly predictions and scores to the dataframe."""
    df = df.copy()
    df["anomaly_score"] = model.decision_function(X)  # higher = more normal, lower = more anomalous
    df["is_anomaly"] = model.predict(X)  # -1 = anomaly, 1 = normal
    return df


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
    DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"

    df = pd.read_csv(PROCESSED_DIR / "AAPL_features.csv", index_col=0, parse_dates=True)

    model, X = fit_isolation_forest(df, contamination=0.05)
    df = add_anomaly_scores(df, model, X)

    n_anomalies = (df["is_anomaly"] == -1).sum()
    print(f"Flagged {n_anomalies} anomalies out of {len(df)} days ({n_anomalies/len(df)*100:.1f}%)")
    print("\nTop 10 most anomalous days (lowest anomaly_score):")
    print(df.nsmallest(10, "anomaly_score")[["Close", "log_return", "rolling_vol_21d", "anomaly_score"]])

    # Plot price with flagged anomalies overlaid
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df["Close"], label="Close price", linewidth=1, alpha=0.7)

    anomalies = df[df["is_anomaly"] == -1]
    plt.scatter(anomalies.index, anomalies["Close"], color="red", s=30, label="Flagged anomaly", zorder=5)

    plt.title("AAPL — Isolation Forest anomaly detection")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(DOCS_DIR / "isolation_forest_anomalies.png")
    print("\nSaved plot to docs/isolation_forest_anomalies.png")