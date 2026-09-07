import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.anomaly_isolation_forest import fit_isolation_forest, add_anomaly_scores, FEATURE_COLUMNS as IF_FEATURES
from models.volatility_garch import fit_garch, forecast_volatility

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"


def evaluate_anomaly_detection_all_tickers() -> pd.DataFrame:
    """
    Run Isolation Forest anomaly detection across every processed ticker
    and summarize how many anomalies were flagged and when.
    """
    results = []

    for csv_file in sorted(PROCESSED_DIR.glob("*_features.csv")):
        ticker = csv_file.stem.replace("_features", "")
        df = pd.read_csv(csv_file, index_col=0, parse_dates=True)

        model, X = fit_isolation_forest(df, contamination=0.05)
        df = add_anomaly_scores(df, model, X)

        anomalies = df[df["is_anomaly"] == -1]
        n_anomalies = len(anomalies)

        results.append({
            "ticker": ticker,
            "n_days": len(df),
            "n_anomalies": n_anomalies,
            "anomaly_rate": n_anomalies / len(df),
            "earliest_anomaly": anomalies.index.min() if n_anomalies > 0 else None,
            "latest_anomaly": anomalies.index.max() if n_anomalies > 0 else None,
            "worst_anomaly_score": anomalies["anomaly_score"].min() if n_anomalies > 0 else None,
        })

    return pd.DataFrame(results)


def evaluate_volatility_forecasting_all_tickers() -> pd.DataFrame:
    """
    Fit GARCH(1,1) on every processed ticker and summarize persistence
    (alpha + beta) and 1-day-ahead volatility forecast.
    """
    results = []

    for csv_file in sorted(PROCESSED_DIR.glob("*_features.csv")):
        ticker = csv_file.stem.replace("_features", "")
        df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        log_returns = df["log_return"].dropna()

        try:
            result = fit_garch(log_returns)
            forecast_df = forecast_volatility(result, horizon=1)

            alpha = result.params.get("alpha[1]", np.nan)
            beta = result.params.get("beta[1]", np.nan)

            results.append({
                "ticker": ticker,
                "alpha": alpha,
                "beta": beta,
                "persistence (alpha+beta)": alpha + beta,
                "1day_vol_forecast": forecast_df["forecasted_volatility"].iloc[0],
            })
        except Exception as e:
            print(f"GARCH failed for {ticker}: {e}")

    return pd.DataFrame(results)


if __name__ == "__main__":
    print("=" * 60)
    print("ANOMALY DETECTION — Isolation Forest across all tickers")
    print("=" * 60)
    anomaly_summary = evaluate_anomaly_detection_all_tickers()
    print(anomaly_summary.to_string(index=False))

    print("\n" + "=" * 60)
    print("VOLATILITY FORECASTING — GARCH across all tickers")
    print("=" * 60)
    volatility_summary = evaluate_volatility_forecasting_all_tickers()
    print(volatility_summary.to_string(index=False))

    # Save both as CSVs for reference
    anomaly_summary.to_csv(DOCS_DIR / "anomaly_summary.csv", index=False)
    volatility_summary.to_csv(DOCS_DIR / "volatility_summary.csv", index=False)
    print("\nSaved anomaly_summary.csv and volatility_summary.csv to docs/")