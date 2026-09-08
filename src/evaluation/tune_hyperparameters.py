import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.anomaly_isolation_forest import fit_isolation_forest, add_anomaly_scores
from models.volatility_garch import fit_garch
from utils.experiment_logger import log_experiment

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def tune_isolation_forest_contamination(ticker: str = "AAPL") -> None:
    """
    Sweep Isolation Forest's contamination parameter and log how the
    anomaly count and average anomaly severity change.
    """
    df = pd.read_csv(PROCESSED_DIR / f"{ticker}_features.csv", index_col=0, parse_dates=True)

    for contamination in [0.01, 0.03, 0.05, 0.08, 0.10]:
        model, X = fit_isolation_forest(df, contamination=contamination)
        result_df = add_anomaly_scores(df, model, X)

        anomalies = result_df[result_df["is_anomaly"] == -1]
        avg_severity = anomalies["anomaly_score"].mean() if len(anomalies) > 0 else None

        log_experiment(
            experiment_name="isolation_forest_tuning",
            params={"ticker": ticker, "contamination": contamination},
            metrics={
                "n_anomalies": len(anomalies),
                "avg_anomaly_score": round(avg_severity, 6) if avg_severity else None,
            },
        )

def tune_garch_order(ticker: str = "AAPL") -> None:
    """
    Compare GARCH(1,1) against alternative (p,q) orders using AIC/BIC —
    standard model selection criteria that penalize unnecessary complexity.
    Lower AIC/BIC indicates a better tradeoff between fit and simplicity.
    """
    df = pd.read_csv(PROCESSED_DIR / f"{ticker}_features.csv", index_col=0, parse_dates=True)
    log_returns = df["log_return"].dropna()

    for p, q in [(1, 1), (1, 2), (2, 1), (2, 2)]:
        try:
            result = fit_garch(log_returns, p=p, q=q)

            log_experiment(
                experiment_name="garch_order_tuning",
                params={"ticker": ticker, "p": p, "q": q},
                metrics={
                    "aic": round(result.aic, 3),
                    "bic": round(result.bic, 3),
                    "log_likelihood": round(result.loglikelihood, 3),
                },
            )
        except Exception as e:
            print(f"GARCH({p},{q}) failed: {e}")


if __name__ == "__main__":
    print("Tuning Isolation Forest contamination...")
    tune_isolation_forest_contamination("AAPL")

    print("\nTuning GARCH order...")
    tune_garch_order("AAPL")

    print("\nDone. Check docs/experiments/ for logged results.")



    