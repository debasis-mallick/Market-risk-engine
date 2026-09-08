import pandas as pd
from pathlib import Path
from datetime import datetime

LOG_DIR = Path(__file__).resolve().parents[2] / "docs" / "experiments"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def log_experiment(experiment_name: str, params: dict, metrics: dict) -> None:
    """
    Append a single experiment run's parameters and metrics to a CSV log.
    Each experiment_name gets its own log file (e.g. isolation_forest_tuning.csv).
    """
    log_path = LOG_DIR / f"{experiment_name}.csv"

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **params,
        **metrics,
    }
    row_df = pd.DataFrame([row])

    if log_path.exists():
        row_df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        row_df.to_csv(log_path, mode="w", header=True, index=False)

    print(f"Logged experiment to {log_path.name}: {params} -> {metrics}")