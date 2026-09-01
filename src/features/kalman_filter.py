import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter


def apply_kalman_filter(prices: pd.Series, process_var: float = 1e-5, measurement_var: float = 1e-2) -> pd.Series:
    """
    Apply a 1D Kalman filter to extract the underlying trend from noisy price data.

    process_var: how much we expect the true trend to change day-to-day (lower = smoother trend)
    measurement_var: how noisy we believe the observed price is (higher = trusts filter's own estimate more)
    """
    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.x = np.array([[prices.iloc[0]]])   # initial state estimate = first observed price
    kf.F = np.array([[1]])                # state transition: trend persists (random walk)
    kf.H = np.array([[1]])                # we directly observe price as a measurement of trend
    kf.P *= 1.0                           # initial uncertainty
    kf.R = measurement_var                # measurement noise
    kf.Q = process_var                    # process noise

    filtered = []
    for price in prices:
        kf.predict()
        kf.update(price)
        filtered.append(kf.x[0, 0])

    return pd.Series(filtered, index=prices.index, name="kalman_trend")

def add_kalman_features(df: pd.DataFrame, price_col: str = "Close", process_var: float = 1e-5, measurement_var: float = 1e-2) -> pd.DataFrame:
    """Add Kalman-filtered trend and residual (deviation from trend) to the dataframe."""
    df = df.copy()
    df["kalman_trend"] = apply_kalman_filter(df[price_col], process_var=process_var, measurement_var=measurement_var)
    df["kalman_residual"] = df[price_col] - df["kalman_trend"]
    return df


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend, avoids Tkinter/Tcl issues
    import matplotlib.pyplot as plt
    from pathlib import Path

    RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
    df = pd.read_csv(RAW_DIR / "AAPL.csv", index_col=0, parse_dates=True)
    df = add_kalman_features(df)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(df.index, df["Close"], label="Raw price", alpha=0.5, linewidth=1)
    axes[0].plot(df.index, df["kalman_trend"], label="Kalman trend", linewidth=1.5, color="red")
    axes[0].set_title("AAPL — Raw price vs Kalman-filtered trend")
    axes[0].legend()

    axes[1].plot(df.index, df["kalman_residual"], color="darkorange", linewidth=0.8)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_title("Kalman residual (price - trend)")

    plt.tight_layout()
    plt.savefig(Path(__file__).resolve().parents[2] / "docs" / "kalman_example.png")
    print("Saved plot to docs/kalman_example.png")