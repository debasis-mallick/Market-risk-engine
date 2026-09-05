import numpy as np
import pandas as pd
from arch import arch_model


def fit_garch(log_returns: pd.Series, p: int = 1, q: int = 1) -> "arch.univariate.base.ARCHModelResult":
    """
    Fit a GARCH(p, q) model on log returns.

    GARCH(1,1) is the standard baseline: tomorrow's variance depends on
    today's squared return (the 'ARCH' term) and today's variance estimate
    (the 'GARCH' term) — directly modeling the volatility clustering
    behavior confirmed in Day 3's ACF analysis.

    Note: arch_model expects returns scaled as percentages (not decimals)
    for numerical stability during optimization.
    """
    returns_pct = log_returns * 100

    model = arch_model(returns_pct, vol="Garch", p=p, q=q, dist="normal")
    result = model.fit(disp="off")

    return result


def forecast_volatility(result, horizon: int = 5) -> pd.DataFrame:
    """
    Forecast volatility (conditional standard deviation) for the next `horizon` days.
    Returns forecasted variance converted back to the original return scale.
    """
    forecast = result.forecast(horizon=horizon, reindex=False)
    variance_forecast = forecast.variance.values[-1]  # last row = forecast from most recent date

    # Convert back from percentage-scale variance to decimal-scale volatility
    volatility_forecast = np.sqrt(variance_forecast) / 100

    return pd.DataFrame({
        "day_ahead": range(1, horizon + 1),
        "forecasted_volatility": volatility_forecast,
    })



if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
    DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"

    df = pd.read_csv(PROCESSED_DIR / "AAPL_features.csv", index_col=0, parse_dates=True)
    log_returns = df["log_return"].dropna()

    result = fit_garch(log_returns)
    print(result.summary())

    forecast_df = forecast_volatility(result, horizon=5)
    print("\n5-day-ahead volatility forecast:")
    print(forecast_df.to_string(index=False))

    # Compare GARCH's in-sample conditional volatility against realized rolling volatility
    conditional_vol = result.conditional_volatility / 100  # back to decimal scale

    plt.figure(figsize=(12, 5))
    plt.plot(df.index[-len(conditional_vol):], conditional_vol, label="GARCH conditional volatility", linewidth=1.2)
    plt.plot(df.index, df["rolling_vol_21d"], label="21-day rolling volatility (realized)", linewidth=1, alpha=0.7)
    plt.title("AAPL — GARCH conditional volatility vs realized rolling volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.legend()
    plt.tight_layout()
    plt.savefig(DOCS_DIR / "garch_volatility.png")
    print("\nSaved plot to docs/garch_volatility.png")