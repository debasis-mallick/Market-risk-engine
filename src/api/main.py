import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd


import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("market_risk_api")
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from models.anomaly_isolation_forest import fit_isolation_forest, add_anomaly_scores
from models.volatility_garch import fit_garch, forecast_volatility

app = FastAPI(
    title="Market Risk & Anomaly Detection Engine",
    description="API for anomaly detection and volatility forecasting on equity data",
    version="1.0.0",
)

@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    logger.info(f"Incoming request: {request.method} {request.url.path}")

    response = await call_next(request)

    duration = time.time() - start_time
    logger.info(f"Completed: {request.method} {request.url.path} — status {response.status_code} — {duration:.3f}s")

    return response

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
AVAILABLE_TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA"]


@app.get("/health")
def health_check():
    """Simple health check endpoint to confirm the API is running."""
    return {"status": "ok", "available_tickers": AVAILABLE_TICKERS}

def load_ticker_data(ticker: str) -> pd.DataFrame:
    """Load processed features for a ticker, raising a clean 404 if not found."""
    ticker = ticker.upper()
    filepath = PROCESSED_DIR / f"{ticker}_features.csv"
    if not filepath.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Ticker '{ticker}' not found. Available tickers: {AVAILABLE_TICKERS}",
        )
    return pd.read_csv(filepath, index_col=0, parse_dates=True)


@app.get("/anomalies/{ticker}")
def get_anomalies(
    ticker: str,
    contamination: float = Query(default=0.05, ge=0.01, le=0.5, description="Expected proportion of anomalies (0.01-0.5)"),
):
    """
    Return flagged anomalous trading days for a given ticker using
    Isolation Forest.
    """
    df = load_ticker_data(ticker)

    model, X = fit_isolation_forest(df, contamination=contamination)
    df = add_anomaly_scores(df, model, X)

    anomalies = df[df["is_anomaly"] == -1]

    return {
        "ticker": ticker.upper(),
        "contamination": contamination,
        "n_anomalies": len(anomalies),
        "anomalies": [
            {
                "date": str(date.date()),
                "close": round(row["Close"], 2),
                "log_return": round(row["log_return"], 4),
                "anomaly_score": round(row["anomaly_score"], 4),
            }
            for date, row in anomalies.iterrows()
        ],
    }


@app.get("/predict/{ticker}")
def predict_volatility(
    ticker: str,
    horizon: int = Query(default=5, ge=1, le=30, description="Forecast horizon in days (1-30)"),
):
    """
    Forecast next-day (and up to `horizon` days ahead) volatility for a
    given ticker using GARCH(1,1).
    """
    df = load_ticker_data(ticker)
    log_returns = df["log_return"].dropna()

    try:
        result = fit_garch(log_returns)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GARCH fitting failed: {e}")

    forecast_df = forecast_volatility(result, horizon=horizon)

    return {
        "ticker": ticker.upper(),
        "horizon_days": horizon,
        "forecast": forecast_df.to_dict(orient="records"),
    }