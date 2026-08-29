import time
import logging
from pathlib import Path
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_ticker(ticker: str, period: str = "2y", interval: str = "1d", max_retries: int = 3) -> pd.DataFrame:
    """Fetch OHLCV data for a single ticker with retry on failure."""
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Fetching {ticker} (attempt {attempt}/{max_retries})")
            df = yf.download(ticker, period=period, interval=interval, progress=False)
            if df.empty:
                raise ValueError(f"No data returned for {ticker}")

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

                before = len(df)
                df = df.dropna(subset=["Open", "High", "Low", "Close"])
                dropped = before - len(df)
            if dropped > 0:
                logger.info(f"Dropped {dropped} incomplete row(s) for {ticker}")


                df.attrs["ticker"] = ticker
                return df
            
        except Exception as e:
            logger.warning(f"Attempt {attempt} failed for {ticker}: {e}")
            if attempt < max_retries:
                time.sleep(2 * attempt)  # exponential-ish backoff
            else:
                logger.error(f"All {max_retries} attempts failed for {ticker}")
                raise

def save_to_cache(df: pd.DataFrame, ticker: str) -> Path:
    """Save fetched data to data/raw/ as CSV."""
    filepath = RAW_DATA_DIR / f"{ticker}.csv"
    df.to_csv(filepath)
    logger.info(f"Saved {ticker} data to {filepath}")
    return filepath


def fetch_multiple_tickers(tickers: list[str], period: str = "2y", interval: str = "1d") -> dict[str, pd.DataFrame]:
    """Fetch and cache data for multiple tickers. Continues even if one ticker fails."""
    results = {}
    failed = []

    for ticker in tickers:
        try:
            df = fetch_ticker(ticker, period=period, interval=interval)
            save_to_cache(df, ticker)
            results[ticker] = df
        except Exception as e:
            logger.error(f"Skipping {ticker} after repeated failures: {e}")
            failed.append(ticker)

    if failed:
        logger.warning(f"Failed tickers: {failed}")
    logger.info(f"Successfully fetched {len(results)}/{len(tickers)} tickers")

    return results


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "TSLA"]
    data = fetch_multiple_tickers(tickers)        