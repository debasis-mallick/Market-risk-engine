import numpy as np
import pandas as pd


def add_returns(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """Add simple and log returns."""
    df = df.copy()
    df["return"] = df[price_col].pct_change()
    df["log_return"] = np.log(df[price_col] / df[price_col].shift(1))
    return df


def add_rolling_volatility(df: pd.DataFrame, windows: list[int] = [5, 21, 63]) -> pd.DataFrame:
    """Add rolling realized volatility over multiple windows."""
    df = df.copy()
    for w in windows:
        df[f"rolling_vol_{w}d"] = df["log_return"].rolling(window=w).std()
    return df


def add_rolling_stats(df: pd.DataFrame, price_col: str = "Close", windows: list[int] = [5, 21]) -> pd.DataFrame:
    """Add rolling mean and rolling z-score of price."""
    df = df.copy()
    for w in windows:
        roll_mean = df[price_col].rolling(window=w).mean()
        roll_std = df[price_col].rolling(window=w).std()
        df[f"rolling_mean_{w}d"] = roll_mean
        df[f"zscore_{w}d"] = (df[price_col] - roll_mean) / roll_std
    return df

def build_feature_pipeline(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """Run the full feature engineering pipeline on raw OHLCV data."""
    df = add_returns(df, price_col=price_col)
    df = add_rolling_volatility(df)
    df = add_rolling_stats(df, price_col=price_col)
    df = df.dropna()
    return df


if __name__ == "__main__":
    import logging
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
    PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for csv_file in RAW_DIR.glob("*.csv"):
        ticker = csv_file.stem
        logger.info(f"Building features for {ticker}")
        raw_df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        feat_df = build_feature_pipeline(raw_df)
        out_path = PROCESSED_DIR / f"{ticker}_features.csv"
        feat_df.to_csv(out_path)
        logger.info(f"Saved {len(feat_df)} rows to {out_path}")