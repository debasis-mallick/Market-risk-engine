import sys
import logging
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parent))

from ingestion.fetch_data import fetch_multiple_tickers
from features.build_features import build_feature_pipeline
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pipeline_refresh")

TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA"]
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def refresh_pipeline():
    """
    Re-fetch the latest market data and rebuild engineered features for
    all tracked tickers. Designed to be run on a schedule (see
    .github/workflows/refresh.yml) so the underlying data stays current.
    """
    logger.info(f"Starting pipeline refresh at {datetime.now().isoformat()}")

    logger.info("Step 1/2: Re-fetching raw data...")
    fetch_multiple_tickers(TICKERS)

    logger.info("Step 2/2: Rebuilding engineered features...")
    for ticker in TICKERS:
        raw_path = RAW_DIR / f"{ticker}.csv"
        if not raw_path.exists():
            logger.warning(f"Skipping {ticker}: raw data not found after fetch")
            continue

        raw_df = pd.read_csv(raw_path, index_col=0, parse_dates=True)
        feat_df = build_feature_pipeline(raw_df)
        out_path = PROCESSED_DIR / f"{ticker}_features.csv"
        feat_df.to_csv(out_path)
        logger.info(f"Refreshed {ticker}: {len(feat_df)} rows -> {out_path.name}")

    logger.info("Pipeline refresh complete.")


if __name__ == "__main__":
    refresh_pipeline()