# Architecture

## Pipeline overview

1. **Ingestion layer** — yfinance/ccxt pulls raw OHLCV data,
   cached to data/raw/. Handles retries and multi-ticker fetch.

2. **Feature engineering** — Kalman filter and wavelet decomposition
   separate trend from noise; rolling volatility and return-based
   features are computed here.

3. **Model layer** — Isolation Forest and Autoencoder models flag
   anomalies; GARCH and LSTM models forecast short-term volatility.

4. **API layer** — FastAPI service exposes /predict, /anomalies,
   and /health endpoints, serving model outputs.

5. **Dashboard layer** — Streamlit app calls the API and displays
   live risk metrics and flagged anomalies.

Data flows top to bottom: ingestion → features → models → API → dashboard.
