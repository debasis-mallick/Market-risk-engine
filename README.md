# Real-Time Market Risk & Anomaly Detection Engine

An end-to-end system that ingests market data, applies signal-processing techniques to separate trend from noise, detects anomalous trading behavior using unsupervised ML, and forecasts short-term volatility — served via a tested, containerized, CI/CD-enabled API.


🔗 **Live demo**: https://market-risk-engine-aowv.onrender.com/docs
*(Free-tier hosting — the service spins down after 15 minutes of inactivity; the first request after that may take 30-60 seconds to wake up.)*



## Why this project

Most portfolio ML projects stop at a notebook with a trained model. This one goes further: every finding is **cross-validated across multiple independent methods**, every model is **compared against honest baselines** (including cases where the simpler model wins), and the whole pipeline is **productionized** — tested, containerized, continuously integrated, and self-refreshing.

## Key findings

- **Five independent methods** (EDA, ACF analysis, Kalman filter residuals, wavelet decomposition, and Isolation Forest/Autoencoder) all converge on the same real anomalous event in AAPL's April 2025 price action — strong evidence of a genuine signal, not a modeling artifact.
- **Anomaly timing is idiosyncratic, not systemic**: testing across 4 tickers showed each stock's flagged anomalies cluster in different time windows, meaning findings from one ticker do not automatically generalize — an important nuance easy to miss.
- **Simpler models won on both major tasks**: GARCH(1,1) beat higher-order GARCH variants by BIC, and a trivial "tomorrow = today" persistence baseline beat an LSTM for volatility forecasting. Model complexity was tested, not assumed.

Full write-ups: [Week 2 Report](docs/WEEK2_REPORT.md) | [Model Comparison](docs/model_comparison.md) | [Volatility Model Comparison](docs/volatility_model_comparison.md)

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full pipeline breakdown (ingestion → feature engineering → models → API → dashboard).

## Tech stack

- **Data & signal processing**: yfinance, pandas, NumPy, SciPy, filterpy (Kalman filtering), PyWavelets
- **Modeling**: scikit-learn (Isolation Forest), PyTorch (Autoencoder, LSTM), arch (GARCH)
- **Serving**: FastAPI, Uvicorn
- **Testing & CI/CD**: pytest, Docker, GitHub Actions (test automation + scheduled data refresh)

## Project structure

    src/
      ingestion/      # Multi-ticker data fetching with retry logic
      features/       # Returns, rolling volatility, Kalman filter, wavelets
      models/         # Isolation Forest, Autoencoder, GARCH, LSTM
      evaluation/     # Cross-ticker evaluation, hyperparameter tuning
      api/            # FastAPI service
      utils/          # Experiment logging
    tests/            # 13 passing unit tests, including regression tests
    docs/             # Architecture, reports, generated plots
    .github/workflows/
      ci.yml          # Runs test suite on every push
      refresh.yml     # Scheduled daily data refresh (weekdays)

## Running it locally

    python -m venv venv
    source venv/Scripts/activate  # or venv/bin/activate on Mac/Linux
    pip install -r requirements-api.txt

    # Fetch data and build features
    python src/pipeline_refresh.py

    # Run the API
    uvicorn src.api.main:app --reload

    # Run tests
    pytest tests/ -v

Or with Docker:

    docker build -t market-risk-engine .
    docker run -p 8000:8000 market-risk-engine

API docs available at `http://localhost:8000/docs` once running.

## API endpoints

- `GET /health` — service health check
- `GET /anomalies/{ticker}?contamination=0.05` — flagged anomalous trading days via Isolation Forest
- `GET /predict/{ticker}?horizon=5` — GARCH volatility forecast

## Known limitations

- Models are fit on-demand per API request rather than pre-trained and cached — acceptable at this project's scale (sub-second fit times), but wouldn't scale to high request volume without adding a model-caching layer.
- Evaluated on 4 large-cap tickers over a 2-year window — findings (especially the idiosyncratic-anomaly-timing result) would benefit from testing on a larger, more diverse universe of tickers.

## Status

✅ Complete — Weeks 1-3 (signal processing, modeling, productionization)