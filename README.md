# Real-Time Market Risk & Anomaly Detection Engine

## Problem
Market risk teams need to detect abnormal price/volume behavior and
forecast short-term volatility faster than manual review allows.
This project builds an end-to-end system that ingests market data,
decomposes signal from noise using filtering techniques, flags
statistical anomalies, and forecasts near-term volatility — served
via an API and live dashboard.

## Why this approach
- Kalman filtering / wavelet decomposition (signal processing,
  not just black-box ML) to separate trend from noise
- Isolation Forest / Autoencoder for anomaly detection
- GARCH + LSTM comparison for volatility forecasting
- Served as a production API (FastAPI), not just a notebook

## Status
🚧 Day 1/24 — Project scaffolding

## Architecture
See [docs/architecture.md](docs/architecture.md)

## Roadmap
See [docs/roadmap.md](docs/roadmap.md)
