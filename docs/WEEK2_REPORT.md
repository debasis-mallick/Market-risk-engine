# Week 2 Report: Anomaly Detection & Volatility Forecasting

## Overview
Week 2 built and evaluated two families of models across 4 tickers
(AAPL, MSFT, GOOGL, TSLA): unsupervised anomaly detection and
volatility forecasting. Models were compared against each other and
against naive baselines, with hyperparameters tuned and logged.

## Anomaly Detection

Two unsupervised models were built and compared:

| Model | Approach | Top finding (AAPL) |
|---|---|---|
| Isolation Forest | Isolates points via random feature-space splits | Top anomaly: 2025-04-09 |
| Autoencoder | Flags high reconstruction error after compression | Same top anomaly: 2025-04-09 |

**Cross-validation:** both models agree on the same top anomalous day,
and their flagged anomalies cluster in the same April 2025 window —
which independently matches the volatility clustering event found in
Week 1's EDA (Day 3), Kalman filter residuals (Day 5), and wavelet
detail coefficients (Day 6). Five independent methods converging on
the same event is strong evidence of a genuine signal, not a modeling
artifact.

**Multi-ticker finding:** running Isolation Forest across all 4
tickers showed anomaly counts were consistent (~5% per ticker, as
configured), but the *timing* of each ticker's earliest anomaly did
NOT align (AAPL: Apr 2025, GOOGL: Dec 2024, MSFT: Jan 2025, TSLA: Mar
2025). This indicates the flagged anomalies are largely idiosyncratic
(company-specific), not a single shared market-wide shock.

**Hyperparameter sensitivity:** sweeping Isolation Forest's
`contamination` from 0.01 to 0.10 showed anomaly count scales roughly
linearly as expected, while average anomaly severity peaked around
0.05-0.08 before diluting at higher values — validating 0.05 as a
reasonable default.

## Volatility Forecasting

Three approaches were compared for next-day volatility forecasting:

| Model | Test RMSE (AAPL) | Notes |
|---|---|---|
| Naive (persistence) | 0.001370 | Tomorrow = today's volatility |
| GARCH(1,1) | N/A (in-sample) | alpha+beta = 0.947 (high persistence) |
| LSTM | 0.001516 | 10.6% WORSE than naive baseline |

**Key finding:** the LSTM underperformed the trivial persistence
baseline. This is an honest, informative result rather than a
failure — volatility in this dataset is highly persistent (confirmed
by GARCH's near-1.0 alpha+beta across ALL 4 tickers, ranging
0.93-0.99), leaving little exploitable structure beyond what naive
persistence already captures. With a relatively small training set
(~350 sequences), the LSTM likely could not learn reliable patterns
beyond this and may have partially fit noise.

**Model selection:** GARCH order tuning via AIC/BIC confirmed that
the simplest model, GARCH(1,1), was BIC-optimal among (1,1), (1,2),
(2,1), (2,2) — higher-order models offered negligible AIC improvement
that didn't justify their added complexity.

## Overall Takeaways

1. **Cross-method validation matters more than any single model's
   output.** The April 2025 AAPL event was independently confirmed by
   5 different techniques (EDA, ACF, Kalman, wavelets, Isolation
   Forest/Autoencoder) — this kind of convergence is far more
   trustworthy than any one method's result in isolation.

2. **Simpler models won on both tasks.** GARCH(1,1) beat higher-order
   GARCH variants by BIC, and naive persistence beat a neural network
   for volatility forecasting. This isn't a limitation of the
   project — it's a documented, realistic finding: model complexity
   must be justified by evidence of exploitable structure, not
   assumed.

3. **Findings don't always generalize the way you'd expect.**
   Volatility persistence was consistent across all 4 tickers, but
   anomaly timing was not — a reminder to explicitly test
   generalization rather than assuming a single-ticker result holds
   broadly.

## Next Steps (Week 3)
Productionize the pipeline: FastAPI service exposing model
predictions, containerization, automated testing, and CI.
