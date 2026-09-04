# Anomaly Detection: Isolation Forest vs Autoencoder

## Summary
Both models were run on AAPL with the same 6 engineered features
(log_return, rolling volatility at 3 windows, z-scores at 2 windows),
each configured to flag ~5% of days as anomalous.

## Agreement
- Both models flag the same top anomalous day: 2025-04-09
- Both models concentrate the majority of flagged anomalies in the
  April 2025 window, corroborating the volatility clustering event
  identified independently via EDA (Day 3), Kalman filter residuals
  (Day 5), and wavelet detail coefficients (Day 6).

## Differences
- Isolation Forest's top anomalies are concentrated tightly around
  the sharpest crash days (April 3-14, 2025).
- The Autoencoder additionally flags several days in late April/May
  2025, suggesting it is more sensitive to the extended recovery
  period following the crash, not just the sharpest single-day moves.

## Interpretation
Isolation Forest isolates points based on how easily they're
separated via random feature-space splits — it tends to favor
extreme, sharply distinct values. The Autoencoder instead measures
how well a compressed representation can reconstruct the input —
it can flag a broader pattern of "doesn't fit the learned normal
structure," which may explain its slightly wider anomaly window.

Both approaches converge on the same core event, which is the more
important finding: cross-model agreement on a real anomaly gives
higher confidence than either model alone.
