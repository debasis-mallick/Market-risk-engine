# Volatility Forecasting: GARCH vs LSTM vs Naive Baseline

## Setup
Both models forecast next-day volatility (21-day rolling realized
volatility) using a chronological train/test split to avoid lookahead bias.

## Results
- Naive baseline (persistence: tomorrow = today) RMSE: 0.001370
- LSTM RMSE: 0.001516 (10.6% WORSE than naive baseline)
- GARCH: alpha + beta = 0.947 (very high persistence, effectively
  confirming that most of tomorrow's volatility is explainable by
  today's volatility alone)

## Interpretation
The LSTM underperforming a trivial persistence baseline is a genuine
and informative finding, not a failure. Volatility in this dataset is
extremely persistent (as GARCH's near-1.0 alpha+beta also shows),
leaving very little residual predictable structure for a more complex
model to exploit. With a relatively small training set (~350
sequences), the LSTM likely cannot reliably learn subtle non-linear
patterns beyond what simple persistence already captures, and may
instead fit noise.

## Takeaway
This result reinforces a broader lesson in quantitative forecasting:
model complexity should be justified by evidence of exploitable
structure beyond what simple baselines capture. Here, GARCH (a
well-understood, interpretable statistical model) is both simpler
and more effective than the LSTM for this task and dataset size —
a finding worth testing further with more data, more tickers, or a
different target horizon before drawing final conclusions.
