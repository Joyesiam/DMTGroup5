# Iteration 138: Auto-ARIMA
**Category: Modeling**

## Source
Dennis-Dekker repo

## Hypothesis
ARIMA can capture temporal autocorrelation in mood time series directly, potentially outperforming ML models that treat each day independently.

## Change
Fit Auto-ARIMA per patient using walk-forward evaluation: fit on history, forecast 1 step ahead, expand window.

## Implementation
- `pmdarima.auto_arima()` per patient with automatic (p, d, q) selection
- Walk-forward: fit on all data up to time t, predict t+1, then expand
- Fall back to naive forecast if ARIMA fails to converge

Run via: `python scripts/run_v6_iterations.py --only 138`
