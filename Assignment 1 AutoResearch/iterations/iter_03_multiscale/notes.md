# Iteration 3: Multi-Scale Windows

## Diagnosis
Iter_02 improved with volatility/interaction features. But we only use 7-day windows.
A 3-day window captures recent mood swings; a 14-day window captures longer trends.
Multi-scale should give the model both short and long-term views.

## Hypothesis
Adding 3-day and 14-day window features alongside the 7-day will provide
multi-scale temporal context, improving both classification and regression.
Expected: +2-4% F1, +0.02-0.05 R2. Risk: feature explosion (3x more features)
but RF handles high dimensions well.

## Change
- Use window_sizes=[3, 7, 14] instead of [7]
- Keep volatility=True, interactions=True (from iter_02)
- NOTE: max_window is now 14, so fewer instances are created (need 14 days of history)
