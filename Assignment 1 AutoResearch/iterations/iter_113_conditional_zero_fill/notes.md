# Iteration 113: Conditional Zero Fill
**[Category: Data Cleaning]**

## Source
iustkuipers

## Hypothesis
NaN in appCat/call/sms likely means zero usage if the patient was active that day. Filling conditionally avoids introducing false zeros on truly missing days.

## Change
For appCat/call/sms NaN: only fill with 0 if the patient was active that day (>= 4 other columns non-null). Otherwise leave as NaN.

## Implementation
Per row, count non-null columns. If count >= 4, fill appCat/call/sms NaN with 0. Otherwise leave NaN for downstream imputation.

Run via: python scripts/run_v6_iterations.py --only 113
