# Iteration 44: drop_sparse

## Hypothesis
Dropping 7 sparse app categories (>80% missing) removes noise features.

## Change
drop_sparse=True. Removes appCat.weather/game/finance/unknown/office/travel/utilities.

## Config (non-default parameters)
- drop_sparse = True
- split_method = leave_patients_out
