# Iteration 15: Leave-Patients-Out Split

## Hypothesis
Holding out 5 complete patients tests whether the model generalizes to
unseen individuals. This is more realistic for deployment (new app users).
Performance will likely drop vs chronological split because the model
cannot learn individual mood baselines for holdout patients.

## Change
- split_method="leave_patients_out", n_holdout_patients=5
- Everything else same as iter_07 baseline
