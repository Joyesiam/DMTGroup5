# Iteration 107: App Grouping
**[Category: Feature Engineering]**

## Source
jorrimprins, emmaarussi

## Hypothesis
Grouping 12 app categories into 4 super-categories reduces sparsity and noise, improving model generalization.

## Change
Map 12 appCat columns into 4 super-categories: social_communication, entertainment_leisure, productivity_work, miscellaneous.

## Implementation
Aggregate durations of member categories into each super-category. Drop original fine-grained appCat columns after merging.

Run via: python scripts/run_v6_iterations.py --only 107
