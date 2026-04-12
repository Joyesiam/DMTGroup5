# Task 5A: Characteristics of MSE and MAE

## Formulae

**Mean Squared Error (MSE):**

MSE = (1/n) * SUM_{i=1}^{n} (y_i - y_hat_i)^2

**Mean Absolute Error (MAE):**

MAE = (1/n) * SUM_{i=1}^{n} |y_i - y_hat_i|

where y_i is the true value, y_hat_i is the predicted value, and n is the number
of observations.

## When to Use One Over the Other

**MSE** penalizes large errors quadratically. An error of 2 contributes 4 to MSE,
while an error of 1 contributes only 1 -- so a single large error is penalized
4x as much as two small errors of the same total magnitude. This makes MSE
suitable when:
- Large errors are disproportionately costly (e.g., predicting structural loads)
- The data has few outliers and errors are approximately normally distributed
- Gradient-based optimization is used (MSE is differentiable everywhere)

**MAE** penalizes all errors linearly. An error of 2 contributes exactly 2x as
much as an error of 1. This makes MAE suitable when:
- Robustness to outliers is important (e.g., house price prediction with
  occasional extreme values)
- All errors matter equally regardless of size
- The target distribution is skewed or heavy-tailed

**Key difference:** MSE optimizes toward the conditional mean of the target
distribution, while MAE optimizes toward the conditional median. When the
distribution is symmetric, these are identical; when skewed, they differ.

## Example Where MSE and MAE Give Identical Results

Consider a dataset with 4 observations where the prediction errors are:

e_1 = +c, e_2 = -c, e_3 = +c, e_4 = -c

for some constant c > 0. That is, all errors have the same absolute magnitude c.

Then:
- MSE = (1/4) * (c^2 + c^2 + c^2 + c^2) = c^2
- MAE = (1/4) * (c + c + c + c) = c

More generally, MSE = MAE^2 when all errors have the same absolute value.

**Concrete example:** Predict mood for 4 patients. True values: [6, 7, 8, 7].
Predicted values: [7, 6, 7, 8]. Errors: [+1, -1, -1, +1].

- MSE = (1 + 1 + 1 + 1) / 4 = 1.0
- MAE = (1 + 1 + 1 + 1) / 4 = 1.0

Both metrics give exactly 1.0 because every individual error has the same
absolute magnitude (|e_i| = 1 for all i). This equality holds whenever the
error distribution is concentrated at a single absolute value, because
squaring a constant c gives c^2, and when c = 1, both MSE and MAE equal 1.

**Mathematical justification:** MSE = MAE when (1/n) * SUM(e_i^2) = ((1/n) * SUM(|e_i|)).
This simplifies to E[e^2] = (E[|e|])^2, which by Jensen's inequality holds
if and only if |e| is constant (i.e., all errors have the same absolute magnitude).
This is because Jensen's inequality states E[f(X)] >= f(E[X]) for convex f,
with equality if and only if X is constant. Here f(x) = x^2, which is strictly
convex, so equality requires |e_i| = c for all i.
