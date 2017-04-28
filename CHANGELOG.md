# Changelog

## 0.9.0
- `copy`, `fill` methods were added to `VectorInterface`

## 0.8.0
- Reflection was removed for all cases (Vector instantiation, Optimizer instantiation)

## 0.7.0
- Abstract `Vector`-class was added as a base for typed and regular vector classes 

## 0.6.0
- Manhattan norm support was added

## 0.5.2
- `README` file was extended and clarified

## 0.5.1
- Random interval obtaining for the mini-batch gradient descent was fixed

## 0.5.0
- `BGDOptimizer`, `MBGDOptimizer` and `GradientOptimizer` were added

## 0.4.0
- `OptimizerInterface` was added
- Stochastic gradient descent optimizer was extracted from the linear regressor class
- Line separators changed for all files (CRLF -> LF)

## 0.3.1
- tests for `sum`, `abs`, `fromRange` methods of the `TypedVector` were added
- tests for `DataTrainTestSplitter` was added

## 0.3.0
- MAPE cost function was added

## 0.2.0
- SGD Regressor refactored (rmse on training removed, estimator added) + example extended

## 0.1.0
- Implementation of `-`, `*`, `/` operators and all vectors methods added to the `TypedVector`

## 0.0.1
- Initial version
