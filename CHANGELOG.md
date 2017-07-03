# Changelog

## 0.19.0
- Support for `Float32x4Vector` class was added (from `dart_vector` library)
- Type `List` for label (target) list replaced with `Float32List` (in `Predictor.train()` and `Optimizer.optimize()`)

## 0.18.0
- class `Vector` and enum `Norm` were extracted to separate library (`https://github.com/gyrdym/dart_vector.git`)

## 0.17.0
- Common interface for loss function was added
- Derivative calculation was fixed (common canonical method was used)
- Squared loss function was added as a separate class

## 0.16.0
- `README.md` was actualized

## 0.15.0
- Tests for gradient optimizers were added
- Interfaces (almost for all entities) for DI and IOC mechanism were added
- `Randomizer` class was added
- Removed separate classes for k-fold cross validation and lpo cross validation, now it resides in `CrossValidation` class

## 0.14.0
- L1 and L2 regularization added

## 0.13.0
- Script for running all unit tests added

## 0.12.0
- Vector interface removed
- Regular vector implementation removed
- `TypedVector` -> `Vector`
- Implicit vectors constructing replaced with explicit `new`-instantiation

## 0.11.0
- Entity names correction

## 0.10.0
- K-fold cross validation added (`KFoldCrossValidation`)
- Leave P out cross validation added (`LpoCrossValidation`)
- `DataTrainTestSplitter` was removed

## 0.9.0
- `copy`, `fill` methods were added to `Vector`

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
