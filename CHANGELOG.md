# Changelog

## 1.1.1
- Examples fixed, dependencies fixed

## 1.1.0
- Support of updated `linalg` package

## 1.0.1
- Readme updated, dependencies fixed

## 1.0.0
- Migration to dart 2.0

## 0.38.1
- [issue](https://github.com/gyrdym/dart_ml/issues/41) solved

## 0.38.0
- Lasso solution refactored

## 0.37.0
- Support of linalg package (former simd_vector) 

## 0.36.0
- Intercept term considered (`fitIntercept` and `interceptScale` parameters)

## 0.35.1
- Logistic regression tests improved

## 0.35.0
- `One versus all` refactored, tests for logistic regression added 

## 0.34.0
- One versus all classifier

## 0.33.0
- Gradient descent regressor type enum added

## 0.32.1
- Gradient optimizer unit tests

## 0.32.0
- Get rid of derivative computation

## 0.31.0
- Get rid of di package usage

## 0.30.1
- File structure flattened

## 0.30.0
- Redundant gradient optimizers removed

## 0.29.0
- `part ... part of` directives removed

## 0.28.0
- Coordinate descent optimizer added
- Lasso regressor added

## 0.27.0
- Gradient calculation changed

## 0.26.1
- Code was optimized (removed unnecessary)
- Refactoring

## 0.26.0
- More distinct modularity was added to the library
- Unit tests were fixed

## 0.25.0
- Tests for gradient optimizers were added
- Gradient calculator was created as a separate entity
- Initial weights generator was created as a separate entity
- Learning rate generator was created as a separate entity

## 0.24.0
- All implementations were hidden

## 0.23.0
- `findMaxima` and `findMinima` methods were added to `Optimizer` interface

## 0.22.0
- File structure reorganized, predictor classes refactored
- `README.md` updated

## 0.21.0
- Logistic regression model added (with example)

## 0.20.2
- `README.md` updated

## 0.20.1
- `simd_vector` dependency url fixed

## 0.20.0
- Repository dependency corrected (dart_vector -> simd_vector)

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
