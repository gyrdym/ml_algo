# Changelog

## 16.0.4
- `README`: learning rate examples

## 16.0.3
- `dartfmt` applied to the project files

## 16.0.2
- `Retrainable`: returning type was fixed

## 16.0.1
- README updated according to null-safety changes
- All files from `lib` directory formatted by `dartfmt` tool

## 16.0.0
- Null-safety stable release

## 15.6.7
- `README`: important notes on data handling added

## 15.6.6
- `LogisticRegressor`, `SoftmaxRegressor`: redundant link function implementations removed

## 15.6.5
- `DecisionTreeTrainer`: redundant helper for trainer creation removed 

## 15.6.4
- `xrange` 1.0.0 supported

## 15.6.3
- `ml_dataframe` 0.4.0 supported
- README.md: example for flutter developers corrected

## 15.6.2
- More strict analyser options added

## 15.6.1
- README.md: example for flutter developers added

## 15.6.0
- Models retraining functionality added 

## 15.5.0
- `KnnClassifier`, `DecisionTreeClassifier`, `LogisticRegressor`, `SoftmaxRegressor`, `KnnRegressor`, `LinearRegressor`
    - hyperparameters added to the interfaces

## 15.4.1
- `DTypeJsonConverter` added
- `MatrixJsonConverter` added
- `VectorJsonConverter` added
- `DistanceTypeJsonConverter` added

## 15.4.0
- `KnnClassifier`:
    - serialization/deserialization functionality added with possibility to save the model into a json file
- `KnnRegressor`:
    - serialization/deserialization functionality added with possibility to save the model into a json file

## 15.3.6
- `ml_dataframe`: version 0.3.0 supported
- `README.md`: build badge corrected

## 15.3.5
- Github actions set up

## 15.3.4
- `DI logic`: 
    - conditional dependency registering added

## 15.3.3
- FUNDING.yml created

## 15.3.2
- Awfully long identifier `SequenceElementsDistributionCalculator` renamed to `DistributionCalculator`

## 15.3.1
- README: 
    - typos corrected
    - LogisticRegressor example corrected

## 15.3.0
- RSS metric added

## 15.2.4
- Documentation for classification metrics improved

## 15.2.3
- Documentation for RMSE metric improved

## 15.2.2
- Documentation for MAPE metric improved

## 15.2.1
- `classificationMetrics` constant list added
- `regressionMetrics` constant list added

## 15.2.0
- Recall metric added

## 15.1.0
- MAPE metric: output range squeezed to [0, 1]

## 15.0.1
- RegressorAssessor: unit tests added

## 15.0.0
- Breaking changes:
    - `CrossValidator`: 
        - `targetNames` argument removed
    - `Assessable`, `assess` method: `targetNames` argument removed
- Precision metric added
- Coordinate descent optimization logic fixed: dtype considered
- `LinearClassifier`:
    - `classNames` property replaced with `targetNames` property in `Predictor`

## 14.2.6
- `injector` lib 1.0.9 supported

## 14.2.5
- `pubspec`:
    - `injector` dependency corrected

## 14.2.4
- `README`:
    - File path note for flutter developers added

## 14.2.3
- `README`:
    - Kfold constructor renamed to kFold
    - brackets removed from LogisticRegressor constructor arguments
    - file path note added

## 14.2.2
- `ml_dataframe` 0.2.0 supported

## 14.2.1
- `README`: Examples on prediction and collecting learning data added

## 14.2.0
- `SoftmaxRegressor`:
    - `Default constructor`: `collectLearningData` parameter added

## 14.1.1
- `README`: Advanced usage example on Logistic regression added

## 14.1.0
- `Model selection`: `splitData` helper added

## 14.0.1
- data splitters renamed and reorganized

## 14.0.0
- Breaking change:
    - `CrossValidator`: `evalute` method's api changed, it returns a Future resolving with scores Vector now instead 
    of a double value

## 13.10.0
- `LinearRegressor`:
    - `Default constructor`: `collectLearningData` parameter added

## 13.9.0
- `LogisticRegressor`:
    - `Default constructor`: `collectLearningData` parameter added

## 13.8.1
- `ml_dataframe` dependency updated
- `xrange` dependency constrain removed

## 13.8.0
- `LinkFunction`:
    - `Float64InverseLogitLinkFunction` added
    - `Float64SoftmaxLinkFunction` added

## 13.7.0
- `LinearRegressor`: serialization/deserialization functionality added with possibility to save the model into a file as json

## 13.6.0
- `SoftmaxRegressor`: serialization/deserialization functionality added with possibility to save the model into a file as json

## 13.5.1
- `DecisionTreeClassifier`: documentation added for `fromJson` constructor

## 13.5.0
- `LogisticRegressor`: serialization/deserialization functionality added with possibility to save the model into a file as json

## 13.4.0
- `DecisionTreeClassifier`: serialization/deserialization functionality added with possibility to save the model into a file as json

## 13.3.7
- `TreeLeafLabel`: probability validation improvements

## 13.3.6
- `DecisionTreeClassifier`: classifier instantiating refactored
- `TreeSolver`: DI support added

## 13.3.5
- `SoftmaxRegressor`: classifier instantiating refactored

## 13.3.4
- `LogisticRegressor`: classifier instantiating refactored

## 13.3.3
- `KnnClassifierImpl`: unit tests for `predictProbability` method added

## 13.3.2
- `KnnClassifier`: classifier instantiating refactored

## 13.3.1
- `readme`: KnnRegressor usage example fixed 

## 13.3.0
- `KnnClassifier` class added

## 13.2.0
- `KNN algorithm`: standardization for distance added
- `KnnRegressor`: 
    - default kernel changed to gaussian
    - `k` parameter is required now

## 13.1.1
- `KNN regression`: documentation for kernel function types added
- `KnnRegressor`: finding weighted average using kernel function fixed

## 13.1.0
- `CrossValidator`: `onDataSplit` hook added

## 13.0.0
- Predictor's API: `DataFrame` used instead of `Matrix`
- `DecisionTreeSolver`: data splitting logic fixed

## 12.1.2
- `xrange` package version locked

## 12.1.1
- `ml_linalg` 11.0.0 supported
- `Unit tests`: `iterable2dAlmostEqualTo` and `iterableAlmostEqualTo` matchers used from `ml_tech`

## 12.1.0
- Decision tree classifier added

## 12.0.2
- `ScoreToProbMapperFactory` removed
- `ScoreToProbMapperType` enum removed
- `ScoreToProbMapper`: the entity renamed to `LinkFunction`

## 12.0.1
- Cost function factory removed
- Cost function type removed

## 12.0.0
- Breaking change: GradientType enum removed
- Breaking change: OptimizerType enum removed
- Breaking change, `Predictor`: `fit` method removed, fitting is happening while a model is being created
- Breaking change, `Predictor`: interface replaced with `Assessable`, redundant properties removed
- Breaking change: `LinearClassifier` reorganized
- Optimizers now have immutable state
- `InterceptPreprocessor` replaced with a helper function `addInterceptIf`

## 11.0.1
- Cross validator refactored
- Data splitters refactored
- Unit tests for cross validator added

## 11.0.0
- Added immutable state to all the predictor subclasses

## 10.3.0
- kernels added:
    - uniform
    - epanechnikov
    - cosine
    - gaussian
- `NoNParametricRegressor.nearestNeighbour`: added possibility to specify the kernel function

## 10.2.1
- test coverage restored

## 10.2.0
- `NoNParametricRegressor` class added
- `KNNRegressor` class added
- `ml_linalg` v9.0.0 supported

## 10.1.0
- `ml_linalg` v7.0.0 support

## 10.0.0
- Data preprocessing: all the entities moved to separate repo - [ml_preprocessing](https://github.com/gyrdym/ml_preprocessing)  

## 9.2.4
- Data preprocessing: All categorical values are now converted to String type  

## 9.2.3
- Examples for Linear regression and Logistic regression updated (vector's `normalize` method used)
- `CategoricalDataEncoderType`: one-hot encoding documentation corrected

## 9.2.2
- Softmax regression example added to README

## 9.2.1
- README corrected

## 9.2.0
- `LinearClassifier.logisticRegressor`: numerical stability improved
- `LinearClassifier.logisticRegressor`: `probabilityThreshold` parameter added
- `DataFrame.fromCsv`: parameter `fieldDelimiter` added

## 9.1.0
- `DataFrame`: `labelName` parameter added

## 9.0.0
- `ml_linalg` v6.0.2 supported
- `Classifier`: type of `weightsByClasses` changed from `Map` to `Matrix` 
- `SoftmaxRegressor`: more detailed unit tests for softmax regression added
- Data preprocessing: `DataFrame` introduced (former `MLData`)

## 8.0.0
- `LinearClassifier.softmaxRegressor` implemented
- `Metric` interface refactored (`getError` renamed to `getScore`)

## 7.2.0
- `SoftmaxMapper` added (aka Softmax activation function)

## 7.1.0
- `ConvergenceDetector` added (this entity stops the optimizer when it is needed)

## 7.0.0
- All the exports packed into `ml_algo` entry

## 6.2.0
- Coefficients in optimizers now are a matrix
- InitialWeightsGenerator instantiating fixed: dtype is passed now 

## 6.1.0
- `LinkFunction` renamed to `ScoreToProbMapper`
- `ScoreToProbMapper` accepts vector and returns vector instead of a scalar

## 6.0.6
- Pedantic package integration added
- Some linter issues fixed

## 6.0.5
- Coveralls integration added
- dartfm check task added

## 6.0.4
- Documentation for linear regression corrected
- Documentation for `MLData` corrected

## 6.0.3
- Documentation for logistic regression corrected

## 6.0.2
- Tests corrected: removed import `test_api.dart`

## 6.0.1
- Readme corrected

## 6.0.0
- Library fully refactored:
    - add possibility to set certain data type for numeric computations
    - all algorithms now are more generic
    - a lot of unit tests added
    - bug fixes 

## 5.2.0
- Ordinal encoder added
- `Float32x4CsvMlData` significantly extended

## 5.1.0
- Real-life example added (black friday dataset)
- `rows` parameter added to `Float32x4CsvMlData`
- Unknown categorical values handling strategy types added

## 5.0.0
- One hot encoder integrated into CSV ML data 

## 4.3.3
- Performance test for one hot encoder added

## 4.3.2
- One hot encoder implemented

## 4.3.1
- enum for categorical data encoding added

## 4.3.0
- Cross validator factory added
- README updated

## 4.2.0
- csv-parser added

## 4.1.0
- `ml_linalg` removed from export file
- README refreshed
- General `datasets` directory created

## 4.0.0
- `ml_linal` ^4.0.0 supported

## 3.5.4
- README.md updated
- build_runner dependency updated

## 3.5.3
- `dartfmt` tool applied to all necessary files

## 3.5.2
- Travis configuration file name corrected 

## 3.5.1
- Travis integration added

## 3.5.0
- Vectorized cost functions applied

## 3.4.0
- `ml_linalg` 2.0.0 supported

## 3.3.0
- Matrix-based gradient calculation added for log likelihood cost function

## 3.2.0
- Matrix-based gradient calculation added for squared cost function

## 3.1.2
- Description corrected

## 3.1.1
- `dartfm` tool applied

## 3.1.0
- Get rid of MLVector's deprecated methods

## 3.0.0
- Library public release

## 2.0.0
- `ml_linalg` supported

## 1.2.1
- subVector -> subvector

## 1.2.0
- Matrices support added

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
