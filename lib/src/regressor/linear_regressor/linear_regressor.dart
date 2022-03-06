import 'package:ml_algo/src/common/constants/default_parameters/common.dart';
import 'package:ml_algo/src/common/constants/default_parameters/coordinate_optimization.dart';
import 'package:ml_algo/src/common/constants/default_parameters/gradient_optimization.dart';
import 'package:ml_algo/src/common/constants/default_parameters/linear_optimization.dart';
import 'package:ml_algo/src/common/serializable/serializable.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';
import 'package:ml_algo/src/linear_optimizer/regularization_type.dart';
import 'package:ml_algo/src/model_selection/assessable.dart';
import 'package:ml_algo/src/predictor/predictor.dart';
import 'package:ml_algo/src/predictor/retrainable.dart';
import 'package:ml_algo/src/regressor/linear_regressor/_init_module.dart';
import 'package:ml_algo/src/regressor/linear_regressor/linear_regressor_factory.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

/// Linear regression
///
/// A typical linear regressor uses the equation of a line for multidimensional
/// space to make a prediction. Each `x` in the equation has its own dedicated
/// coefficient (weight) and the combination of these `x`-es and its dedicated
/// coefficients gives the `y` term (outcome). The latter is the value that the
/// regressor should predict, and since all the `x` values are known (they
/// are the input for the algorithm), the regressor should find the best
/// coefficients (weights) for each `x`-es to make a best prediction of `y` term.
abstract class LinearRegressor
    implements
        Assessable,
        Serializable,
        Retrainable<LinearRegressor>,
        Predictor {
  /// Parameters:
  ///
  /// [fittingData] A [DataFrame] with observations that is used by the
  /// regressor to learn coefficients of the predicting hyperplane. Must contain
  /// [targetName] column.
  ///
  /// [targetName] A string that serves as a name of the target column
  /// containing observation labels.
  ///
  /// [optimizerType] Defines an algorithm of optimization that will be used
  /// to find the best coefficients. Also defines which regularization type
  /// (L1 or L2) one may use to learn a linear regressor. By default -
  /// [LinearOptimizerType.closedForm].
  ///
  /// [iterationsLimit] A number of fitting iterations. Uses as a condition of
  /// convergence in the optimization algorithm. Default value is `100`.
  ///
  /// [initialLearningRate] The initial value defining velocity of the convergence of
  /// gradient descent-based optimizers. Default value is `1e-3`.
  ///
  /// [decay] The value meaning "speed" of learning rate decrease. Applicable only
  /// for [LearningRateType.timeBased], [LearningRateType.stepBased], and
  /// [LearningRateType.exponential] strategies
  ///
  /// [dropRate] The value that is used as a number of learning iterations after
  /// which the learning rate will be decreased. The value is applicable only for
  /// [LearningRateType.stepBased] learning rate; it will be omitted for other
  /// learning rate strategies
  ///
  /// [minCoefficientsUpdate] A minimum distance between coefficient vectors in
  /// two contiguous iterations. Uses as a condition of convergence in the
  /// optimization algorithm. If difference between the two vectors is small
  /// enough, there is no reason to continue fitting. Default value is `1e-12`
  ///
  /// [lambda] A coefficient of regularization. Uses to prevent the regressor's
  /// overfitting. The more the value of [lambda], the more regular the
  /// coefficients of the equation of the predicting hyperplane are. Extremely
  /// large [lambda] may decrease the coefficients to nothing, otherwise too
  /// small [lambda] may be a cause of too large absolute values of the
  /// coefficients.
  ///
  /// [regularizationType] A way the coefficients of the regressor will be
  /// regularized to prevent the model's overfitting.
  ///
  /// [randomSeed] A seed value that will be passed to a random value generator,
  /// used by stochastic optimizers. Will be ignored, if the solver is not
  /// stochastic. Remember, each time you run the stochastic regressor with the
  /// same parameters but with unspecified [randomSeed], you will receive
  /// different results. To avoid it, define [randomSeed].
  ///
  /// [batchSize] A size of data (in rows) that will be used for fitting per
  /// one iteration. Applicable not to all optimizers. If gradient-based
  /// optimizer is used and if [batchSize] == `1`, stochastic mode will be
  /// activated; if `1` < [batchSize] < `total number of rows`, mini-batch mode
  /// will be activated; if [batchSize] == `total number of rows`, full-batch
  /// mode will be activated.
  ///
  /// [fitIntercept] Whether or not to fit intercept term. Default value is
  /// `false`. Intercept in 2-dimensional space is a bias of the line (relative
  /// to X-axis).
  ///
  /// [interceptScale] A value defining a size of the intercept.
  ///
  /// [isFittingDataNormalized] Defines whether the [fittingData] is normalized
  /// or not. Normalization should be performed column-wise. Normalized data
  /// may be required by some optimizers (e.g., for
  /// [LinearOptimizerType.coordinate]).
  ///
  /// [learningRateType] A value defining a strategy for the learning rate
  /// behaviour throughout the whole fitting process.
  ///
  /// [initialCoefficientsType] Defines the coefficients that will be
  /// autogenerated at the first iteration of optimization. By default,
  /// all the autogenerated coefficients are equal to zeroes at the start.
  /// If [initialCoefficients] are provided, the parameter will be ignored.
  ///
  /// [initialCoefficients] Coefficients to be used during the first iteration of
  /// optimization algorithm. [initialCoefficients] should have length that is
  /// equal to the number of features in the [fittingData].
  ///
  /// [collectLearningData] Whether or not to collect learning data, for
  /// instance cost function value per each iteration. Affects performance much.
  /// If [collectLearningData] is true, one may access [costPerIteration]
  /// getter in order to evaluate learning process more thoroughly.
  ///
  /// [dtype] A data type for all the numeric values, used by the algorithm. Can
  /// affect performance or accuracy of the computations. Default value is
  /// [DType.float32].
  factory LinearRegressor(
    DataFrame fittingData,
    String targetName, {
    LinearOptimizerType optimizerType = LinearOptimizerType.closedForm,
    int iterationsLimit = iterationLimitDefaultValue,
    LearningRateType learningRateType = learningRateTypeDefaultValue,
    InitialCoefficientsType initialCoefficientsType =
        initialCoefficientsTypeDefaultValue,
    double initialLearningRate = initialLearningRateDefaultValue,
    double decay = decayDefaultValue,
    int dropRate = dropRateDefaultValue,
    double minCoefficientsUpdate = minCoefficientsUpdateDefaultValue,
    double lambda = lambdaDefaultValue,
    bool fitIntercept = fitInterceptDefaultValue,
    double interceptScale = interceptScaleDefaultValue,
    int batchSize = batchSizeDefaultValue,
    bool isFittingDataNormalized = isFittingDataNormalizedDefaultValue,
    bool collectLearningData = collectLearningDataDefaultValue,
    DType dtype = dTypeDefaultValue,
    RegularizationType? regularizationType,
    int? randomSeed,
    Matrix? initialCoefficients,
  }) =>
      initLinearRegressorModule().get<LinearRegressorFactory>().create(
            fittingData: fittingData,
            targetName: targetName,
            optimizerType: optimizerType,
            iterationsLimit: iterationsLimit,
            learningRateType: learningRateType,
            initialCoefficientsType: initialCoefficientsType,
            initialLearningRate: initialLearningRate,
            decay: decay,
            dropRate: dropRate,
            minCoefficientsUpdate: minCoefficientsUpdate,
            lambda: lambda,
            regularizationType: regularizationType,
            fitIntercept: fitIntercept,
            interceptScale: interceptScale,
            randomSeed: randomSeed,
            batchSize: batchSize,
            initialCoefficients: initialCoefficients,
            isFittingDataNormalized: isFittingDataNormalized,
            collectLearningData: collectLearningData,
            dtype: dtype,
          );

  /// Lasso regression
  ///
  /// Lasso regression is a kind of linear regression which uses L1 (Lasso)
  /// regularization.
  ///
  /// Coordinate descent is used as an optimization algorithm for this particular
  /// implementation of Lasso regression.
  ///
  /// [trainData] A [DataFrame] with observations that is used by the
  /// regressor to learn coefficients of the predicting hyperplane. Must contain
  /// [targetName] column.
  ///
  /// [targetName] A string that serves as a name of the target column
  /// containing dependent variables
  ///
  /// [iterationLimit] A number of fitting iterations. Uses as a condition of
  /// convergence in the optimization algorithm. Default value is `100`.
  ///
  /// [minCoefficientUpdate] A minimum distance between coefficient vectors in
  /// two contiguous iterations. Uses as a condition of convergence in the
  /// optimization algorithm. If difference between the two vectors is small
  /// enough, there is no reason to continue fitting. Default value is `1e-12`
  ///
  /// [lambda] L1 (Lasso) regularization coefficient using for feature selection.
  /// The greater the value of [lambda], the stricter feature selection is.
  ///
  /// [fitIntercept] Whether or not to fit intercept term. Default value is
  /// `false`. Intercept in 2-dimensional space is a bias of the line (relative
  /// to X-axis).
  ///
  /// [interceptScale] A value defining a size of the intercept.
  ///
  /// [isDataNormalized] Defines whether the [trainData] is normalized
  /// or not. Normalization should be performed column-wise.
  ///
  /// [initialCoefficientType] Initial coefficients generation way.
  /// If [initialCoefficients] are provided, the parameter will be ignored.
  ///
  /// [initialCoefficients] Coefficients to be used during the first iteration of
  /// the optimization algorithm. [initialCoefficients] should have length that is
  /// equal to the number of features in the [trainData].
  ///
  /// [collectLearningData] Whether or not to collect learning data, for
  /// instance cost function value per each iteration. Affects performance much.
  /// If [collectLearningData] is true, one may access [costPerIteration]
  /// getter in order to evaluate learning process more thoroughly.
  ///
  /// [dtype] A data type for all the numeric values, used by the algorithm. Can
  /// affect performance or accuracy of the computations. Default value is
  /// [DType.float32].
  factory LinearRegressor.lasso(DataFrame trainData, String targetName,
          {int iterationLimit = iterationLimitDefaultValue,
          InitialCoefficientsType initialCoefficientType =
              initialCoefficientsTypeDefaultValue,
          double minCoefficientUpdate = minCoefficientsUpdateDefaultValue,
          double lambda = lambdaDefaultValue,
          bool fitIntercept = fitInterceptDefaultValue,
          double interceptScale = interceptScaleDefaultValue,
          bool isDataNormalized = isFittingDataNormalizedDefaultValue,
          bool collectLearningData = collectLearningDataDefaultValue,
          DType dtype = dTypeDefaultValue,
          Matrix? initialCoefficients}) =>
      initLinearRegressorModule().get<LinearRegressorFactory>().create(
            fittingData: trainData,
            targetName: targetName,
            optimizerType: LinearOptimizerType.coordinate,
            iterationsLimit: iterationLimit,
            learningRateType: defaultLearningRateType,
            initialCoefficientsType: initialCoefficientType,
            initialLearningRate: initialLearningRateDefaultValue,
            decay: decayDefaultValue,
            dropRate: dropRateDefaultValue,
            minCoefficientsUpdate: minCoefficientUpdate,
            lambda: lambda,
            fitIntercept: fitIntercept,
            interceptScale: interceptScale,
            batchSize: 0,
            initialCoefficients: initialCoefficients,
            isFittingDataNormalized: isDataNormalized,
            collectLearningData: collectLearningData,
            dtype: dtype,
          );

  /// Restores previously fitted [LinearRegressor] instance from the [json]
  ///
  ///
  /// ````dart
  /// import 'dart:io';
  /// import 'package:ml_dataframe/ml_dataframe.dart';
  ///
  /// final data = <Iterable>[
  ///   ['feature 1', 'feature 2', 'feature 3', 'outcome']
  ///   [        5.0,         7.0,         6.0,      98.0],
  ///   [        1.0,         2.0,         3.0,      10.0],
  ///   [       10.0,        12.0,        31.0,    -977.0],
  ///   [        9.0,         8.0,         5.0,       0.0],
  ///   [        4.0,         0.0,         1.0,       6.0],
  /// ];
  /// final targetName = 'outcome';
  /// final samples = DataFrame(data, headerExists: true);
  /// final regressor = LinearRegressor(
  ///   samples,
  ///   targetName,
  ///   iterationsLimit: 2,
  ///   learningRateType: LearningRateType.constant,
  ///   initialLearningRate: 1.0,
  ///   batchSize: 5,
  ///   fitIntercept: true,
  ///   interceptScale: 3.0,
  /// );
  ///
  /// final pathToFile = './classifier.json';
  ///
  /// await regressor.saveAsJson(pathToFile);
  ///
  /// final file = File(pathToFile);
  /// final json = await file.readAsString();
  /// final restoredRegressor = LinearRegressor.fromJson(json);
  ///
  /// // here you can use previously fitted restored regressor to make
  /// // some prediction, e.g. via `restoredRegressor.predict(...)`;
  /// ````
  factory LinearRegressor.fromJson(String json) =>
      initLinearRegressorModule().get<LinearRegressorFactory>().fromJson(json);

  /// Optimization algorithm that was used to learn the model's coefficients
  LinearOptimizerType get optimizerType;

  /// A maximum number of optimization iterations that was used
  /// during model's coefficients learning
  int get iterationsLimit;

  /// Learning rate update strategy that was used to learn the model's
  /// coefficients
  LearningRateType get learningRateType;

  /// Coefficients generator type that was used at the very first optimization
  /// iteration during the model's coefficients learning
  InitialCoefficientsType get initialCoefficientsType;

  /// Initial learning rate value of chosen optimization algorithm
  ///
  /// The value is read-only, it's a hyperparameter of the model
  num get initialLearningRate;

  /// A value that was used for the learning rate decay
  ///
  /// The value is read-only, it's a hyperparameter of the model
  num get decay;

  /// A value that was used for the learning rate drop rate
  ///
  /// The value is read-only, it's a hyperparameter of the model
  int get dropRate;

  /// A coefficients update value that was used as a stop criteria during the
  /// model's coefficients learning process
  num get minCoefficientsUpdate;

  /// A regularization value that was used to prevent overfitting of the model
  num get lambda;

  /// A regularization strategy that was used to prevent overfitting of the
  /// model
  RegularizationType? get regularizationType;

  /// A value that was used during the model's coefficients learning stage to
  /// init the randomizer for a stochastic optimizer (if the latter was chosen
  /// to learn the model's coefficients)
  int? get randomSeed;

  /// A size of a batch of data that was used in a single iteration of the
  /// optimization algorithm
  int get batchSize;

  /// Coefficients that were used at the very first optimization iteration
  /// during the model's coefficients learning stage
  Matrix? get initialCoefficients;

  /// Was the train data normalized or not prior to the model's coefficients
  /// learning stage
  bool get isFittingDataNormalized;

  /// Learned coefficients (or weights) for given features
  Vector get coefficients;

  /// A string that serves as a name of the target column containing
  /// observation labels. Uses in a predicted dataframe returning from [predict]
  /// method
  String get targetName;

  /// A flag denoting whether the intercept term is considered during
  /// learning of the regressor or not
  bool get fitIntercept;

  /// A value defining a size of the intercept if [fitIntercept] is
  /// `true`
  num get interceptScale;

  /// Returns a list of cost values per each learning iteration. Returns null
  /// if the parameter `collectLearningData` of the default constructor is false
  List<num>? get costPerIteration;
}
