import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';
import 'package:ml_algo/src/model_selection/assessable.dart';
import 'package:ml_algo/src/regressor/linear_regressor_impl.dart';
import 'package:ml_algo/src/regressor/_helpers/squared_cost_optimizer_factory.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

/// A factory for all the linear regressors.
///
/// A typical linear regressor uses the equation of a line for multidimensional
/// space to make a prediction. Each `x` in the equation has its own coefficient
/// (weight) and the combination of these `x`-es and its coefficients gives the
/// `y` term. The latter is a thing, that the regressor should predict, and
/// as one knows all the `x` values (since it is the input for the algorithm),
/// the regressor should find the best coefficients (weights) (they are unknown)
/// to make a best prediction of `y` term.
abstract class LinearRegressor implements Assessable {
  /// Creates a gradient linear regressor. Uses gradient descent solver to
  /// find the optimal weights
  ///
  /// Parameters:
  ///
  /// [fittingData] A [DataFrame] with observations, that will be used by the
  /// regressor to learn coefficients of the predicting line
  ///
  /// [targetName] A string, that serves as a name of the target column (a
  /// column, that contains outcomes for the associated features)
  ///
  /// [iterationsLimit] A number of fitting iterations. Uses as a condition of
  /// convergence in the solver. Default
  /// value is `100`
  ///
  /// [initialLearningRate] A value, defining velocity of the convergence of the
  /// gradient descent solver.
  ///
  /// [minCoefficientsUpdate] A minimum distance between weights vectors in two
  /// subsequent iterations. Uses as a condition of convergence in the solver.
  /// In other words, if difference is small, there is no reason to continue
  /// fitting. Default value is `1e-12`
  ///
  /// [lambda] A coefficient for regularization. For this particular regressor
  /// there is only L2 regularization type.
  ///
  /// [randomSeed] A seed, that will be passed to a random value generator, used
  /// by stochastic optimizers. Will be ignored, if the solver is not
  /// stochastic. Remember, each time you run the regressor with the same
  /// parameters, you will receive a different result. To avoid it, define
  /// [randomSeed]
  ///
  /// [batchSize] A size of data (in rows), that will be used for fitting per
  /// one iteration.
  ///
  /// [fitIntercept] Whether or not to fit intercept term. Default value is
  /// `false`.
  ///
  /// [interceptScale] A value, defining a size of the intercept term
  ///
  /// [learningRateType] A value, defining a strategy for the learning rate
  /// behaviour throughout the whole fitting process
  ///
  /// [dtype] A data type for all the numeric values, used by the algorithm.
  /// Can affect performance or accuracy of the computations. Default value is
  /// [Float32x4]
  factory LinearRegressor(DataFrame fittingData, String targetName, {
    LinearOptimizerType optimizerType,
    int iterationsLimit = 100,
    LearningRateType learningRateType = LearningRateType.constant,
    InitialCoefficientsType initialCoefficientsType = InitialCoefficientsType.zeroes,
    double initialLearningRate = 1e-3,
    double minCoefficientsUpdate = 1e-12,
    double lambda,
    bool fitIntercept = false,
    double interceptScale = 1.0,
    int randomSeed,
    int batchSize = 1,
    Matrix initialCoefficients,
    bool isTrainDataNormalized = false,
    DType dtype = DType.float32,
  }) {
    final optimizer = createSquaredCostOptimizer(
      fittingData,
      targetName,
      optimizerType: optimizerType,
      iterationsLimit: iterationsLimit,
      initialLearningRate: initialLearningRate,
      minCoefficientsUpdate: minCoefficientsUpdate,
      lambda: lambda,
      randomSeed: randomSeed,
      batchSize: batchSize,
      learningRateType: learningRateType,
      initialCoefficientsType: initialCoefficientsType,
      fitIntercept: fitIntercept,
      interceptScale: interceptScale,
      isFittingDataNormalized: isTrainDataNormalized,
      dtype: dtype,
    );

    final coefficients = optimizer.findExtrema(
        initialCoefficients: initialCoefficients,
        isMinimizingObjective: true,
    ).getColumn(0);

    return LinearRegressorImpl(
      coefficients,
      targetName,
      fitIntercept: fitIntercept,
      interceptScale: interceptScale,
    );
  }

  /// Learned coefficients (or weights) for given features
  Vector get coefficients;

  bool get fitIntercept;

  double get interceptScale;
}
