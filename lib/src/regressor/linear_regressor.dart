import 'package:ml_algo/src/helpers/features_target_split.dart';
import 'package:ml_algo/src/model_selection/assessable.dart';
import 'package:ml_algo/src/solver/linear/gradient/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/solver/linear/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_algo/src/regressor/coordinate_regressor.dart';
import 'package:ml_algo/src/regressor/gradient_regressor.dart';
import 'package:ml_algo/src/regressor/regressor.dart';
import 'package:ml_algo/src/utils/parameter_default_values.dart';
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
abstract class LinearRegressor implements Regressor, Assessable {
  /// Creates a gradient linear regressor. Uses gradient descent solver to
  /// find the optimal weights
  ///
  /// Parameters:
  ///
  /// [fittingData] A [DataFrame] with observations, that will be used by the
  /// regressor to learn coefficients of the predicting line
  ///
  /// [targetIndex] An index of the target column (a column, that contains
  /// outcomes for the associated features)
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
  /// [minWeightsUpdate] A minimum distance between weights vectors in two
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
  factory LinearRegressor.gradient(DataFrame fittingData, {
    int targetIndex,
    String targetName,
    int iterationsLimit = ParameterDefaultValues.iterationsLimit,
    LearningRateType learningRateType = LearningRateType.constant,
    InitialWeightsType initialWeightsType = InitialWeightsType.zeroes,
    double initialLearningRate = ParameterDefaultValues.initialLearningRate,
    double minWeightsUpdate = ParameterDefaultValues.minCoefficientsUpdate,
    double lambda,
    bool fitIntercept = false,
    double interceptScale = 1.0,
    int randomSeed,
    int batchSize = 1,
    Matrix initialWeights,
    DType dtype,
  }) {
    final featuresTargetSplits = featuresTargetSplit(fittingData)
        .toList();

    return GradientRegressor(
      featuresTargetSplits[0].toMatrix(),
      featuresTargetSplits[1].toMatrix(),
      iterationsLimit: iterationsLimit,
      learningRateType: learningRateType,
      initialWeightsType: initialWeightsType,
      initialLearningRate: initialLearningRate,
      minWeightsUpdate: minWeightsUpdate,
      lambda: lambda,
      fitIntercept: fitIntercept,
      interceptScale: interceptScale,
      randomSeed: randomSeed,
      batchSize: batchSize,
      initialWeights: initialWeights,
      dtype: dtype,
    );
  }

  /// Creates a linear regressor with a coordinate descent solver with
  /// possibility of using L1 regularization.
  ///
  /// L1 regularization allows to select more important features, making
  /// coefficients of less important ones zeroes.
  ///
  /// Parameters:
  ///
  /// [fittingData] A [DataFrame] with observations, that will be used by the
  /// regressor to learn coefficients of the predicting line
  ///
  /// [targetIndex] An index of the target column (a column, that contains
  /// outcomes for the associated features)
  ///
  /// [targetName] A string, that serves as a name of the target column (a
  /// column, that contains outcomes for the associated features)
  ///
  /// [iterationsLimit] A number of fitting iterations. Uses as a condition of
  /// convergence in the solver. Default value is `100`
  ///
  /// [minWeightsUpdate] A minimum distance between weights vectors in two
  /// subsequent iterations. Uses as a condition of convergence in the solver.
  /// In other words, if difference is small, there is no reason to continue
  /// fitting. Default value is `1e-12`
  ///
  /// [lambda] A coefficient for regularization. For this particular regressor
  /// there is only L1 regularization type.
  ///
  /// [fitIntercept] Whether or not to fit intercept term. Default value is
  /// `false`.
  ///
  /// [interceptScale] A value, defining a size of the intercept term
  ///
  /// [dtype] A data type for all the numeric values, used by the algorithm.
  /// Can affect performance or accuracy of the computations. Default value is
  /// [Float32x4]
  factory LinearRegressor.coordinate(DataFrame fittingData, {
    int targetIndex,
    String targetName,
    int iterationsLimit,
    double minWeightsUpdate,
    double lambda,
    bool fitIntercept = false,
    double interceptScale = 1.0,
    InitialWeightsType initialWeightsType = InitialWeightsType.zeroes,
    DType dtype,
    Matrix initialWeights,
    bool isTrainDataNormalized = false,
  }) {
    final featuresTargetSplits = featuresTargetSplit(fittingData)
        .toList();

    return CoordinateRegressor(
      featuresTargetSplits[0].toMatrix(),
      featuresTargetSplits[1].toMatrix(),
      iterationsLimit: iterationsLimit,
      minWeightsUpdate: minWeightsUpdate,
      lambda: lambda,
      fitIntercept: fitIntercept,
      interceptScale: interceptScale,
      initialWeightsType: initialWeightsType,
      dtype: dtype,
      initialWeights: initialWeights,
      isTrainDataNormalized: isTrainDataNormalized,
    );
  }

  /// Learned coefficients (or weights) for given features
  Vector get coefficients;
}
