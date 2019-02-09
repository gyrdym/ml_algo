import 'package:ml_algo/gradient_type.dart';
import 'package:ml_algo/learning_rate_type.dart';
import 'package:ml_algo/predictor.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_algo/src/regressor/gradient_regressor.dart';
import 'package:ml_algo/src/regressor/lasso_regressor.dart';

/// A factory for all the linear regressors. A typical linear regressor uses the equation of a line for multidimensional
/// space to make a prediction. Each `x` in the equation has its own coefficient or weight and the combination of these
/// `x`-es and its coefficients gives the `y` term. The latter is a thing, that the regressor should predict, and
/// as one knows all the `x` values (since it is the input for the algorithm), the regressor should find the best
/// coefficients or weights to make a best prediction of `y` term.
abstract class LinearRegressor implements Predictor {
  /// Returns gradient linear regressor. Uses gradient descent optimizer to find the weights
  /// Parameters:
  /// [iterationsLimit] A number of fitting iterations. Uses as a condition of convergence in the optimizer. Default
  /// value is 100
  /// [initialLearningRate] A value, defining velocity of the convergence of the gradient descent optimizer.
  /// [minWeightsUpdate] A minimum distance between weights vectors in two subsequent iterations. Uses as a condition
  /// of convergence in the optimizer. In other words, if difference is small, there is no reason to continue fitting.
  /// Default value is 1e-12
  /// [lambda] A coefficient for regularization. For this particular regressor there is only L2 regularization type.
  /// [randomSeed] A seed, that will be passed to a random value generator, used by stochastic optimizers. Will be
  /// ignored, if the optimizer is not stochastic. Remember, each time you run the regressor with the same parameters,
  /// you will receive a different result. To avoid it, define [randomSeed]
  /// [batchSize] A size of data (in rows), that will be used for fitting per one iteration.
  /// [fitIntercept] Whether or not to fit intercept term. Default value is false.
  /// [interceptScale] A value, defining a size of the intercept term
  /// [learningRateType] A value, defining a strategy for the learning rate behaviour throughout the whole fitting
  /// process
  /// [gradientType] A type of gradient descent optimizer (stochastic, mini batch, batch)
  /// [dtype] A data type for all the numeric values, used by the algorithm. Can affect performance or accuracy of the
  /// computations. Default value is [Float32x4]
  factory LinearRegressor.gradient({
    int iterationsLimit,
    LearningRateType learningRateType,
    InitialWeightsType initialWeightsType,
    double initialLearningRate,
    double minWeightsUpdate,
    double lambda,
    GradientType gradientType,
    bool fitIntercept,
    double interceptScale,
    int randomSeed,
    int batchSize,
    Type dtype,
  }) = GradientRegressor;

  /// Returns a linear regressor with a coordinate descent optimizer to use L1 regularization.
  /// L1 regularization allows to select more important features, make less important as zeroes.
  /// Parameters:
  /// [iterationsLimit] A number of fitting iterations. Uses as a condition of convergence in the optimizer. Default
  /// value is 100
  /// [minWeightsUpdate] A minimum distance between weights vectors in two subsequent iterations. Uses as a condition
  /// of convergence in the optimizer. In other words, if difference is small, there is no reason to continue fitting.
  /// Default value is 1e-12
  /// [lambda] A coefficient for regularization. For this particular regressor there is only L1 regularization type.
  /// [fitIntercept] Whether or not to fit intercept term. Default value is false.
  /// [interceptScale] A value, defining a size of the intercept term
  /// [dtype] A data type for all the numeric values, used by the algorithm. Can affect performance or accuracy of the
  /// computations. Default value is [Float32x4]
  factory LinearRegressor.lasso({
    int iterationsLimit,
    double minWeightsUpdate,
    double lambda,
    bool fitIntercept,
    double interceptScale,
    InitialWeightsType initialWeightsType,
    Type dtype,
  }) = LassoRegressor;
}
