import 'package:ml_algo/gradient_type.dart';
import 'package:ml_algo/learning_rate_type.dart';
import 'package:ml_algo/predictor.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_algo/src/regressor/gradient_regressor.dart';
import 'package:ml_algo/src/regressor/lasso_regressor.dart';

abstract class LinearRegressor implements Predictor {
  factory LinearRegressor.gradient({
    int iterationLimit,
    LearningRateType learningRateType,
    InitialWeightsType initialWeightsType,
    double learningRate,
    double minWeightsUpdate,
    double lambda,
    GradientType gradientType,
    bool fitIntercept,
    double interceptScale,
    int randomSeed,
    int batchSize,
    Type dtype,
  }) = GradientRegressor;

  factory LinearRegressor.lasso({
    int iterationLimit,
    double minWeightUpdate,
    double lambda,
    bool fitIntercept,
    double interceptScale,
    InitialWeightsType initialWeightsType,
    Type dtype,
  }) = LassoRegressor;

  factory LinearRegressor.kernel() => throw UnimplementedError();
}
