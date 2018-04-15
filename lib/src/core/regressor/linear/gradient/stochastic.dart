import 'package:dart_ml/src/core/metric/regression/type.dart';
import 'package:dart_ml/src/core/optimizer/regularization.dart';
import 'package:dart_ml/src/core/regressor/regressor_impl.dart';
import 'package:dart_ml/src/di/factory.dart';

class SGDRegressor extends RegressorImpl {
  SGDRegressor({
    int iterationLimit,
    double learningRate,
    double minWeightsDistance,
    double alpha,
    double argumentIncrement,
    RegressionMetricType metric,
    Regularization regularization
  }) : super(
    ModuleFactory.SGDRegressionModule(
      learningRate: learningRate,
      minWeightsDistance: minWeightsDistance,
      iterationLimit: iterationLimit,
      metric: metric,
      regularization: regularization,
      alpha: alpha,
      argumentIncrement: argumentIncrement
    )
  );
}
