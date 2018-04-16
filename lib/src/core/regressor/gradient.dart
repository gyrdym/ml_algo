import 'package:dart_ml/src/core/metric/regression/type.dart';
import 'package:dart_ml/src/core/regressor/regressor_impl.dart';
import 'package:dart_ml/src/di/factory.dart';

class GradientRegressor extends RegressorImpl {
  GradientRegressor({
    int iterationLimit,
    double learningRate,
    double minWeightsDistance,
    double lambda = 0.0,
    double argumentIncrement,
    RegressionMetricType metric,
    int batchSize = 1,
  }) : super(
    ModuleFactory.GradientRegressionModule(
      learningRate: learningRate,
      minWeightsDistance: minWeightsDistance,
      iterationLimit: iterationLimit,
      metric: metric,
      lambda: lambda,
      argumentIncrement: argumentIncrement,
      batchSize: batchSize
    )
  );
}
