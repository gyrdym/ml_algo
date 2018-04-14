part of 'package:dart_ml/src/core/implementation.dart';

class LassoRegressor extends _RegressorImpl {
  LassoRegressor({
    int iterationLimit,
    double minWeightsDistance,
    Metric metric,
    ScoreFunction scoreFn,
    double lambda
  }) : super(createLassoRegressionModule(
      iterationLimit: iterationLimit,
      minWeightsDistance: minWeightsDistance,
      metric: metric,
      scoreFn: scoreFn,
      lambda: lambda
  ));
}
