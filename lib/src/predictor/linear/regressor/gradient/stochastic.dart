part of 'package:dart_ml/src/predictor/predictor.dart';

class SGDRegressor extends _GradientRegressor<SGDOptimizer> {
  SGDRegressor({double learningRate, double minWeightsDistance, int iterationLimit, Metric metric,
                 Regularization regularization, alpha})

      : super(learningRate: learningRate, minWeightsDistance: minWeightsDistance, iterationLimit: iterationLimit,
                  metric: metric, regularization: regularization, alpha: alpha);
}