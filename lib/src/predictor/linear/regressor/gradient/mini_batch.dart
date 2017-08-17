part of 'package:dart_ml/src/predictor/predictor.dart';

class MBGDRegressor extends _GradientRegressor<MBGDOptimizer> {
  MBGDRegressor({double learningRate, double minWeightsDistance, int iterationLimit, Metric metric,
                 Regularization regularization, ModuleInjector customInjector, alpha})

      : super(learningRate: learningRate, minWeightsDistance: minWeightsDistance, iterationLimit: iterationLimit,
                  metric: metric, regularization: regularization, customInjector: customInjector, alpha: alpha);
}