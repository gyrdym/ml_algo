part of '../../../predictor.dart';

class BGDRegressor extends GradientRegressor<BGDOptimizer> {
  BGDRegressor({double learningRate, double minWeightsDistance, int iterationLimit, Metric metric,
                 Regularization regularization, ModuleInjector customInjector, alpha}) : super();
}
