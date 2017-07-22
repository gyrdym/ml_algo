part of '../../../predictor.dart';

class GradientRegressor<T extends GradientOptimizer> extends GradientLinearPredictor {
  GradientRegressor({double learningRate, double minWeightsDistance, int iterationLimit, Metric metric,
                      Regularization regularization, ModuleInjector customInjector, alpha}) : super(metric: metric) {

    injector = customInjector ?? InjectorFactory.create();

    _optimizer = injector.get(T)
      ..configure(
          learningRate: learningRate,
          minWeightsDistance: minWeightsDistance,
          iterationLimit: iterationLimit,
          regularization: regularization,
          lossFunction: new LossFunction.Squared(),
          scoreFunction: new ScoreFunction.Linear(),
          alpha: alpha
      );
  }
}
