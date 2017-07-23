part of 'package:dart_ml/src/predictor/predictor.dart';

class _GradientRegressor<T extends GradientOptimizer> extends _GradientLinearPredictor {
  _GradientRegressor({double learningRate, double minWeightsDistance, int iterationLimit, Metric metric,
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
