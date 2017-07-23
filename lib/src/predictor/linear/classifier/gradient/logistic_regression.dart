part of 'package:dart_ml/src/predictor/predictor.dart';

class LogisticRegressor extends GradientLinearClassifier {
  LogisticRegressor({double learningRate, double minWeightsDistance, int iterationLimit, ClassificationMetric metric,
                      Regularization regularization, ModuleInjector customInjector, alpha}) : super(metric) {

    injector = customInjector ?? InjectorFactory.create();

    _optimizer = (injector.get(SGDOptimizer) as SGDOptimizer)
      ..configure(
          learningRate: learningRate,
          minWeightsDistance: minWeightsDistance,
          iterationLimit: iterationLimit,
          regularization: regularization,
          lossFunction: new LossFunction.LogisticLoss(),
          scoreFunction: new ScoreFunction.Linear(),
          alpha: alpha
        );
  }
}
