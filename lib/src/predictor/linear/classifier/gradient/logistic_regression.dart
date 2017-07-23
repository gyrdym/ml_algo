part of 'package:dart_ml/src/predictor/predictor.dart';

class LogisticRegressor extends _GradientLinearClassifier<SGDOptimizer> {
  LogisticRegressor({double learningRate, double minWeightsDistance, int iterationLimit, Metric metric,
                 Regularization regularization, ModuleInjector customInjector, alpha})

      : super(lossFunction: new LossFunction.LogisticLoss(), learningRate: learningRate,
                  minWeightsDistance: minWeightsDistance, iterationLimit: iterationLimit, metric: metric,
                  regularization: regularization, customInjector: customInjector, alpha: alpha);
}
