part of 'package:dart_ml/src/core/implementation.dart';

class LogisticRegressor extends _ClassifierImpl {
  LogisticRegressor({int iterationLimit, double learningRate, double minWeightsDistance, double alpha,
                      double argumentIncrement, ClassificationMetricType metric, Regularization regularization}) :

        super(ModuleFactory.logisticRegressionModule(learningRate: learningRate,
                                                               minWeightsDistance: minWeightsDistance,
                                                               iterationLimit: iterationLimit,
                                                               metricType: metric,
                                                               regularization: regularization,
                                                               lambda: alpha,
                                                               argumentIncrement: argumentIncrement));
}
