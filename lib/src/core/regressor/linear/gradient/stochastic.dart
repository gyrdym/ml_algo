part of 'package:dart_ml/src/core/implementation.dart';

class SGDRegressor extends _RegressorImpl {
  SGDRegressor({int iterationLimit, double learningRate, double minWeightsDistance, double alpha, double argumentIncrement,
        RegressionMetricType metric, Regularization regularization}) :

        super(ModuleFactory.SGDRegressionModule(learningRate: learningRate,
                                                          minWeightsDistance: minWeightsDistance,
                                                          iterationLimit: iterationLimit,
                                                          metric: metric,
                                                          regularization: regularization,
                                                          alpha: alpha,
                                                          argumentIncrement: argumentIncrement));
}
