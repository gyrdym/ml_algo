part of 'package:dart_ml/src/core/implementation.dart';

class BGDRegressor extends _RegressorImpl {
  BGDRegressor({int iterationLimit, double learningRate, double minWeightsDistance, double alpha, double argumentIncrement,
        RegressionMetricType metric, Regularization regularization}) :

        super(ModuleFactory.BGDRegressionModule(learningRate: learningRate,
                                                    minWeightsDistance: minWeightsDistance,
                                                    iterationLimit: iterationLimit,
                                                    metric: metric,
                                                    regularization: regularization,
                                                    alpha: alpha,
                                                    argumentIncrement: argumentIncrement));
}
