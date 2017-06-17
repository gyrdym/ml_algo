import 'package:dart_ml/src/di/injector.dart';
import 'package:dart_ml/src/metric/metric.dart';
import 'package:dart_ml/src/optimizer/gradient/interface/stochastic.dart';
import 'package:dart_ml/src/optimizer/regularization/regularization.dart';
import 'package:dart_ml/src/loss_function/loss_function.dart';

import 'package:dart_ml/src/predictor/linear/regressor/gradient/base.dart';

class SGDRegressor extends GradientLinearRegressor {
  SGDRegressor({double learningRate = 1e-5,
                 double minWeightsDistance = 1e-8,
                 int iterationLimit = 10000,
                 Metric metric,
                 Regularization regularization = Regularization.L2,
                 LossFunction lossFunction,
                 alpha = .00001})
      : super(
      (injector.get(SGDOptimizer) as SGDOptimizer)
        ..configure(learningRate, minWeightsDistance, iterationLimit, regularization,
                        lossFunction ?? new LossFunction.Squared(), alpha: alpha),
      metric: metric);
}
