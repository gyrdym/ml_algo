import 'package:dart_ml/src/predictors/linear_regresor.dart';
import 'package:dart_ml/src/optimizers/gradient/batch_optimizer.dart';

class BGDLinearRegressor extends LinearRegressor {
  final BGDOptimizer optimizer;

  BGDLinearRegressor({double learningRate = 1e-5, double minWeightsDistance = 1e-8, int iterationLimit = 10000})
      : optimizer = new BGDOptimizer(learningRate, minWeightsDistance, iterationLimit);
}
