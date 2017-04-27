import 'package:dart_ml/src/predictors/linear_regresor.dart';
import 'package:dart_ml/src/optimizers/gradient/mini_batch_optimizer.dart';

class MBGDLinearRegressor extends LinearRegressor {
  final MBGDOptimizer optimizer;

  MBGDLinearRegressor({double learningRate = 1e-5, double minWeightsDistance = 1e-8, int iterationLimit = 10000})
      : optimizer = new MBGDOptimizer(learningRate, minWeightsDistance, iterationLimit);
}
