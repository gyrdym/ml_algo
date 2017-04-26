import 'package:dart_ml/src/math/vector/vector_interface.dart';
import 'package:dart_ml/src/predictors/linear_regression.dart';
import 'package:dart_ml/src/optimizers/gradient/mini_batch_optimizer.dart';

class MBGDLinearRegressor<T extends VectorInterface> extends LinearRegressor<T> {
  final MBGDOptimizer<T> optimizer;

  MBGDLinearRegressor({double learningRate = 1e-5, double minWeightsDistance = 1e-8, int iterationLimit = 10000})
      : optimizer = new MBGDOptimizer<T>(learningRate, minWeightsDistance, iterationLimit);
}
