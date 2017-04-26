import 'package:dart_ml/src/math/vector/vector_interface.dart';
import 'package:dart_ml/src/predictors/linear_regression.dart';
import 'package:dart_ml/src/optimizers/gradient/batch_optimizer.dart';

class BGDLinearRegressor<T extends VectorInterface> extends LinearRegressor<T> {
  final BGDOptimizer<T> optimizer;

  BGDLinearRegressor({double learningRate = 1e-5, double minWeightsDistance = 1e-8, int iterationLimit = 10000})
      : optimizer = new BGDOptimizer<T>(learningRate, minWeightsDistance, iterationLimit);
}
