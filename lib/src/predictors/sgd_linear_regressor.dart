import 'package:dart_ml/src/math/vector/vector_interface.dart';
import 'package:dart_ml/src/predictors/linear_regression.dart';
import 'package:dart_ml/src/optimizers/gradient/stochastic_optimizer.dart';

class SGDLinearRegressor<T extends VectorInterface> extends LinearRegressor<T> {
  final SGDOptimizer<T> optimizer;

  SGDLinearRegressor({double learningRate = 1e-5, double minWeightsDistance = 1e-8, int iterationLimit = 10000})
      : optimizer = new SGDOptimizer<T>(learningRate, minWeightsDistance, iterationLimit);
}
