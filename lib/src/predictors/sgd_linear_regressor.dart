import 'package:dart_ml/src/estimators/rmse.dart';
import 'package:dart_ml/src/predictors/linear_regresor.dart';
import 'package:dart_ml/src/optimizers/gradient/stochastic_optimizer.dart';

class SGDLinearRegressor extends LinearRegressor {
  @override
  final SGDOptimizer optimizer;

  @override
  final RMSEEstimator defaultEstimator = new RMSEEstimator();

  SGDLinearRegressor({double learningRate = 1e-5, double minWeightsDistance = 1e-8, int iterationLimit = 10000})
      : optimizer = new SGDOptimizer(learningRate, minWeightsDistance, iterationLimit);
}
