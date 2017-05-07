import 'package:dart_ml/src/estimators/rmse.dart';
import 'package:dart_ml/src/predictors/linear_regresor.dart';
import 'package:dart_ml/src/optimizers/gradient/batch_optimizer.dart';

class BGDLinearRegressor extends LinearRegressor {
  @override
  final BGDOptimizer optimizer;

  @override
  final RMSEEstimator defaultEstimator = new RMSEEstimator();

  BGDLinearRegressor({double learningRate = 1e-5, double minWeightsDistance = 1e-8, int iterationLimit = 10000})
      : optimizer = new BGDOptimizer(learningRate, minWeightsDistance, iterationLimit);
}
