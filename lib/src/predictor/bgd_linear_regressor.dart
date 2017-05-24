import 'package:dart_ml/src/estimator/rmse.dart';
import 'package:dart_ml/src/predictor/linear_regresor.dart';
import 'package:dart_ml/src/optimizer/gradient/batch_optimizer.dart';

class BGDLinearRegressor extends LinearRegressor {
  @override
  final BGDOptimizer optimizer;

  @override
  final RMSEEstimator defaultEstimator = new RMSEEstimator();

  BGDLinearRegressor({double learningRate = 1e-5, double minWeightsDistance = 1e-8, int iterationLimit = 10000})
      : optimizer = new BGDOptimizer(learningRate, minWeightsDistance, iterationLimit);
}
