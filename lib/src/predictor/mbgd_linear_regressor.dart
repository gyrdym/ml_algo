import 'package:dart_ml/src/estimator/rmse.dart';
import 'package:dart_ml/src/predictor/linear_regresor.dart';
import 'package:dart_ml/src/optimizer/gradient/mini_batch_optimizer.dart';

class MBGDLinearRegressor extends LinearRegressor {
  @override
  final MBGDOptimizer optimizer;

  @override
  final RMSEEstimator defaultEstimator = new RMSEEstimator();

  MBGDLinearRegressor({double learningRate = 1e-5, double minWeightsDistance = 1e-8, int iterationLimit = 10000})
      : optimizer = new MBGDOptimizer(learningRate, minWeightsDistance, iterationLimit);
}
