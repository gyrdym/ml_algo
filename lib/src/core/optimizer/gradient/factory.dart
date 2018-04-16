import 'package:dart_ml/src/core/optimizer/gradient/optimizer.dart';
import 'package:dart_ml/src/core/optimizer/optimizer.dart';

Optimizer gradientOptimizerFactory(
  double learningRate,
  double minWeightsDistance,
  int iterationLimit,
  double lambda,
  double argumentIncrement,
  int batchSize
) =>
  new GradientOptimizerImpl(
    learningRate: learningRate,
    minCoefficientsUpdate: minWeightsDistance,
    iterationLimit: iterationLimit,
    lambda: lambda,
    argumentIncrement: argumentIncrement,
    batchSize: batchSize
  );