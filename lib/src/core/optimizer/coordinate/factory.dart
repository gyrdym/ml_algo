part of 'package:dart_ml/src/core/implementation.dart';

class CoordinateOptimizerFactory {
  static Optimizer createCoordinateOptimizer(
    double minWeightsDiff,
    int iterationLimit,
    double lambda
  ) => new _CoordinateOptimizerImpl(
    minCoefficientsDiff: minWeightsDiff,
    iterationLimit: iterationLimit,
    lambda: lambda
  );
}
