part of 'package:dart_ml/src/core/implementation.dart';

class CoordinateOptimizerFactory {
  static Optimizer createCoordinateOptimizerFactory(
    double minWeightsDiff,
    int iterationLimit,
    double lambda
  ) => new _CoordinateOptimizerImpl(
    minWeightsDiff: minWeightsDiff,
    iterationLimit: iterationLimit,
    lambda: lambda
  );
}
