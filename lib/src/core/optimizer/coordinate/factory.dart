import 'package:dart_ml/src/core/optimizer/coordinate/optimizer.dart';
import 'package:dart_ml/src/core/optimizer/optimizer.dart';

class CoordinateOptimizerFactory {
  static Optimizer createCoordinateOptimizer(
    double minWeightsDiff,
    int iterationLimit,
    double lambda
  ) => new CoordinateOptimizerImpl(
    minCoefficientsDiff: minWeightsDiff,
    iterationLimit: iterationLimit,
    lambda: lambda
  );
}
