import 'package:ml_linalg/matrix.dart';

abstract class LinearOptimizer {
  /// [initialWeights] initial weights (coefficients) to start optimization (e.g. random values)
  ///
  /// [isMinimizingObjective] should the solver find a maxima or minima
  Matrix findExtrema({
      Matrix initialWeights,
      bool isMinimizingObjective,
    });
}
