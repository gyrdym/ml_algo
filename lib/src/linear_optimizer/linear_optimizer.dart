import 'package:ml_linalg/matrix.dart';

abstract class LinearOptimizer {
  /// [initialCoefficients] initial weights (coefficients) to start optimization (e.g. random values)
  ///
  /// [isMinimizingObjective] should the solver find a maxima or minima
  Matrix findExtrema({
      Matrix initialCoefficients,
      bool isMinimizingObjective,
    });
}
