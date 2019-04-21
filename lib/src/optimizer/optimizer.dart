import 'package:ml_linalg/matrix.dart';

abstract class Optimizer {
  /// Accepts points (a matrix of all the X-coordinates values), a vector of y-labels and returns a matrix
  /// of corresponding coefficients (weights).
  ///
  /// [points] input X coordinates values
  ///
  /// [labels] input y coordinate values
  ///
  /// [initialWeights] initial weights (coefficients) to start optimization (e.g. random values)
  ///
  /// [isMinimizingObjective] should the optimizer find a maxima or minima
  Matrix findExtrema(Matrix points, Matrix labels, {
      Matrix initialWeights,
      bool isMinimizingObjective,
    });
}
