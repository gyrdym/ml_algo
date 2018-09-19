import 'package:linalg/vector.dart';

typedef double OptimizationFunction(
  Vector targetVector,
  Iterable<Vector> vectorArgs,
  Iterable<double> scalarArgs
);

abstract class GradientCalculator {
  Vector getGradient(
    OptimizationFunction function,
    SIMDVector targetVector,
    Iterable<SIMDVector> vectorArgs,
    Iterable<double> scalarArgs,
    double argumentDelta
  );
}
