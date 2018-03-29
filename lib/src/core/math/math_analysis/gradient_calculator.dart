part of 'package:dart_ml/src/core/interface.dart';

typedef double OptimizationFunction(
  Float32x4Vector targetVector,
  List<Float32x4Vector> vectorArgs,
  List<double> scalarArgs
);

abstract class GradientCalculator {
  Float32x4Vector getGradient(
    OptimizationFunction function,
    Float32x4Vector targetVector,
    Iterable<Float32x4Vector> vectorArgs,
    Iterable<double> scalarArgs,
    double argumentDelta
  );
}
