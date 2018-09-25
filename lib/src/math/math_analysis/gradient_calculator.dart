import 'package:linalg/vector.dart';

typedef double OptimizationFunction<S extends List<E>, T extends List<double>, E> (
  SIMDVector<S, T, E> targetVector,
  Iterable<SIMDVector<S, T, E>> vectorArgs,
  Iterable<double> scalarArgs
);

abstract class GradientCalculator<S extends List<E>, T extends List<double>, E> {
  SIMDVector<S, T, E> getGradient(
    OptimizationFunction<S, T, E> function,
    SIMDVector<S, T, E> targetVector,
    Iterable<SIMDVector<S, T, E>> vectorArgs,
    Iterable<double> scalarArgs,
    double argumentDelta
  );
}
