import 'package:linalg/linalg.dart';

typedef double OptimizationFunction<E> (
  Vector<E> targetVector,
  Iterable<Vector<E>> vectorArgs,
  Iterable<double> scalarArgs
);

abstract class GradientCalculator<E> {
  Vector<E> getGradient(
    OptimizationFunction<E> function,
    Vector<E> targetVector,
    Iterable<Vector<E>> vectorArgs,
    Iterable<double> scalarArgs,
    double argumentDelta
  );
}
