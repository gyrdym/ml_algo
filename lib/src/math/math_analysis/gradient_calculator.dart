import 'package:ml_linalg/linalg.dart';

typedef double OptimizationFunction<E> (
  MLVector<E> targetVector,
  Iterable<MLVector<E>> vectorArgs,
  Iterable<double> scalarArgs
);

abstract class GradientCalculator<E> {
  MLVector<E> getGradient(
    OptimizationFunction<E> function,
    MLVector<E> targetVector,
    Iterable<MLVector<E>> vectorArgs,
    Iterable<double> scalarArgs,
    double argumentDelta
  );
}
