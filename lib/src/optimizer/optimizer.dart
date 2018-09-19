import 'package:linalg/vector.dart';

abstract class Optimizer<E, S extends List<E>, T extends List<double>> {

  SIMDVector<S, T, E> findExtrema(
    List<SIMDVector<S, T, E>> points,
    SIMDVector<S, T, E> labels,
    {
      SIMDVector<S, T, E> initialWeights,
      bool isMinimizingObjective,
      bool arePointsNormalized
    }
  );
}