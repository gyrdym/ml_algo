import 'package:linalg/vector.dart';

abstract class Optimizer<E> {
  Vector<E> findExtrema(
    List<Vector<E>> points,
    Vector<E> labels,
    {
      Vector<E> initialWeights,
      bool isMinimizingObjective,
      bool arePointsNormalized
    }
  );
}