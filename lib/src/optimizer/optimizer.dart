import 'package:linalg/linalg.dart';

abstract class Optimizer<E, T extends Vector<E>> {
  Vector<E> findExtrema(
    Matrix<E, T> points,
    Vector<E> labels,
    {
      Vector<E> initialWeights,
      bool isMinimizingObjective,
      bool arePointsNormalized
    }
  );
}