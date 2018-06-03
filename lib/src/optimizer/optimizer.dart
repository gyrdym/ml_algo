import 'package:linalg/vector.dart';

abstract class Optimizer {
  Vector findExtrema(
    Iterable<Vector> points,
    Vector labels,
    {
      Vector initialWeights,
      bool isMinimizingObjective,
      bool arePointsNormalized
    }
  );
}