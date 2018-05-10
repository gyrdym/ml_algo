import 'package:simd_vector/vector.dart';

abstract class Optimizer {
  Vector findExtrema(
    Iterable<Vector> points,
    Vector labels,
    {
      Vector initialWeights,
      bool isMinimizingObjective,
      bool arePointsNormalized,
      bool fitIntercept
    }
  );
}