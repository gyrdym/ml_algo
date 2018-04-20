import 'package:simd_vector/vector.dart';

abstract class Optimizer {
  Vector findExtrema(
    Iterable<Vector> points,
    Iterable<double> labels,
    {
      Vector initialWeights,
      bool isMinimizingObjective
    }
  );
}