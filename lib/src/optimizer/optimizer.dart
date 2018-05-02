import 'package:dart_ml/src/cost_function/cost_function.dart';
import 'package:simd_vector/vector.dart';

abstract class Optimizer {
  CostFunction get costFunction;
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