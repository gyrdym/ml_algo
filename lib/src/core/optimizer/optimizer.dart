part of 'package:dart_ml/src/core/interface.dart';

abstract class Optimizer {
  Float32x4Vector findExtrema(
    List<Float32x4Vector> features,
    Float32List labels,
    {
      Float32x4Vector weights,
      bool isMinimizingObjective
    }
  );
}