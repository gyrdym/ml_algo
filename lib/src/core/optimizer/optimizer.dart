import 'dart:typed_data';

import 'package:simd_vector/vector.dart';

abstract class Optimizer {
  Float32x4Vector findExtrema(
    List<Float32x4Vector> features,
    Float32List labels,
    {
      Float32x4Vector initialWeights,
      bool isMinimizingObjective
    }
  );
}