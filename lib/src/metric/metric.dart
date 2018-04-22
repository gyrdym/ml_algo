import 'package:simd_vector/vector.dart';

abstract class Metric {
  double getError(Float32x4Vector predictedLabels, Float32x4Vector origLabels);
}