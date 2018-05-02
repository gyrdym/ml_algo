import 'package:simd_vector/vector.dart';

abstract class Metric {
  double getError(Vector predictedLabels, Vector origLabels);
}