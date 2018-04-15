import 'package:dart_ml/src/core/metric/metric.dart';
import 'package:simd_vector/vector.dart';

abstract class RegressionMetric implements Metric {
  double getError(Float32x4Vector predictedLabels, Float32x4Vector origLabels);
}
