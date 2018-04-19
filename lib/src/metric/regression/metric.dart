import 'package:dart_ml/src/metric/metric.dart';
import 'package:simd_vector/vector.dart';

abstract class RegressionMetric implements Metric {
  double getError(Vector predictedLabels, Vector origLabels);
}
