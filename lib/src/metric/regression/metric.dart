import 'package:dart_ml/src/metric/metric.dart';
import 'package:linalg/vector.dart';

abstract class RegressionMetric implements Metric {
  double getError(Vector predictedLabels, Vector origLabels);
}
