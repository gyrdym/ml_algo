import 'dart:math' as math;
import 'package:dart_ml/src/metric/regression/metric.dart';
import 'package:linalg/vector.dart';

class RMSEMetric implements RegressionMetric {
  const RMSEMetric();

  double getError(Vector predictedLabels, Vector origLabels) =>
    math.sqrt(((predictedLabels - origLabels).toIntegerPower(2)).mean());
}
