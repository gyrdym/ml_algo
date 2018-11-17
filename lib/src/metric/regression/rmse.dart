import 'dart:math' as math;
import 'dart:typed_data';
import 'package:dart_ml/src/metric/regression/metric.dart';
import 'package:linalg/vector.dart';

class RMSEMetric implements RegressionMetric<Float32x4> {

  const RMSEMetric();

  @override
  double getError(Vector<Float32x4> predictedLabels, Vector<Float32x4> origLabels) =>
      math.sqrt(((predictedLabels - origLabels).toIntegerPower(2)).mean());
}
