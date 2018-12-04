import 'dart:math' as math;
import 'dart:typed_data';
import 'package:ml_algo/src/metric/regression/metric.dart';
import 'package:ml_linalg/linalg.dart';

class RMSEMetric implements RegressionMetric<Float32x4> {
  const RMSEMetric();

  @override
  double getError(MLVector<Float32x4> predictedLabels, MLVector<Float32x4> origLabels) =>
      math.sqrt(((predictedLabels - origLabels).toIntegerPower(2)).mean());
}
