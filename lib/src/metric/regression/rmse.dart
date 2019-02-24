import 'dart:math' as math;

import 'package:ml_algo/src/metric/metric.dart';
import 'package:ml_linalg/linalg.dart';

class RMSEMetric implements Metric {
  const RMSEMetric();

  @override
  double getScore(MLVector predictedLabels, MLVector origLabels) =>
      math.sqrt(((predictedLabels - origLabels).toIntegerPower(2)).mean());
}
