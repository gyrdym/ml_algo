import 'dart:math' as math;

import 'package:ml_algo/src/metric/regression/metric.dart';
import 'package:ml_linalg/linalg.dart';

class RMSEMetric implements RegressionMetric {
  const RMSEMetric();

  @override
  double getError(MLVector predictedLabels, MLVector origLabels) =>
      math.sqrt(((predictedLabels - origLabels).toIntegerPower(2)).mean());
}
