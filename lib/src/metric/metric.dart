library metric;

import 'dart:math' as math;
import 'package:dart_ml/src/math/vector/vector.dart';

part 'mape.dart';
part 'rmse.dart';

abstract class Metric {
  double getError(Vector predictedLabels, Vector origLabels);

  factory Metric.RMSE() => const _RMSEMetric();
  factory Metric.MAPE() => const _MAPEMetric();
}
