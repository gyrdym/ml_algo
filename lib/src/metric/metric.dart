library metric;

import 'dart:math' as math;
import 'package:dart_vector/vector.dart';

part 'mape.dart';
part 'rmse.dart';

abstract class Metric {
  double getError(Float32x4Vector predictedLabels, Float32x4Vector origLabels);

  factory Metric.RMSE() => const _RMSEMetric();
  factory Metric.MAPE() => const _MAPEMetric();
}
