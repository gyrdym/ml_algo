library metric;

import 'dart:math' as math;
import 'package:simd_vector/vector.dart';

part 'package:dart_ml/src/metric/regression/mape.dart';
part 'package:dart_ml/src/metric/regression/rmse.dart';
part 'package:dart_ml/src/metric/classification/accuracy.dart';

abstract class Metric {
  double getError(Float32x4Vector predictedLabels, Float32x4Vector origLabels);

  factory Metric.RMSE() => const _RMSEMetric();
  factory Metric.MAPE() => const _MAPEMetric();
  factory Metric.Accuracy() => const _AccuracyMetric();
}
