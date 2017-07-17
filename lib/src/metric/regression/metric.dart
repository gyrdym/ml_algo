import 'dart:math' as math;
import 'package:simd_vector/vector.dart';
import 'package:dart_ml/src/metric/base.dart';

part 'mape.dart';
part 'rmse.dart';

abstract class RegressionMetric implements Metric {
  double getError(Float32x4Vector predictedLabels, Float32x4Vector origLabels);

  factory RegressionMetric.RMSE() => const _RMSEMetric();
  factory RegressionMetric.MAPE() => const _MAPEMetric();
}
