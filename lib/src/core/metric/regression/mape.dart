import 'package:dart_ml/src/core/metric/regression/metric.dart';
import 'package:simd_vector/vector.dart';

class MAPEMetric implements RegressionMetric {
  const MAPEMetric();

  double getError(Float32x4Vector predictedLabels, Float32x4Vector origLabels) =>
      100 / predictedLabels.length * ((origLabels - predictedLabels) / origLabels).abs().sum();
}
