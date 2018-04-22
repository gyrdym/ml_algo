import 'package:dart_ml/src/metric/regression/metric.dart';
import 'package:simd_vector/vector.dart';

class MAPEMetric implements RegressionMetric {
  const MAPEMetric();

  double getError(Vector predictedLabels, Vector origLabels) =>
      100 / predictedLabels.length * ((origLabels - predictedLabels) / origLabels).abs().sum();
}
