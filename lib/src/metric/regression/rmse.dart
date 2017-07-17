part of 'metric.dart';

class _RMSEMetric implements RegressionMetric {
  const _RMSEMetric();

  double getError(Float32x4Vector predictedLabels, Float32x4Vector origLabels) =>
    math.sqrt(((predictedLabels - origLabels).intPow(2)).mean());
}
