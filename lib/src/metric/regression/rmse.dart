part of metric;

class _RMSEMetric implements Metric {
  const _RMSEMetric();

  double getError(Float32x4Vector predictedLabels, Float32x4Vector origLabels) =>
    math.sqrt(((predictedLabels - origLabels).intPow(2)).mean());
}
