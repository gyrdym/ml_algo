part of metric;

class _RMSEMetric implements Metric {
  const _RMSEMetric();

  double getError(Vector predictedLabels, Vector origLabels) =>
    math.sqrt(((predictedLabels - origLabels).intPow(2)).mean());
}
