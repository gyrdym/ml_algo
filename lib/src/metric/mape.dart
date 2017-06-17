part of metric;

class _MAPEMetric implements Metric {
  const _MAPEMetric();

  double getError(Vector predictedLabels, Vector origLabels) =>
      100 / predictedLabels.length * ((origLabels - predictedLabels) / origLabels).abs(inPlace: true).sum();
}
