part of metric;

class _MAPEMetric implements Metric {
  const _MAPEMetric();

  double getError(Float32x4Vector predictedLabels, Float32x4Vector origLabels) =>
      100 / predictedLabels.length * ((origLabels - predictedLabels) / origLabels).abs().sum();
}
