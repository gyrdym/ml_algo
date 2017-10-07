part of 'package:dart_ml/src/implementation.dart';

class _AccuracyMetric implements ClassificationMetric {
  const _AccuracyMetric();

  double getError(Float32x4Vector predictedLabels, Float32x4Vector origLabels) =>
      1 - getScore(predictedLabels, origLabels);

  double getScore(Float32x4Vector predictedLabels, Float32x4Vector origLabels) {
    double score = 0.0;

    List<double> _origLabels = origLabels.asList();
    List<double> _predictedLabels = predictedLabels.asList();

    for (int i = 0; i < origLabels.length; i++) {
      if (_origLabels[i] == _predictedLabels[i]) {
        score++;
      }
    }

    return score / origLabels.length;
  }
}
