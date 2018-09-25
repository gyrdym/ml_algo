import 'package:dart_ml/src/metric/classification/metric.dart';
import 'package:linalg/vector.dart';

class AccuracyMetric implements ClassificationMetric {
  const AccuracyMetric();

  @override
  double getError(Vector predictedLabels, Vector origLabels) =>
      1 - getScore(predictedLabels, origLabels);

  @override
  double getScore(Vector predictedLabels, Vector origLabels) {
    double score = 0.0;
    for (int i = 0; i < origLabels.length; i++) {
      if (origLabels[i] == predictedLabels[i]) {
        score++;
      }
    }

    return score / origLabels.length;
  }
}
