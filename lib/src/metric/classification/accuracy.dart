import 'package:dart_ml/src/metric/classification/metric.dart';
import 'package:simd_vector/vector.dart';

class AccuracyMetric implements ClassificationMetric {
  const AccuracyMetric();

  double getError(Vector predictedLabels, Vector origLabels) =>
      1 - getScore(predictedLabels, origLabels);

  double getScore(Vector predictedLabels, Vector origLabels) {
    double score = 0.0;
    final _origLabels = origLabels as List;
    final _predictedLabels = predictedLabels as List;

    for (int i = 0; i < origLabels.length; i++) {
      if (_origLabels[i] == _predictedLabels[i]) {
        score++;
      }
    }

    return score / origLabels.length;
  }
}
