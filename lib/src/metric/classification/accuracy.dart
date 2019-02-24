import 'package:ml_algo/src/metric/metric.dart';
import 'package:ml_linalg/linalg.dart';

class AccuracyMetric implements Metric {
  const AccuracyMetric();

  @override
  double getScore(MLVector predictedLabels, MLVector origLabels) {
    double score = 0.0;
    for (int i = 0; i < origLabels.length; i++) {
      if (origLabels[i] == predictedLabels[i]) {
        score++;
      }
    }
    return score / origLabels.length;
  }
}
