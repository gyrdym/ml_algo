import 'package:ml_algo/src/metric/metric.dart';
import 'package:ml_linalg/linalg.dart';

class AccuracyMetric implements Metric {
  const AccuracyMetric();

  @override
  double getScore(Matrix predictedLabels, Matrix origLabels) {
    if (predictedLabels.rowsNum != origLabels.rowsNum &&
        predictedLabels.columnsNum != origLabels.columnsNum) {
      throw Exception('Predicated labels and original labels should have '
          'the same dimensions');
    }

    double score = 0.0;
    for (int i = 0; i < origLabels.rowsNum; i++) {
      if (origLabels.getRow(i) == predictedLabels.getRow(i)) {
        score++;
      }
    }

    return score / origLabels.rowsNum;
  }
}
