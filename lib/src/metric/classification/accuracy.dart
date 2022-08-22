import 'package:ml_algo/src/metric/metric.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:quiver/iterables.dart';

class AccuracyMetric implements Metric {
  const AccuracyMetric();

  @override
  double getScore(Matrix predictedLabels, Matrix origLabels) {
    if (predictedLabels.rowsNum != origLabels.rowsNum &&
        predictedLabels.columnsNum != origLabels.columnsNum) {
      throw Exception('Predicted labels and original labels should have '
          'the same dimensions');
    }

    final score = zip([origLabels.rows, predictedLabels.rows])
        .where((rows) => rows.first == rows.last)
        .length;

    return score / origLabels.rowsNum;
  }
}
