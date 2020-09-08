import 'package:ml_algo/src/metric/metric.dart';
import 'package:ml_linalg/linalg.dart';

class MapeMetric implements Metric {
  const MapeMetric();

  @override
  double getScore(Matrix predictedLabels, Matrix origLabels) {
    if (predictedLabels.columnsNum != 1 || origLabels.columnsNum != 1) {
      throw Exception('Both predicted labels and original labels have to be '
          'a matrix-column');
    }
    final predicted = predictedLabels.getColumn(0);
    final original = origLabels.getColumn(0);

    return 100 / predicted.length *
        ((original - predicted) / original).abs().sum();
  }
}
