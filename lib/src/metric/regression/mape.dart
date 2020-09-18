import 'package:ml_algo/src/helpers/validate_matrix_columns.dart';
import 'package:ml_algo/src/metric/metric.dart';
import 'package:ml_linalg/linalg.dart';

class MapeMetric implements Metric {
  const MapeMetric();

  @override
  double getScore(Matrix predictedLabels, Matrix originalLabels) {
    validateMatrixColumns([predictedLabels, originalLabels]);

    final predicted = predictedLabels
        .toVector();
    final original = originalLabels
        .toVector();

    return ((original - predicted) / original)
        .abs()
        .mean();
  }
}
