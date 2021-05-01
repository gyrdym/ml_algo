import 'package:ml_algo/src/helpers/validate_matrix_columns.dart';
import 'package:ml_algo/src/metric/metric.dart';
import 'package:ml_linalg/matrix.dart';

class RssMetric implements Metric {
  const RssMetric();

  @override
  double getScore(Matrix predictedLabels, Matrix origLabels) {
    validateMatrixColumns([predictedLabels, origLabels]);

    final predicted = predictedLabels.toVector();
    final original = origLabels.toVector();

    return (predicted - original).pow(2).sum();
  }
}
