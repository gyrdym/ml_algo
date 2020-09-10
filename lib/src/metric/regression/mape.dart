import 'package:ml_algo/src/common/exception/matrix_column_exception.dart';
import 'package:ml_algo/src/metric/metric.dart';
import 'package:ml_linalg/linalg.dart';

class MapeMetric implements Metric {
  const MapeMetric();

  @override
  double getScore(Matrix predictedLabels, Matrix originalLabels) {
    if (predictedLabels.columnsNum != 1) {
      throw MatrixColumnException(predictedLabels);
    }

    if (originalLabels.columnsNum != 1) {
      throw MatrixColumnException(originalLabels);
    }

    final predicted = predictedLabels.getColumn(0);
    final original = originalLabels.getColumn(0);

    return ((original - predicted) / original)
        .abs()
        .mean();
  }
}
