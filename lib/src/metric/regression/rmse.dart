import 'dart:math' as math;

import 'package:ml_algo/src/helpers/validate_matrix_columns.dart';
import 'package:ml_algo/src/metric/metric.dart';
import 'package:ml_linalg/linalg.dart';

class RmseMetric implements Metric {
  const RmseMetric();

  @override
  double getScore(Matrix predictedLabels, Matrix origLabels) {
    validateMatrixColumns([predictedLabels, origLabels]);

    final predicted = predictedLabels.getColumn(0);
    final original = origLabels.getColumn(0);

    return math.sqrt(((predicted - original).pow(2)).mean());
  }
}
