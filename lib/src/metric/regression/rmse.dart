import 'dart:math' as math;

import 'package:ml_algo/src/metric/metric.dart';
import 'package:ml_linalg/linalg.dart';

class RmseMetric implements Metric {
  const RmseMetric();

  @override
  double getScore(Matrix predictedLabels, Matrix origLabels) {
    if (predictedLabels.columnsNum != 1 || origLabels.columnsNum != 1) {
      throw Exception('Both predicted labels and original labels have to be '
          'a matrix-column');
    }

    final predicted = predictedLabels
        .getColumn(0);
    final original = origLabels
        .getColumn(0);

    return math.sqrt(((predicted - original).pow(2)).mean());
  }
}
