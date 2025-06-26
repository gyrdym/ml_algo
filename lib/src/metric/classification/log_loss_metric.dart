import 'package:ml_algo/src/helpers/validate_matrix_columns.dart';
import 'package:ml_algo/src/metric/metric.dart';
import 'package:ml_linalg/matrix.dart';
import 'dart:math' as math;

class LogLossMetric implements Metric {
  const LogLossMetric({this.eps = 1e-15});

  final double eps;

  double _clip(double p) => p < eps ? eps : (p > 1.0 - eps ? 1.0 - eps : p);

  @override
  double getScore(Matrix predictedLabels, Matrix origLabels) {
    validateMatrixColumns([predictedLabels, origLabels]);

    final preds = predictedLabels.toVector();
    final orig = origLabels.toVector();

    var sum = 0.0;
    for (var i = 0; i < preds.length; i++) {
      final p = _clip(preds[i]);
      final y = orig[i];
      sum += y == 1 ? -math.log(p) : -math.log(1.0 - p);
    }
    return sum / preds.length;
  }
}
