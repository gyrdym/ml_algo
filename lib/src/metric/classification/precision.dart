import 'package:ml_algo/src/metric/classification/_helpers/divide_true_positive_by.dart';
import 'package:ml_algo/src/metric/metric.dart';
import 'package:ml_linalg/matrix.dart';

class PrecisionMetric implements Metric {
  const PrecisionMetric();

  @override

  /// Accepts [predictedLabels] and [origLabels] with entries with `1` as
  /// positive label and `0` as negative one
  double getScore(Matrix predictedLabels, Matrix origLabels) {
    final predictedTrueCounts =
        predictedLabels.reduceRows((counts, row) => counts + row);

    return divideTruePositiveBy(
        predictedTrueCounts, origLabels, predictedLabels);
  }
}
