import 'package:ml_algo/src/helpers/normalize_class_labels.dart';
import 'package:ml_algo/src/metric/metric.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:quiver/iterables.dart';

/// TODO: add warning if predicted values are all zeroes
class PrecisionMetric implements Metric {
  const PrecisionMetric();

  @override
  /// Accepts [predictedLabels] and [origLabels] with entries with `1` as
  /// positive label and `0` as negative one
  double getScore(Matrix predictedLabels, Matrix origLabels) {
    final allPredictedPositiveCounts = predictedLabels
        .reduceRows((counts, row) => counts + row);

    // Let's say we have the following data:
    //
    // orig labels | predicted labels
    // -------------------------------
    //     1       |        1
    //     1       |        0
    //     0       |        1
    //     0       |        0
    //     1       |        1
    //--------------------------------
    //
    // in order to count correctly predicted positive labels in matrix notation
    // we may multiple predicted labels by 2, and then subtract the two
    // matrices from each other:
    //
    // 1 - (1 * 2) = -1
    // 1 - (0 * 2) =  1
    // 0 - (1 * 2) = -2
    // 0 - (0 * 2) =  0
    // 1 - (1 * 2) = -1
    //
    // we see that matrices subtraction in case of original positive label and a
    // predicted positive label gives -1, thus we need to count number of elements
    // with value equals -1 in the resulting matrix
    final difference = origLabels - (predictedLabels * 2);
    final correctPositiveCounts = difference
        .reduceRows(
            (counts, row) => counts + row.mapToVector((diff) => diff == -1
                ? 1 : 0),
            initValue: Vector.zero(
              origLabels.columnsNum,
              dtype: origLabels.dtype,
            ));
    final aggregatedScore = (correctPositiveCounts / allPredictedPositiveCounts)
        .mean();

    if (aggregatedScore.isFinite) {
      return aggregatedScore;
    }

    return zip([
      correctPositiveCounts,
      allPredictedPositiveCounts,
    ]).fold(0, (aggregated, pair) {
      final correctPositiveCount = pair.first;
      final allPredictedPositiveCount = pair.last;

      if (allPredictedPositiveCount != 0) {
        return aggregated + correctPositiveCount / allPredictedPositiveCount;
      }

      return aggregated + (correctPositiveCount == 0 ? 1 : 0);
    });
  }
}
