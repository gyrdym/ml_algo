import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:quiver/iterables.dart';

double divideTruePositiveBy(
    Vector divider,
    Matrix origLabels,
    Matrix predictedLabels,
) {
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
  // we may multiply predicted labels by 2, and then subtract the two
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
  final truePositiveCounts = difference
      .reduceRows(
          (counts, row) => counts + row.mapToVector((diff) => diff == -1
          ? 1 : 0),
      initValue: Vector.zero(
        origLabels.columnsNum,
        dtype: origLabels.dtype,
      ));
  final aggregatedScore = (truePositiveCounts / divider).mean();

  if (aggregatedScore.isFinite) {
    return aggregatedScore;
  }

  return zip([
    truePositiveCounts,
    divider,
  ]).fold(0, (aggregated, pair) {
    final truePositiveCount = pair.first;
    final dividerElement = pair.last;

    if (dividerElement != 0) {
      return aggregated + truePositiveCount / dividerElement;
    }

    return aggregated + (truePositiveCount == 0 ? 1 : 0);
  });
}
