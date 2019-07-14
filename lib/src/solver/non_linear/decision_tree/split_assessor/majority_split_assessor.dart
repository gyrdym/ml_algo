import 'dart:collection';
import 'dart:math' as math;

import 'package:ml_algo/src/solver/non_linear/decision_tree/split_assessor/split_assessor.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:xrange/zrange.dart';

class MajoritySplitAssessor implements SplitAssessor {
  const MajoritySplitAssessor();

  @override
  double getAggregatedError(Iterable<Matrix> splitObservations,
      ZRange outcomesRange) {
    int errorCount = 0;
    int totalCount = 0;
    for (final nodeObservations in splitObservations) {
      errorCount += _getErrorCount(nodeObservations
          .submatrix(columns: outcomesRange));
      totalCount += nodeObservations.rowsNum;
    }
    return errorCount / totalCount;
  }

  @override
  double getError(Matrix splitObservations, ZRange outcomesRange) =>
      _getErrorCount(splitObservations.submatrix(columns: outcomesRange)) /
          splitObservations.rowsNum;

  int _getErrorCount(Matrix outcomes) {
    if (outcomes.rowsNum == 0) {
      throw Exception('Given node does not have any observations');
    }
    final majorityCount = outcomes.columnsNum == 1
        ? _getMajorityCount<double>(outcomes.toVector())
        : _getMajorityCount<Vector>(outcomes.rows);
    return outcomes.rowsNum - majorityCount;
  }

  int _getMajorityCount<T>(Iterable<T> iterable) {
    final bins = HashMap<T, int>();
    iterable.forEach((value) =>
        bins.update(value, (existing) => existing + 1, ifAbsent: () => 1));
    return bins.values.reduce(math.max);
  }
}
