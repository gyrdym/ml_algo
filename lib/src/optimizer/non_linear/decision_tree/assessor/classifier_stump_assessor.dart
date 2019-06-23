import 'dart:collection';
import 'dart:math' as math;

import 'package:ml_algo/src/optimizer/non_linear/decision_tree/assessor/stump_assessor.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:xrange/zrange.dart';

class ClassifierStumpAssessor implements StumpAssessor {
  const ClassifierStumpAssessor();

  @override
  double getErrorOnStump(Iterable<Matrix> stumpObservations,
      ZRange outcomesRange) {
    int errorCount = 0;
    int totalCount = 0;
    for (final nodeObservations in stumpObservations) {
      errorCount += _getErrorCount(nodeObservations
          .submatrix(columns: outcomesRange));
      totalCount += nodeObservations.rowsNum;
    }
    return errorCount / totalCount;
  }

  @override
  double getErrorOnNode(Matrix nodeObservations, ZRange outcomesRange) =>
      _getErrorCount(nodeObservations.submatrix(columns: outcomesRange)) /
          nodeObservations.rowsNum;

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
