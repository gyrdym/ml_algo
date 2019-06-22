import 'dart:collection';
import 'dart:math' as math;

import 'package:ml_algo/src/optimizer/non_linear/decision_tree/assessor/stump_assessor.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class ClassifierStumpAssessor implements StumpAssessor {
  const ClassifierStumpAssessor();

  @override
  double getErrorOnStump(Iterable<Matrix> stump) {
    int errorCount = 0;
    int totalCount = 0;
    for (final node in stump) {
      errorCount += _getErrorCount(node);
      totalCount += node.rowsNum;
    }
    return errorCount / totalCount;
  }

  @override
  double getErrorOnNode(Matrix observations) =>
      _getErrorCount(observations) / observations.rowsNum;

  int _getErrorCount(Matrix observations) {
    if (observations.rowsNum == 0) {
      throw Exception('Given node does not have any observations');
    }
    final majorityCount = observations.columnsNum == 1
        ? _getMajorityCount<double>(observations.toVector())
        : _getMajorityCount<Vector>(observations.rows);
    return observations.rowsNum - majorityCount;
  }

  int _getMajorityCount<T>(Iterable<T> iterable) {
    final bins = HashMap<T, int>();
    iterable.forEach((value) =>
        bins.update(value, (existing) => existing + 1, ifAbsent: () => 1));
    return bins.values.reduce(math.max);
  }
}
