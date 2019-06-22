import 'dart:collection';
import 'dart:math' as math;

import 'package:ml_algo/src/optimizer/non_linear/decision_tree/assessor/stump_assessor.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class ClassifierStumpAssessor implements StumpAssessor {
  @override
  int getErrorOnStump(Iterable<Matrix> stump) => stump.fold(0,
          (error, observations) => error += getErrorOnNode(observations));

  @override
  int getErrorOnNode(Matrix observations) {
    if (observations.rowsNum == 0) {
      throw Exception('Given node does not have any observations');
    }
    return observations.columnsNum == 1
        ? _getError<double>(observations.toVector(), observations.rowsNum)
        : _getError<Vector>(observations.rows, observations.rowsNum);
  }

  int _getError<T>(Iterable<T> iterable, int length) {
    final maxCount = _getMajorityCount<T>(iterable);
    return length - maxCount;
  }

  int _getMajorityCount<T>(Iterable<T> iterable) {
    final bins = HashMap<T, int>();
    iterable.forEach((value) =>
        bins.update(value, (existing) => existing + 1, ifAbsent: () => 1));
    return bins.values.reduce(math.max);
  }
}
