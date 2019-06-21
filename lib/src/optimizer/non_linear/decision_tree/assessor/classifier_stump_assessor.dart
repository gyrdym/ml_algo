import 'dart:collection';
import 'dart:math' as math;

import 'package:ml_algo/src/optimizer/non_linear/decision_tree/assessor/stump_assessor.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class ClassifierStumpAssessor implements StumpAssessor {
  @override
  double getErrorOnStump(Iterable<Matrix> stump) => stump.fold(0,
          (error, observations) => error += getErrorOnNode(observations));

  @override
  double getErrorOnNode(Matrix observations) {
    if (observations.rowsNum == 0) {
      throw Exception('Given node does not have any observations');
    }
    return observations.columnsNum == 1
        ? _getError<double>(observations.toVector(), observations.rowsNum)
        : _getError<Vector>(observations.rows, observations.rowsNum);
  }

  double _getError<T>(Iterable<T> iterable, int length) {
    final bins = HashMap<T, int>();
    iterable.forEach((value) =>
        bins.update(value, (existing) => existing + 1, ifAbsent: () => 1));
    final maxCount = bins.values.reduce(math.max);
    return (length - maxCount) / length;
  }
}
