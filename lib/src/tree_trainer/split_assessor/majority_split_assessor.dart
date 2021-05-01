import 'dart:collection';
import 'dart:math' as math;

import 'package:ml_algo/src/tree_trainer/split_assessor/split_assessor.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class MajorityTreeSplitAssessor implements TreeSplitAssessor {
  const MajorityTreeSplitAssessor();

  @override
  double getAggregatedError(Iterable<Matrix> splitObservations, int targetId) {
    var errorCount = 0;
    var totalCount = 0;
    for (final nodeObservations in splitObservations) {
      if (nodeObservations.columnsNum == 0) {
        throw Exception('Observations on the node are empty');
      }
      if (targetId >= nodeObservations.columnsNum) {
        throw ArgumentError.value(
            targetId,
            'targetId',
            'the value should be in [0..${nodeObservations.columnsNum - 1}] '
                'range, but given');
      }
      errorCount += _getErrorCount(nodeObservations.getColumn(targetId));
      totalCount += nodeObservations.rowsNum;
    }
    return errorCount / totalCount;
  }

  @override
  double getError(Matrix splitObservations, int targetId) =>
      _getErrorCount(splitObservations.getColumn(targetId)) /
      splitObservations.rowsNum;

  int _getErrorCount(Vector outcomes) {
    if (outcomes.isEmpty) {
      throw Exception('Given node does not have any observations');
    }
    final majorityCount = _getMajorityCount<double>(outcomes);

    return outcomes.length - majorityCount;
  }

  int _getMajorityCount<T>(Iterable<T> iterable) {
    final bins = HashMap<T, int>();
    iterable.forEach((value) =>
        bins.update(value, (existing) => existing + 1, ifAbsent: () => 1));
    return bins.values.reduce(math.max);
  }
}
