import 'dart:collection';
import 'dart:math' as math;

import 'package:ml_algo/src/tree_trainer/tree_assessor/tree_assessor.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class MajorityTreeAssessor implements TreeAssessor {
  const MajorityTreeAssessor();

  @override
  double getAggregatedError(Iterable<Matrix> split, int targetId) {
    var errorCount = 0;
    var totalCount = 0;

    for (final samples in split) {
      if (samples.columnsNum == 0) {
        continue;
      }

      if (targetId >= samples.columnsNum) {
        throw ArgumentError.value(
            targetId,
            'targetId',
            'the value should be in [0..${samples.columnsNum - 1}] '
                'range, but given');
      }
      errorCount += _getErrorCount(samples.getColumn(targetId));
      totalCount += samples.rowsNum;
    }

    return errorCount / totalCount;
  }

  @override
  double getError(Matrix samples, int targetId) =>
      _getErrorCount(samples.getColumn(targetId)) / samples.rowsNum;

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
