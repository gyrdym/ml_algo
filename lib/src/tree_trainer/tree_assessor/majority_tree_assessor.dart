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
      if (samples.columnCount == 0) {
        continue;
      }

      if (targetId >= samples.columnCount) {
        throw ArgumentError.value(
            targetId,
            'targetId',
            'the value should be in [0..${samples.columnCount - 1}] '
                'range, but given');
      }
      errorCount += _getErrorCount(samples.getColumn(targetId));
      totalCount += samples.rowCount;
    }

    return errorCount / totalCount;
  }

  @override
  double getError(Matrix samples, int targetId) =>
      _getErrorCount(samples.getColumn(targetId)) / samples.rowCount;

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
