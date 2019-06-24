import 'package:ml_algo/src/optimizer/non_linear/decision_tree/assessor/stump_assessor.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/stump_selector/node_splitter/node_splitter.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/stump_selector/stump_selector.dart';
import 'package:ml_linalg/axis.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:xrange/zrange.dart';

class GreedyStumpSelector implements StumpSelector {
  GreedyStumpSelector(this._assessor, this._nodeSplitter);

  final StumpAssessor _assessor;
  final NodeSplitter _nodeSplitter;

  @override
  Iterable<Matrix> select(Matrix observations, ZRange splittingFeatureRange,
      ZRange outcomesRange, [List<Vector> categoricalValues]) =>
      categoricalValues != null
          ? _selectByCategoricalValues(observations, splittingFeatureRange,
            categoricalValues)
          : _selectByNumber(observations, splittingFeatureRange.firstValue,
            outcomesRange);

  List<Matrix> _selectByCategoricalValues(Matrix observations,
      ZRange splittingColumnRange, List<Vector> splittingValues) {
    if (splittingColumnRange.firstValue < 0 ||
        splittingColumnRange.lastValue > observations.columnsNum) {
      throw Exception('Unappropriate range given: $splittingColumnRange, '
          'expected a range within or equal '
          '${ZRange.closed(0, observations.columnsNum)}');
    }
    return splittingValues.map((value) {
      final foundRows = observations.rows
          .where((row) => row.subvectorByRange(splittingColumnRange) == value)
          .toList(growable: false);
      return Matrix.fromRows(foundRows);
    }).where((node) => node.rowsNum > 0).toList(growable: false);
  }

  List<Matrix> _selectByNumber(Matrix observations, int selectedColumnIdx,
      ZRange outcomesRange) {
    final errors = <double, List<Matrix>>{};
    final sortedRows = observations
        .sort((row) => row[selectedColumnIdx], Axis.rows).rows;
    var prevValue = sortedRows.first[selectedColumnIdx];
    for (final row in sortedRows.skip(1)) {
      final splittingValue = (prevValue + row[selectedColumnIdx]) / 2;
      final stump = _nodeSplitter
          .split(observations, selectedColumnIdx, splittingValue);
      final error = _assessor.getErrorOnStump(stump, outcomesRange);
      errors[error] = stump;
      prevValue = row[selectedColumnIdx];
    }
    final sorted = errors.keys.toList(growable: false)
      ..sort();
    final minError = sorted.first;
    return errors[minError];
  }
}
