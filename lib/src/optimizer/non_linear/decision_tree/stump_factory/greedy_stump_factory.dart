import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree_stump.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/split_assessor/split_assessor.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/stump_factory/samples_splitter/samples_splitter.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/stump_factory/stump_factory.dart';
import 'package:ml_linalg/axis.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:xrange/zrange.dart';

class GreedyStumpFactory implements StumpFactory {
  GreedyStumpFactory(this._assessor, this._nodeSplitter);

  final SplitAssessor _assessor;
  final SamplesSplitter _nodeSplitter;

  @override
  DecisionTreeStump create(Matrix observations, ZRange splittingColumnRange,
      ZRange outcomesRange, [List<Vector> categoricalValues]) =>
      categoricalValues != null
          ? _createByCategoricalValues(observations, splittingColumnRange,
            categoricalValues)
          : _createByNumber(observations, splittingColumnRange, outcomesRange);

  DecisionTreeStump _createByCategoricalValues(Matrix observations,
      ZRange splittingColumnRange, List<Vector> categoricalValues) {
    if (splittingColumnRange.firstValue < 0 ||
        splittingColumnRange.lastValue > observations.columnsNum) {
      throw Exception('Unappropriate range given: $splittingColumnRange, '
          'expected a range within or equal '
          '${ZRange.closed(0, observations.columnsNum)}');
    }
    final stumpObservations = categoricalValues.map((value) {
      final foundRows = observations.rows
          .where((row) => row.subvectorByRange(splittingColumnRange) == value)
          .toList(growable: false);
      return Matrix.fromRows(foundRows);
    }).where((node) => node.rowsNum > 0).toList(growable: false);

    return DecisionTreeStump(null, categoricalValues, splittingColumnRange,
        stumpObservations);
  }

  DecisionTreeStump _createByNumber(Matrix observations,
      ZRange splittingColumnRange, ZRange outcomesRange) {
    final selectedColumnIdx = splittingColumnRange.firstValue;
    final errors = <double, DecisionTreeStump>{};
    final sortedRows = observations
        .sort((row) => row[selectedColumnIdx], Axis.rows).rows;
    var prevValue = sortedRows.first[selectedColumnIdx];
    for (final row in sortedRows.skip(1)) {
      final splittingValue = (prevValue + row[selectedColumnIdx]) / 2;
      final stumpObservations = _nodeSplitter
          .split(observations, selectedColumnIdx, splittingValue);
      final error = _assessor
          .getAggregatedError(stumpObservations, outcomesRange);
      errors[error] = DecisionTreeStump(splittingValue, null,
          splittingColumnRange, stumpObservations);
      prevValue = row[selectedColumnIdx];
    }
    final sorted = errors.keys.toList(growable: false)
      ..sort();
    final minError = sorted.first;
    return errors[minError];
  }
}
