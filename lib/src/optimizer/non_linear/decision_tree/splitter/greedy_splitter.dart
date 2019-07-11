import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree_node.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/split_assessor/split_assessor.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/splitter/nominal_splitter/nominal_splitter.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/splitter/numerical_splitter/numerical_splitter.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/splitter/splitter.dart';
import 'package:ml_linalg/axis.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:xrange/zrange.dart';

class GreedySplitter implements Splitter {
  GreedySplitter(this._assessor, this._numericalSplitter,
      this._nominalSplitter);

  final SplitAssessor _assessor;
  final NumericalSplitter _numericalSplitter;
  final NominalSplitter _nominalSplitter;

  @override
  Map<DecisionTreeNode, Matrix> split(Matrix samples, ZRange splittingRange,
      ZRange outcomeColumnRange, [List<Vector> nominalValues]) =>
      nominalValues != null
          ? _createByNominalValues(samples, splittingRange,
            nominalValues)
          : _createByNumericalValue(samples, splittingRange,
          outcomeColumnRange);

  Map<DecisionTreeNode, Matrix> _createByNominalValues(Matrix samples,
      ZRange splittingRange, List<Vector> values) {
    if (splittingRange.firstValue < 0 ||
        splittingRange.lastValue > samples.columnsNum) {
      throw Exception('Unappropriate range given: $splittingRange, '
          'expected a range within or equal '
          '${ZRange.closed(0, samples.columnsNum)}');
    }
    return _nominalSplitter.split(samples,
        splittingRange, values);
  }

  Map<DecisionTreeNode, Matrix> _createByNumericalValue(Matrix samples,
      ZRange splittingRange, ZRange outcomesRange) {
    final selectedColumnIdx = splittingRange.firstValue;
    final errors = <double, Map<DecisionTreeNode, Matrix>>{};
    final sortedRows = samples
        .sort((row) => row[selectedColumnIdx], Axis.rows).rows;
    var prevValue = sortedRows.first[selectedColumnIdx];

    for (final row in sortedRows.skip(1)) {
      final splittingValue = (prevValue + row[selectedColumnIdx]) / 2;
      final split = _numericalSplitter
          .split(samples, splittingRange, splittingValue);
      final error = _assessor
          .getAggregatedError(split.values, outcomesRange);

      errors[error] = split;

      prevValue = row[selectedColumnIdx];
    }

    final sorted = errors.keys.toList(growable: false)
      ..sort();
    final minError = sorted.first;
    return errors[minError];
  }
}
