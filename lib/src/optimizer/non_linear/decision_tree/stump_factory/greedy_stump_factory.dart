import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree_stump.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/nominal_splitter/nominal_splitter.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/numerical_splitter/numerical_splitter.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/split_assessor/split_assessor.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/stump_factory/stump_factory.dart';
import 'package:ml_linalg/axis.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:xrange/zrange.dart';

class GreedyStumpFactory implements StumpFactory {
  GreedyStumpFactory(this._assessor, this._numericalSplitter,
      this._nominalSplitter);

  final SplitAssessor _assessor;
  final NumericalSplitter _numericalSplitter;
  final NominalSplitter _nominalSplitter;

  @override
  DecisionTreeStump create(Matrix samples, ZRange splittingRange,
      ZRange outcomeColumnRange, [List<Vector> nominalValues]) =>
      nominalValues != null
          ? _createByNominalValues(samples, splittingRange,
            nominalValues)
          : _createByNumericalValue(samples, splittingRange,
          outcomeColumnRange);

  DecisionTreeStump _createByNominalValues(Matrix samples,
      ZRange splittingRange, List<Vector> values) {
    if (splittingRange.firstValue < 0 ||
        splittingRange.lastValue > samples.columnsNum) {
      throw Exception('Unappropriate range given: $splittingRange, '
          'expected a range within or equal '
          '${ZRange.closed(0, samples.columnsNum)}');
    }
    final stumpObservations = _nominalSplitter.split(samples,
        splittingRange, values);
    return DecisionTreeStump(null, values, splittingRange,
        stumpObservations);
  }

  DecisionTreeStump _createByNumericalValue(Matrix samples,
      ZRange splittingRange, ZRange outcomesRange) {
    final selectedColumnIdx = splittingRange.firstValue;
    final errors = <double, DecisionTreeStump>{};
    final sortedRows = samples
        .sort((row) => row[selectedColumnIdx], Axis.rows).rows;
    var prevValue = sortedRows.first[selectedColumnIdx];

    for (final row in sortedRows.skip(1)) {
      final splittingValue = (prevValue + row[selectedColumnIdx]) / 2;
      final stumpObservations = _numericalSplitter
          .split(samples, selectedColumnIdx, splittingValue);
      final error = _assessor
          .getAggregatedError(stumpObservations, outcomesRange);

      errors[error] = DecisionTreeStump(splittingValue, null,
          splittingRange, stumpObservations);

      prevValue = row[selectedColumnIdx];
    }

    final sorted = errors.keys.toList(growable: false)
      ..sort();
    final minError = sorted.first;
    return errors[minError];
  }
}
