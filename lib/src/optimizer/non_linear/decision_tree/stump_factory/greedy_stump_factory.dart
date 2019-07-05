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
  GreedyStumpFactory(this._assessor, this._byNumericalValueSplitter,
      this._byNominalValueSplitter);

  final SplitAssessor _assessor;
  final NumericalSplitter _byNumericalValueSplitter;
  final NominalSplitter _byNominalValueSplitter;

  @override
  DecisionTreeStump create(Matrix samples, ZRange splittingColumnRange,
      ZRange outcomeColumnRange, [List<Vector> nominalValues]) =>
      nominalValues != null
          ? _createByNominalValues(samples, splittingColumnRange,
            nominalValues)
          : _createByNumericalValue(samples, splittingColumnRange,
          outcomeColumnRange);

  DecisionTreeStump _createByNominalValues(Matrix samples,
      ZRange splittingColumnRange, List<Vector> nominalValues) {
    if (splittingColumnRange.firstValue < 0 ||
        splittingColumnRange.lastValue > samples.columnsNum) {
      throw Exception('Unappropriate range given: $splittingColumnRange, '
          'expected a range within or equal '
          '${ZRange.closed(0, samples.columnsNum)}');
    }
    final stumpObservations = _byNominalValueSplitter.split(samples,
        splittingColumnRange, nominalValues);
    return DecisionTreeStump(null, nominalValues, splittingColumnRange,
        stumpObservations);
  }

  DecisionTreeStump _createByNumericalValue(Matrix observations,
      ZRange splittingColumnRange, ZRange outcomesRange) {
    final selectedColumnIdx = splittingColumnRange.firstValue;
    final errors = <double, DecisionTreeStump>{};
    final sortedRows = observations
        .sort((row) => row[selectedColumnIdx], Axis.rows).rows;
    var prevValue = sortedRows.first[selectedColumnIdx];

    for (final row in sortedRows.skip(1)) {
      final splittingValue = (prevValue + row[selectedColumnIdx]) / 2;
      final stumpObservations = _byNumericalValueSplitter
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
