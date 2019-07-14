import 'dart:collection';

import 'package:ml_algo/src/common/sequence_elements_distribution_calculator/distribution_calculator.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree_leaf_label.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/leaf_label_factory/leaf_label_factory.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:quiver/iterables.dart' as quiver_iterables;
import 'package:xrange/zrange.dart';

class MajorityLeafLabelFactory implements DecisionTreeLeafLabelFactory {
  MajorityLeafLabelFactory(this.distributionCalculator);

  final SequenceElementsDistributionCalculator distributionCalculator;

  @override
  DecisionTreeLeafLabel create(Matrix samples, ZRange outcomesColumnRange,
      bool isClassLabelNominal) {
    final outcomes = samples.submatrix(columns: outcomesColumnRange);
    final totalRecordsCount = outcomes.rowsNum;

    if (isClassLabelNominal) {
      final labelData = _getLabelData<Vector>(outcomes.rows, totalRecordsCount);
      return DecisionTreeLeafLabel.nominal(labelData.value,
          probability: labelData.probability);
    }

    final labelColumn = outcomes.toVector();
    final labelData = _getLabelData<double>(labelColumn, totalRecordsCount);
    return DecisionTreeLeafLabel.numerical(labelData.value,
        probability: labelData.probability);
  }

  _LabelData<T> _getLabelData<T>(Iterable<T> values, int totalCount) {
    final distribution = distributionCalculator
        .calculate<T>(values, totalCount);
    final targetLabelEntry = _findEntryWithMaxProbability(distribution);
    return _LabelData<T>(targetLabelEntry.key, targetLabelEntry.value);
  }

  MapEntry<T, double> _findEntryWithMaxProbability<T>(
      HashMap<T, double> distribution) =>
    quiver_iterables.max<MapEntry<T, double>>(
      distribution.entries, (first, second) {
        final diff = first.value - second.value;
        if (diff < 0) return -1;
        if (diff > 0) return 1;
        return 0;
      },
    );
}

class _LabelData<T> {
  _LabelData(this.value, this.probability);

  final T value;
  final double probability;
}
