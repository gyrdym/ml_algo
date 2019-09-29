import 'dart:collection';

import 'package:ml_algo/src/common/sequence_elements_distribution_calculator/distribution_calculator.dart';
import 'package:ml_algo/src/decision_tree_solver/decision_tree_leaf_label.dart';
import 'package:ml_algo/src/decision_tree_solver/leaf_label_factory/leaf_label_factory.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:quiver/iterables.dart' as quiver_iterables;

class MajorityLeafLabelFactory implements DecisionTreeLeafLabelFactory {
  MajorityLeafLabelFactory(this.distributionCalculator);

  final SequenceElementsDistributionCalculator distributionCalculator;

  @override
  DecisionTreeLeafLabel create(Matrix samples, int targetIdx) {
    final outcomes = samples.getColumn(targetIdx);
    final totalRecordsCount = outcomes.length;

    final labelData = _getLabelData<double>(outcomes, totalRecordsCount);
    return DecisionTreeLeafLabel(labelData.value,
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
