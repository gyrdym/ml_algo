import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree_leaf_label.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/leaf_label_factory/leaf_label_factory.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/observations_distribution_counter/distribution_counter.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:xrange/zrange.dart';
import 'package:quiver/iterables.dart' as quiver_iterables;

class MajorityLeafLabelFactory implements DecisionTreeLeafLabelFactory {
  MajorityLeafLabelFactory(this.distributionCounter);

  final ObservationsDistributionCounter distributionCounter;

  @override
  DecisionTreeLeafLabel create(Matrix observations, ZRange outcomesColumnRange,
      bool isClassLabelCategorical) {
    final outcomes = observations.submatrix(columns: outcomesColumnRange);

    if (isClassLabelCategorical) {
      final leafLabelValue = _getLabelValue<Vector>(outcomes.rows);
      return DecisionTreeLeafLabel.categorical(leafLabelValue);
    }

    final leafLabelValue = _getLabelValue<double>(outcomes.toVector());
    return DecisionTreeLeafLabel.numerical(leafLabelValue);
  }

  T _getLabelValue<T>(Iterable<T> values) {
    final distribution = distributionCounter.count<T>(values);
    final targetValue = quiver_iterables.max<MapEntry<T, int>>(
      distribution.entries,
      (first, second) => first.value - second.value,
    );
    return targetValue.key;
  }
}
