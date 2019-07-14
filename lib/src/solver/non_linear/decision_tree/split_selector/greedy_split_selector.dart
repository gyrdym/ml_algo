import 'package:ml_algo/src/solver/non_linear/decision_tree/decision_tree_node.dart';
import 'package:ml_algo/src/solver/non_linear/decision_tree/split_assessor/split_assessor.dart';
import 'package:ml_algo/src/solver/non_linear/decision_tree/split_selector/split_selector.dart';
import 'package:ml_algo/src/solver/non_linear/decision_tree/splitter/splitter.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:xrange/zrange.dart';

class GreedySplitSelector implements SplitSelector {
  GreedySplitSelector(this._assessor, this._splitter);

  final SplitAssessor _assessor;
  final Splitter _splitter;

  @override
  Map<DecisionTreeNode, Matrix> select(
      Matrix samples,
      ZRange outcomesColumnRange,
      Iterable<ZRange> featuresColumnRanges,
      [Map<ZRange, List<Vector>> rangeToNominalValues]) {
    final errors = <double, List<Map<DecisionTreeNode, Matrix>>>{};
    featuresColumnRanges.forEach((range) {
      final nominalValues = rangeToNominalValues != null
          ? rangeToNominalValues[range]
          : null;
      final split = _splitter.split(samples, range,
          outcomesColumnRange, nominalValues);
      final error = _assessor.getAggregatedError(split.values,
          outcomesColumnRange);
      errors.update(error, (splits) => splits..add(split),
          ifAbsent: () => [split]);
    });
    final sorted = errors.keys.toList(growable: false)
      ..sort();
    final minError = sorted.first;
    return errors[minError].first;
  }
}
