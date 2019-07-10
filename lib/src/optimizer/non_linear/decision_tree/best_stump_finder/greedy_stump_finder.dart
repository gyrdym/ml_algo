import 'package:ml_algo/src/optimizer/non_linear/decision_tree/best_stump_finder/best_stump_finder.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/decision_tree_node.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/split_assessor/split_assessor.dart';
import 'package:ml_algo/src/optimizer/non_linear/decision_tree/stump_factory/stump_factory.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:xrange/zrange.dart';

class GreedyStumpFinder implements BestStumpFinder {
  GreedyStumpFinder(this._assessor, this._stumpFactory);

  final SplitAssessor _assessor;
  final StumpFactory _stumpFactory;

  @override
  Map<DecisionTreeNode, Matrix> find(
      Matrix samples,
      ZRange outcomesColumnRange,
      Iterable<ZRange> featuresColumnRanges,
      [Map<ZRange, List<Vector>> rangeToNominalValues]) {
    final errors = <double, List<Map<DecisionTreeNode, Matrix>>>{};
    featuresColumnRanges.forEach((range) {
      final nominalValues = rangeToNominalValues != null
          ? rangeToNominalValues[range]
          : null;
      final split = _stumpFactory.create(samples, range,
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
