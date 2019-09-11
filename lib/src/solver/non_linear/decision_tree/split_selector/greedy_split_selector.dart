import 'package:ml_algo/src/solver/non_linear/decision_tree/decision_tree_node.dart';
import 'package:ml_algo/src/solver/non_linear/decision_tree/split_assessor/split_assessor.dart';
import 'package:ml_algo/src/solver/non_linear/decision_tree/split_selector/split_selector.dart';
import 'package:ml_algo/src/solver/non_linear/decision_tree/splitter/splitter.dart';
import 'package:ml_linalg/matrix.dart';

class GreedySplitSelector implements SplitSelector {
  GreedySplitSelector(this._assessor, this._splitter);

  final SplitAssessor _assessor;
  final Splitter _splitter;

  @override
  Map<DecisionTreeNode, Matrix> select(
      Matrix samples,
      int targetId,
      Iterable<int> featuresColumnIdxs,
      [Map<int, List<num>> columnIdToUniqueValues]) {
    final errors = <double, List<Map<DecisionTreeNode, Matrix>>>{};
    featuresColumnIdxs.forEach((colIdx) {
      final uniqueValues = columnIdToUniqueValues != null
          ? columnIdToUniqueValues[colIdx]
          : null;
      final split = _splitter.split(samples, colIdx, targetId, uniqueValues);
      final error = _assessor.getAggregatedError(split.values, targetId);
      errors.update(error, (splits) => splits..add(split),
          ifAbsent: () => [split]);
    });
    final sorted = errors.keys.toList(growable: false)
      ..sort();
    final minError = sorted.first;
    return errors[minError].first;
  }
}
