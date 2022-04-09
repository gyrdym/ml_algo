import 'package:ml_algo/src/tree_trainer/split_selector/split_selector.dart';
import 'package:ml_algo/src/tree_trainer/splitter/splitter.dart';
import 'package:ml_algo/src/tree_trainer/tree_assessor/tree_assessor.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/tree_node.dart';
import 'package:ml_linalg/matrix.dart';

class GreedyTreeSplitSelector implements TreeSplitSelector {
  GreedyTreeSplitSelector(this._assessor, this._splitter);

  final TreeAssessor _assessor;
  final TreeSplitter _splitter;

  @override
  Map<TreeNode, Matrix> select(
      Matrix samples, int targetId, Iterable<int> featuresColumnIdxs,
      [Map<int, List<num>>? columnIdToUniqueValues]) {
    final errors = <double, List<Map<TreeNode, Matrix>>>{};

    featuresColumnIdxs.forEach((colIdx) {
      final uniqueValues = columnIdToUniqueValues != null
          ? columnIdToUniqueValues[colIdx]
          : null;
      final split = _splitter.split(samples, colIdx, targetId, uniqueValues);
      final error = _assessor.getAggregatedError(split.values, targetId);

      errors.update(error, (splits) => splits..add(split),
          ifAbsent: () => [split]);
    });

    final sorted = errors.keys.toList(growable: false)..sort();
    final minError = sorted.first;

    return errors[minError]!.first;
  }
}
