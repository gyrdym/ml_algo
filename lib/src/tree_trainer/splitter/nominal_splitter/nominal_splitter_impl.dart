import 'package:ml_algo/src/tree_trainer/splitter/nominal_splitter/nominal_splitter.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/decision_tree_node/decision_tree_node.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/splitting_predicate/_helper/get_tree_node_splitting_predicate_by_type.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/splitting_predicate/tree_node_splitting_predicate_type.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/tree_node.dart';
import 'package:ml_linalg/matrix.dart';

class NominalTreeSplitterImpl implements NominalTreeSplitter {
  const NominalTreeSplitterImpl();

  @override
  Map<T, Matrix> split<T extends TreeNode>(
          Matrix samples, int splittingIdx, List<num> uniqueValues) =>
      Map.fromEntries(
        uniqueValues.map((value) {
          final splittingClauseType = TreeNodeSplittingPredicateType.equalTo;
          final splittingPredicate =
              getTreeNodeSplittingPredicateByType(splittingClauseType);

          final foundRows = samples.rows
              .where((row) => splittingPredicate(row, splittingIdx, value))
              .toList(growable: false);

          final node = DecisionTreeNode(
            splittingClauseType,
            value,
            splittingIdx,
            null,
            null,
          ) as T;

          return MapEntry(
            node,
            Matrix.fromRows(foundRows, dtype: samples.dtype),
          );
        }).where((entry) => entry.value.rowsNum > 0),
      );
}
