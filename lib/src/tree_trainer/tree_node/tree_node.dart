import 'package:ml_algo/src/tree_trainer/leaf_label/leaf_label.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/splitting_predicate/tree_node_splitting_predicate.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/splitting_predicate/tree_node_splitting_predicate_type.dart';
import 'package:ml_linalg/vector.dart';

class TreeNode {
  TreeNode(
      this.predicateType,
      this.splittingValue,
      this.splittingIndex,
      this.children,
      this.label,
      [
        this.level = 0,
      ]
  );

  final List<TreeNode> children;
  final TreeLeafLabel label;
  final TreeNodeSplittingPredicateType predicateType;
  final num splittingValue;
  final int splittingIndex;
  final int level;

  bool get isLeaf => children == null || children.isEmpty;

  bool isSamplePassed(Vector sample) =>
      getSplittingPredicateByType(predicateType)(
        sample,
        splittingIndex,
        splittingValue,
      );
}
