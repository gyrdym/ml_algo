import 'package:ml_algo/src/tree_trainer/tree_node/intermediate_tree_node_factory.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/kd_tree_node/kd_tree_node.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/splitting_predicate/tree_node_splitting_predicate_type.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/tree_node.dart';

class KDIntermediateTreeNodeFactory implements IntermediateTreeNodeFactory {
  const KDIntermediateTreeNodeFactory();

  @override
  TreeNode create(TreeNodeSplittingPredicateType predicateType,
      num splittingValue, int splittingIdx) {
    return KDTreeNode(predicateType, splittingValue, splittingIdx, null, null);
  }
}
