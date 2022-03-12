import 'package:ml_algo/src/tree_trainer/tree_node/splitting_predicate/tree_node_splitting_predicate_type.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/tree_node.dart';

abstract class IntermediateTreeNodeFactory {
  TreeNode create(
    TreeNodeSplittingPredicateType predicateType,
    num splittingValue,
    int splittingIdx,
  );
}
