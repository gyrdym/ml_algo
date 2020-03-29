import 'package:ml_algo/src/tree_trainer/tree_node/splitting_predicate/tree_node_splitting_predicate_type.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/splitting_predicate/tree_node_splitting_predicate_type_serializable_value.dart';

TreeNodeSplittingPredicateType deserialize(String serialized) {
  if (serialized == '') {
    return null;
  }

  switch (serialized) {
    case treeNodeSplittingPredicateTypeLessThan:
      return TreeNodeSplittingPredicateType.lessThan;

    case treeNodeSplittingPredicateTypeLessThanOrEqualTo:
      return TreeNodeSplittingPredicateType.lessThanOrEqualTo;

    case treeNodeSplittingPredicateTypeEqualTo:
      return TreeNodeSplittingPredicateType.equalTo;

    case treeNodeSplittingPredicateTypeGreaterThanOrEqualTo:
      return TreeNodeSplittingPredicateType.greaterThanOrEqualTo;

    case treeNodeSplittingPredicateTypeGreaterThan:
      return TreeNodeSplittingPredicateType.greaterThan;

    default:
      throw UnsupportedError('Failed to deserialize string `$serialized`');
  }
}
