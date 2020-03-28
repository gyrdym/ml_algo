import 'package:ml_algo/src/tree_trainer/tree_node/splitting_predicate/tree_node_splitting_predicate_type.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/splitting_predicate/tree_node_splitting_predicate_type_serializable_value.dart';

String serialize(TreeNodeSplittingPredicateType type) {
  switch (type) {
    case TreeNodeSplittingPredicateType.lessThan:
      return treeNodeSplittingPredicateTypeLessThan;

    case TreeNodeSplittingPredicateType.lessThanOrEqualTo:
      return treeNodeSplittingPredicateTypeLessThanOrEqualTo;

    case TreeNodeSplittingPredicateType.equalTo:
      return treeNodeSplittingPredicateTypeEqualTo;

    case TreeNodeSplittingPredicateType.greaterThanOrEqualTo:
      return treeNodeSplittingPredicateTypeGreaterThanOrEqualTo;

    case TreeNodeSplittingPredicateType.greaterThan:
      return treeNodeSplittingPredicateTypeGreaterThan;

    default:
      throw UnsupportedError('Tree node splitting predicate type $type is not '
          'supported');
  }
}
