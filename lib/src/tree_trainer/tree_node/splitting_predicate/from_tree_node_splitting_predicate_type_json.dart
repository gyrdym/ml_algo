import 'package:ml_algo/src/tree_trainer/tree_node/splitting_predicate/tree_node_splitting_predicate_type.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/splitting_predicate/tree_node_splitting_predicate_type_encoded_values.dart';

TreeNodeSplittingPredicateType fromSplittingPredicateTypeJson(String json) {
  switch (json) {
    case lessThanEncodedValue:
      return TreeNodeSplittingPredicateType.lessThan;

    case lessThanOrEqualToEncodedValue:
      return TreeNodeSplittingPredicateType.lessThanOrEqualTo;

    case equalToEncodedValue:
      return TreeNodeSplittingPredicateType.equalTo;

    case greaterThanOrEqualToEncodedValue:
      return TreeNodeSplittingPredicateType.greaterThanOrEqualTo;

    case greaterThanEncodedValue:
      return TreeNodeSplittingPredicateType.greaterThan;

    default:
      return null;
  }
}
