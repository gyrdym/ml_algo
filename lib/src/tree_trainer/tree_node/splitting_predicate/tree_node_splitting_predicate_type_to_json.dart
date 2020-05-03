import 'package:ml_algo/src/tree_trainer/tree_node/splitting_predicate/tree_node_splitting_predicate_type.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/splitting_predicate/tree_node_splitting_predicate_type_encoded_values.dart';

String splittingPredicateTypeToJson(TreeNodeSplittingPredicateType type) {
  switch (type) {
    case TreeNodeSplittingPredicateType.lessThan:
      return lessThanEncodedValue;

    case TreeNodeSplittingPredicateType.lessThanOrEqualTo:
      return lessThanOrEqualToEncodedValue;

    case TreeNodeSplittingPredicateType.equalTo:
      return equalToEncodedValue;

    case TreeNodeSplittingPredicateType.greaterThanOrEqualTo:
      return greaterThanOrEqualToEncodedValue;

    case TreeNodeSplittingPredicateType.greaterThan:
      return greaterThanEncodedValue;

    default:
      return null;
  }
}
