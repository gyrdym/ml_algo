import 'package:ml_algo/src/tree_trainer/tree_node/splitting_predicate/tree_node_splitting_predicate_type.dart';

String formatPredicate(TreeNodeSplittingPredicateType? predicate) {
  switch (predicate) {
    case TreeNodeSplittingPredicateType.lessThan:
      return '&#60;';

    case TreeNodeSplittingPredicateType.lessThanOrEqualTo:
      return '&#8804;';

    case TreeNodeSplittingPredicateType.equalTo:
      return '==';

    case TreeNodeSplittingPredicateType.greaterThan:
      return '&#62;';

    case TreeNodeSplittingPredicateType.greaterThanOrEqualTo:
      return '&#8805;';

    default:
      return '-';
  }
}
