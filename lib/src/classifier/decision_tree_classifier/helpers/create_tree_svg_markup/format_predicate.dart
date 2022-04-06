import 'package:ml_algo/src/tree_trainer/tree_node/split_predicate/predicate_type.dart';

String formatPredicate(PredicateType predicate) {
  switch (predicate) {
    case PredicateType.lessThan:
      return '&#60;';

    case PredicateType.lessThanOrEqualTo:
      return '&#8804;';

    case PredicateType.equalTo:
      return '==';

    case PredicateType.greaterThan:
      return '&#62;';

    case PredicateType.greaterThanOrEqualTo:
      return '&#8805;';
  }
}
