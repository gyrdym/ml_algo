import 'package:ml_algo/src/tree_trainer/tree_node/split_predicate/predicate_type.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/split_predicate/predicate_type_encoded_values.dart';

PredicateType? fromPredicateTypeJson(String? json) {
  switch (json) {
    case lessThanEncodedValue:
      return PredicateType.lessThan;

    case lessThanOrEqualToEncodedValue:
      return PredicateType.lessThanOrEqualTo;

    case equalToEncodedValue:
      return PredicateType.equalTo;

    case greaterThanOrEqualToEncodedValue:
      return PredicateType.greaterThanOrEqualTo;

    case greaterThanEncodedValue:
      return PredicateType.greaterThan;

    default:
      return null;
  }
}
