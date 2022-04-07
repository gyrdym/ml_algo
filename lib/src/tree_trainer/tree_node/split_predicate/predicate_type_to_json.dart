import 'package:ml_algo/src/tree_trainer/tree_node/split_predicate/predicate_type.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/split_predicate/predicate_type_encoded_values.dart';

String? predicateTypeToJson(PredicateType? type) {
  switch (type) {
    case PredicateType.lessThan:
      return lessThanEncodedValue;

    case PredicateType.lessThanOrEqualTo:
      return lessThanOrEqualToEncodedValue;

    case PredicateType.equalTo:
      return equalToEncodedValue;

    case PredicateType.greaterThanOrEqualTo:
      return greaterThanOrEqualToEncodedValue;

    case PredicateType.greaterThan:
      return greaterThanEncodedValue;

    default:
      return null;
  }
}
