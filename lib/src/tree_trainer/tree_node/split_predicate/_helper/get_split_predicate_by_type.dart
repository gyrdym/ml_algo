import 'package:ml_algo/src/tree_trainer/tree_node/split_predicate/predicate_type.dart';
import 'package:ml_linalg/vector.dart';

typedef SplittingPredicate = bool Function(
    Vector sample, int splittingIdx, num value);

SplittingPredicate getSplitPredicateByType(PredicateType type) {
  switch (type) {
    case PredicateType.lessThan:
      return _lessThanClause;

    case PredicateType.lessThanOrEqualTo:
      return _lessThanOrEqualToClause;

    case PredicateType.equalTo:
      return _equalToClause;

    case PredicateType.greaterThanOrEqualTo:
      return _greaterThanOrEqualToClause;

    case PredicateType.greaterThan:
      return _greaterThanClause;

    default:
      throw UnsupportedError('Splitting clause type $type is not supported');
  }
}

bool _lessThanClause(Vector sample, int splittingIdx, num value) =>
    sample[splittingIdx] < value;

bool _lessThanOrEqualToClause(Vector sample, int splittingIdx, num value) =>
    sample[splittingIdx] <= value;

bool _equalToClause(Vector sample, int splittingIdx, num value) =>
    sample[splittingIdx] == value;

bool _greaterThanOrEqualToClause(Vector sample, int splittingIdx, num value) =>
    sample[splittingIdx] >= value;

bool _greaterThanClause(Vector sample, int splittingIdx, num value) =>
    sample[splittingIdx] > value;
