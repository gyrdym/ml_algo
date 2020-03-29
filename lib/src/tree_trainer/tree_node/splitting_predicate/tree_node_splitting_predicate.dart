import 'package:ml_algo/src/tree_trainer/tree_node/splitting_predicate/tree_node_splitting_predicate_type.dart';
import 'package:ml_linalg/vector.dart';

typedef SplittingPredicate = bool Function(Vector sample, int splittingIdx,
    num value);

SplittingPredicate getSplittingPredicateByType(
    TreeNodeSplittingPredicateType type) {

  switch (type) {
    case TreeNodeSplittingPredicateType.lessThan:
      return _lessThanClause;

    case TreeNodeSplittingPredicateType.lessThanOrEqualTo:
      return _lessThanOrEqualToClause;

    case TreeNodeSplittingPredicateType.equalTo:
      return _equalToClause;

    case TreeNodeSplittingPredicateType.greaterThanOrEqualTo:
      return _greaterThanOrEqualToClause;

    case TreeNodeSplittingPredicateType.greaterThan:
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

bool _greaterThanClause(Vector sample, int splittingIdx, num value) =>
    sample[splittingIdx] > value;

bool _greaterThanOrEqualToClause(Vector sample, int splittingIdx,
    num value) => sample[splittingIdx] >= value;
