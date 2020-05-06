import 'package:ml_algo/src/tree_trainer/tree_node/splitting_predicate/from_tree_node_splitting_predicate_type_json.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/splitting_predicate/tree_node_splitting_predicate_type.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/splitting_predicate/tree_node_splitting_predicate_type_encoded_values.dart';
import 'package:test/test.dart';

void main() {
  group('fromSplittingPredicateTypeJson', () {
    test('should decode a `less than` predicate type', () {
      expect(fromSplittingPredicateTypeJson(lessThanEncodedValue),
          TreeNodeSplittingPredicateType.lessThan);
    });

    test('should decode a `less than or equal to` predicate type', () {
      expect(fromSplittingPredicateTypeJson(lessThanOrEqualToEncodedValue),
          TreeNodeSplittingPredicateType.lessThanOrEqualTo);
    });

    test('should decode a `equal to` predicate type', () {
      expect(fromSplittingPredicateTypeJson(equalToEncodedValue),
          TreeNodeSplittingPredicateType.equalTo);
    });

    test('should decode a `greater than orequal to` predicate type', () {
      expect(fromSplittingPredicateTypeJson(greaterThanOrEqualToEncodedValue),
          TreeNodeSplittingPredicateType.greaterThanOrEqualTo);
    });

    test('should decode a `greater than` predicate type', () {
      expect(fromSplittingPredicateTypeJson(greaterThanEncodedValue),
          TreeNodeSplittingPredicateType.greaterThan);
    });

    test('should return null if unknown string is passed', () {
      expect(fromSplittingPredicateTypeJson('unknown_string'), isNull);
    });

    test('should return null if null is passed', () {
      expect(fromSplittingPredicateTypeJson(null), isNull);
    });
  });
}
