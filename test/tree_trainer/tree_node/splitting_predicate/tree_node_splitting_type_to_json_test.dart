import 'package:ml_algo/src/tree_trainer/tree_node/splitting_predicate/tree_node_splitting_predicate_type.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/splitting_predicate/tree_node_splitting_predicate_type_encoded_values.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/splitting_predicate/tree_node_splitting_predicate_type_to_json.dart';
import 'package:test/test.dart';

void main() {
  group('splittingPredicateTypeToJson', () {
    test('should encode `less than` type', () {
      final encoded = splittingPredicateTypeToJson(
          TreeNodeSplittingPredicateType.lessThan);
      expect(encoded, lessThanEncodedValue);
    });

    test('should encode `less than or equal to` type', () {
      final encoded = splittingPredicateTypeToJson(
          TreeNodeSplittingPredicateType.lessThanOrEqualTo);
      expect(encoded, lessThanOrEqualToEncodedValue);
    });

    test('should encode `equal to` type', () {
      final encoded = splittingPredicateTypeToJson(
          TreeNodeSplittingPredicateType.equalTo);
      expect(encoded, equalToEncodedValue);
    });

    test('should encode `greater than or equal to` type', () {
      final encoded = splittingPredicateTypeToJson(
          TreeNodeSplittingPredicateType.greaterThanOrEqualTo);
      expect(encoded, greaterThanOrEqualToEncodedValue);
    });

    test('should encode `greater than` type', () {
      final encoded = splittingPredicateTypeToJson(
          TreeNodeSplittingPredicateType.greaterThan);
      expect(encoded, greaterThanEncodedValue);
    });

    test('should return null if null is passed', () {
      final encoded = splittingPredicateTypeToJson(null);
      expect(encoded, isNull);
    });
  });
}
