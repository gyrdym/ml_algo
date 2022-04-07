import 'package:ml_algo/src/tree_trainer/tree_node/split_predicate/predicate_type.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/split_predicate/predicate_type_encoded_values.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/split_predicate/predicate_type_to_json.dart';
import 'package:test/test.dart';

void main() {
  group('splittingPredicateTypeToJson', () {
    test('should encode `less than` type', () {
      final encoded = predicateTypeToJson(PredicateType.lessThan);
      expect(encoded, lessThanEncodedValue);
    });

    test('should encode `less than or equal to` type', () {
      final encoded = predicateTypeToJson(PredicateType.lessThanOrEqualTo);
      expect(encoded, lessThanOrEqualToEncodedValue);
    });

    test('should encode `equal to` type', () {
      final encoded = predicateTypeToJson(PredicateType.equalTo);
      expect(encoded, equalToEncodedValue);
    });

    test('should encode `greater than or equal to` type', () {
      final encoded = predicateTypeToJson(PredicateType.greaterThanOrEqualTo);
      expect(encoded, greaterThanOrEqualToEncodedValue);
    });

    test('should encode `greater than` type', () {
      final encoded = predicateTypeToJson(PredicateType.greaterThan);
      expect(encoded, greaterThanEncodedValue);
    });

    test('should return null if null is passed', () {
      final encoded = predicateTypeToJson(null);
      expect(encoded, isNull);
    });
  });
}
