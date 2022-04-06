import 'package:ml_algo/src/tree_trainer/tree_node/split_predicate/from_predicate_type_json.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/split_predicate/predicate_type.dart';
import 'package:ml_algo/src/tree_trainer/tree_node/split_predicate/predicate_type_encoded_values.dart';
import 'package:test/test.dart';

void main() {
  group('fromSplittingPredicateTypeJson', () {
    test('should decode a `less than` predicate type', () {
      expect(
          fromPredicateTypeJson(lessThanEncodedValue), PredicateType.lessThan);
    });

    test('should decode a `less than or equal to` predicate type', () {
      expect(fromPredicateTypeJson(lessThanOrEqualToEncodedValue),
          PredicateType.lessThanOrEqualTo);
    });

    test('should decode a `equal to` predicate type', () {
      expect(fromPredicateTypeJson(equalToEncodedValue), PredicateType.equalTo);
    });

    test('should decode a `greater than orequal to` predicate type', () {
      expect(fromPredicateTypeJson(greaterThanOrEqualToEncodedValue),
          PredicateType.greaterThanOrEqualTo);
    });

    test('should decode a `greater than` predicate type', () {
      expect(fromPredicateTypeJson(greaterThanEncodedValue),
          PredicateType.greaterThan);
    });

    test('should return null if unknown string is passed', () {
      expect(fromPredicateTypeJson('unknown_string'), isNull);
    });

    test('should return null if null is passed', () {
      expect(fromPredicateTypeJson(null), isNull);
    });
  });
}
