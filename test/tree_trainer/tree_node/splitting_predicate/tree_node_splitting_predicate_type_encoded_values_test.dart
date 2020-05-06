import 'package:ml_algo/src/tree_trainer/tree_node/splitting_predicate/tree_node_splitting_predicate_type_encoded_values.dart';
import 'package:test/test.dart';

void main() {
  group('Tree node splitting predicate type encoded values', () {
    test('should have value for `less than` type', () {
      expect(lessThanEncodedValue, 'LT');
    });

    test('should have value for `less than or equal to` type', () {
      expect(lessThanOrEqualToEncodedValue, 'LET');
    });

    test('should have value for `equal to` type', () {
      expect(equalToEncodedValue, 'ET');
    });

    test('should have value for `greater than or equal to` type', () {
      expect(greaterThanOrEqualToEncodedValue, 'GET');
    });

    test('should have value for `greater than` type', () {
      expect(greaterThanEncodedValue, 'GT');
    });
  });
}
