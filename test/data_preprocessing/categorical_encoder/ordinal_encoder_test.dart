import 'package:ml_algo/src/data_preprocessing/categorical_encoder/ordinal_encoder.dart';
import 'package:test/test.dart';

void main() {
  group('OrdinalEncoder', () {
    test('should encode categorical data, ordered collection of non-repeatable '
        'labels', () {
      final encoder = OrdinalEncoder();
      expect(encoder.encode(['group A', 'group B', 'group C', 'group D']),
        equals([
          [0],
          [1],
          [2],
          [3],
        ]),
      );
    });

    test('should encode categorical data, unordered collection of unrepeatable'
        'labels', () {
      final encoder = OrdinalEncoder();
      expect(encoder.encode(['group B', 'group D', 'group A', 'group C']),
        equals([
          [0],
          [1],
          [2],
          [3],
        ]),
      );
    });

    test('should encode categorical data, unordered collection of repeatable'
        'labels', () {
      final encoder = OrdinalEncoder();
      expect(encoder.encode(['group B', 'group D', 'group B', 'group A',
        'group C', 'group C', 'group A']),
        equals([
          [0],
          [1],
          [0],
          [2],
          [3],
          [3],
          [2],
        ]),
      );
    });

    test('should throw an error if an empty label list is passed', () {
      final encoder = OrdinalEncoder();
      expect(() => encoder.encode([]), throwsException);
    });
  });
}
