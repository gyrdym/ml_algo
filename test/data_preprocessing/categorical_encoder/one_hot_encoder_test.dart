import 'package:ml_algo/src/data_preprocessing/categorical_encoder/one_hot_encoder.dart';
import 'package:test/test.dart';

void main() {
  group('OneHotEncoder', () {
    test('should encode categorical data, ordered collection of non-repeatable '
        'labels', () {
      final encoder = OneHotEncoder();
      expect(encoder.encode(['group A', 'group B', 'group C', 'group D']),
          equals([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
          ]),
      );
    });

    test('should encode categorical data, unordered collection of unrepeatable'
        'labels', () {
      final encoder = OneHotEncoder();
      expect(encoder.encode(['group B', 'group D', 'group A', 'group C']),
        equals([
          [1, 0, 0, 0],
          [0, 1, 0, 0],
          [0, 0, 1, 0],
          [0, 0, 0, 1],
        ]),
      );
    });

    test('should encode categorical data, unordered collection of repeatable'
        'labels', () {
      final encoder = OneHotEncoder();
      expect(encoder.encode(['group B', 'group D', 'group B', 'group A',
        'group C', 'group C', 'group A']),
        equals([
          [1, 0, 0, 0],
          [0, 1, 0, 0],
          [1, 0, 0, 0],
          [0, 0, 1, 0],
          [0, 0, 0, 1],
          [0, 0, 0, 1],
          [0, 0, 1, 0],
        ]),
      );
    });

    test('should throw an error if an empty label list is passed', () {
      final encoder = OneHotEncoder();
      expect(() => encoder.encode([]), throwsException);
    });
  });
}
