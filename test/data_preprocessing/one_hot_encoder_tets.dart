import 'package:ml_algo/encode_unknown_value_strategy.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/one_hot_encoder.dart';
import 'package:test/test.dart';

void main() {
  final encoder = OneHotEncoder({
    'gender': ['male', 'female'],
    'race/ethnicity': ['group A', 'group B', 'group C', 'group D'],
    'married': [true, false],
  }, EncodeUnknownValueStrategy.ignore);

  group('OneHotEncoder', () {
    test('should encode categorical data', () {
      expect(encoder.encode('gender', 'male'), equals([1, 0]));
      expect(encoder.encode('gender', 'female'), equals([0, 1]));

      expect(encoder.encode('race/ethnicity', 'group A'), equals([1, 0, 0, 0]));
      expect(encoder.encode('race/ethnicity', 'group B'), equals([0, 1, 0, 0]));
      expect(encoder.encode('race/ethnicity', 'group C'), equals([0, 0, 1, 0]));
      expect(encoder.encode('race/ethnicity', 'group D'), equals([0, 0, 0, 1]));

      expect(encoder.encode('married', true), equals([1, 0]));
      expect(encoder.encode('married', false), equals([0, 1]));
    });

    test('should throw an appropriate error if invalid value is passed', () {
      expect(() => encoder.encode('lunch', 'standart'), throwsUnsupportedError);
      expect(() => encoder.encode('gender', 'some unsupported value'), throwsUnsupportedError);
      expect(() => encoder.encode('race/ethnicity', 'another unsupported value'), throwsUnsupportedError);
    });
  });
}
