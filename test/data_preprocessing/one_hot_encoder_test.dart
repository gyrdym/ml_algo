import 'package:ml_algo/encode_unknown_value_strategy.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/one_hot_encoder.dart';
import 'package:test/test.dart';

OneHotEncoder createEncoder(EncodeUnknownValueStrategy strategy) =>
  OneHotEncoder({
    'gender': ['male', 'female'],
    'race/ethnicity': ['group A', 'group B', 'group C', 'group D'],
    'married': [true, false],
  }, strategy);

void main() {
  group('OneHotEncoder', () {
    test('should encode categorical data', () {
      final encoder = createEncoder(EncodeUnknownValueStrategy.returnZeroes);

      expect(encoder.encode('gender', 'male'), equals([1, 0]));
      expect(encoder.encode('gender', 'female'), equals([0, 1]));

      expect(encoder.encode('race/ethnicity', 'group A'), equals([1, 0, 0, 0]));
      expect(encoder.encode('race/ethnicity', 'group B'), equals([0, 1, 0, 0]));
      expect(encoder.encode('race/ethnicity', 'group C'), equals([0, 0, 1, 0]));
      expect(encoder.encode('race/ethnicity', 'group D'), equals([0, 0, 0, 1]));

      expect(encoder.encode('married', true), equals([1, 0]));
      expect(encoder.encode('married', false), equals([0, 1]));
    });

    test('should throw an error if unknown value is passed and unknown value encoding strategy is `throwError`', () {
      final encoder = createEncoder(EncodeUnknownValueStrategy.throwError);
      expect(() => encoder.encode('lunch', 'standart'), throwsUnsupportedError);
      expect(() => encoder.encode('gender', 'some unsupported value'), throwsUnsupportedError);
      expect(() => encoder.encode('race/ethnicity', 'another unsupported value'), throwsUnsupportedError);
    });

    test('should return all zeroes if unknown value is passed and unknown value encoding strategy is `returnZeroes`', () {
      final encoder = createEncoder(EncodeUnknownValueStrategy.returnZeroes);
      expect(encoder.encode('gender', 'some unsupported value'), equals([0, 0]));
      expect(encoder.encode('race/ethnicity', 'another unsupported value'), equals([0, 0, 0, 0]));
    });

    test('should throw an error if unknown category is passed', () {
      final encoder = createEncoder(EncodeUnknownValueStrategy.returnZeroes);
      expect(() => encoder.encode('lunch', 'standart'), throwsUnsupportedError);
    });
  });
}
