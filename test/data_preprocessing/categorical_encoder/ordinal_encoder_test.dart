import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_type.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../test_utils/mocks.dart';
import '../test_helpers/create_encoder.dart';

void main() {
  group('Ordinal encoder', () {
    final valuesExtractor = CategoryValuesExtractorMock();

    tearDown(() {
      clearInteractions(valuesExtractor);
    });

    test('should encode numeric categorical test_data', () {
      when(valuesExtractor.extractCategoryValues(any))
          .thenReturn(<int>[20, 10]);
      final encoder = createEncoder(
          strategy: EncodeUnknownValueStrategy.returnZeroes,
          extractor: valuesExtractor,
          values: [],
          type: CategoricalDataEncoderType.ordinal);
      expect(encoder.encodeSingle(20), equals([1]));
      expect(encoder.encodeSingle(10), equals([2]));
    });

    test('should encode string categorical test_data', () {
      when(valuesExtractor.extractCategoryValues(any))
          .thenReturn(<String>['group A', 'group B', 'group C', 'group D']);
      final encoder = createEncoder(
          strategy: EncodeUnknownValueStrategy.returnZeroes,
          extractor: valuesExtractor,
          values: [],
          type: CategoricalDataEncoderType.ordinal);
      expect(encoder.encodeSingle('group A'), equals([1]));
      expect(encoder.encodeSingle('group B'), equals([2]));
      expect(encoder.encodeSingle('group C'), equals([3]));
      expect(encoder.encodeSingle('group D'), equals([4]));
    });

    test('should encode boolean categorical test_data', () {
      when(valuesExtractor.extractCategoryValues(any))
          .thenReturn(<bool>[true, false]);
      final encoder = createEncoder(
          strategy: EncodeUnknownValueStrategy.returnZeroes,
          extractor: valuesExtractor,
          values: [],
          type: CategoricalDataEncoderType.ordinal);
      expect(encoder.encodeSingle(true), equals([1]));
      expect(encoder.encodeSingle(false), equals([2]));
    });

    test(
        'should throw an error if unknown value is passed and unknown value encoding strategy is `throwError`',
        () {
      when(valuesExtractor.extractCategoryValues(any))
          .thenReturn(<int>[1, 2, 3, 4]);
      final encoder = createEncoder(
          strategy: EncodeUnknownValueStrategy.throwError,
          extractor: valuesExtractor,
          values: [],
          type: CategoricalDataEncoderType.ordinal);
      expect(() => encoder.encodeSingle(234), throwsUnsupportedError);
    });

    test(
        'should return all zeroes if unknown value is passed and unknown value encoding strategy is `returnZeroes`',
        () {
      when(valuesExtractor.extractCategoryValues(any))
          .thenReturn(<int>[10, 20]);
      final encoder = createEncoder(
          strategy: EncodeUnknownValueStrategy.returnZeroes,
          extractor: valuesExtractor,
          values: [],
          type: CategoricalDataEncoderType.ordinal);
      expect(encoder.encodeSingle(21), equals([0]));
    });
  });
}
