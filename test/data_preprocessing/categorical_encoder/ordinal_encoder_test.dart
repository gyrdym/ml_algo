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

    test('should encode categorical data', () {
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

    test('should throw an error if unknown value is passed and unknown value '
        'encoding strategy is `throwError`', () {
      when(valuesExtractor.extractCategoryValues(any))
          .thenReturn(['1', '2', '3', '4']);
      final encoder = createEncoder(
          strategy: EncodeUnknownValueStrategy.throwError,
          extractor: valuesExtractor,
          values: [],
          type: CategoricalDataEncoderType.ordinal);
      expect(() => encoder.encodeSingle('234'), throwsUnsupportedError);
    });

    test('should return all zeroes if unknown value is passed and unknown '
        'value encoding strategy is `returnZeroes`', () {
      when(valuesExtractor.extractCategoryValues(any))
          .thenReturn(['10', '20']);
      final encoder = createEncoder(
          strategy: EncodeUnknownValueStrategy.returnZeroes,
          extractor: valuesExtractor,
          values: [],
          type: CategoricalDataEncoderType.ordinal);
      expect(encoder.encodeSingle('21'), equals([0]));
    });
  });
}
