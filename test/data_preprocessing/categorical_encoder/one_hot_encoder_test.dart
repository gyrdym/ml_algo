import 'package:ml_algo/ml_algo.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../test_utils/mocks.dart';
import '../test_helpers/create_encoder.dart';

void main() {
  group('OneHotEncoder', () {
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
          values: []);
      expect(encoder.encodeSingle('group A'), equals([1, 0, 0, 0]));
      expect(encoder.encodeSingle('group B'), equals([0, 1, 0, 0]));
      expect(encoder.encodeSingle('group C'), equals([0, 0, 1, 0]));
      expect(encoder.encodeSingle('group D'), equals([0, 0, 0, 1]));
    });

    test('should throw an error if unknown value is passed and unknown value '
        'encoding strategy is `throwError`', () {
      when(valuesExtractor.extractCategoryValues(any))
          .thenReturn(<String>['1', '2', '3', '4']);
      final encoder = createEncoder(
          strategy: EncodeUnknownValueStrategy.throwError,
          extractor: valuesExtractor,
          values: []);
      expect(() => encoder.encodeSingle('234'), throwsUnsupportedError);
    });

    test('should return all zeroes if unknown value is passed and unknown '
        'value encoding strategy is `returnZeroes`', () {
      when(valuesExtractor.extractCategoryValues(any))
          .thenReturn(<String>['10', '20']);
      final encoder = createEncoder(
          strategy: EncodeUnknownValueStrategy.returnZeroes,
          extractor: valuesExtractor,
          values: []);
      expect(encoder.encodeSingle('21'), equals([0, 0]));
    });
  });
}
