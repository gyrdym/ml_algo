import 'package:ml_algo/src/data_preprocessing/categorical_encoder/category_values_extractor_impl.dart';
import 'package:test/test.dart';

void main() {
  group('CategoryValuesExtractorImpl', () {
    test('should find unique values and return them as List', () {
      final extractor = const CategoryValuesExtractorImpl<String>();
      final values = <String>[
        '1',
        '2',
        'hello',
        '1',
        '1',
        'hi',
        '2',
        '',
        '',
        'trololo'
      ];
      final actual = extractor.extractCategoryValues(values);
      final expected = <String>['1', '2', 'hello', 'hi', '', 'trololo'];
      expect(actual, equals(expected));
    });
  });
}
