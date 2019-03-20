import 'package:ml_algo/src/data_preprocessing/data_frame/variables_extractor/variables_extractor_factory_impl.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/variables_extractor/variables_extractor_impl.dart';
import 'package:test/test.dart';

import '../../../test_utils/mocks.dart';

void main() {
  group('VariablesExtractorFactoryImpl', () {
    test('should create a proper variable extractor instance', () {
      final factory = const VariablesExtractorFactoryImpl();
      final extractor = factory.create([[1, 2, 3]], [], [], {}, 0,
          ToNumberConverterMock());
      expect(extractor.runtimeType, VariablesExtractorImpl);
    });
  });
}
