import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/features_extractor/features_extractor_impl.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/value_converter/value_converter.dart';
import 'package:mockito/mockito.dart';
import 'package:test_api/test_api.dart';

class MLDataValueConverterMock extends Mock implements MLDataValueConverter {
  @override
  double convert(Object value, [double fallbackValue]) => value as double;
}

void main() {
  final data = [
    [10.0, 20.0, 30.0, 40.0, 50.0],
    [100.0, 200.0, 300.0, 400.0, 500.0],
    [110.0, 120.0, 130.0, 140.0, 150.0],
    [210.0, 220.0, 230.0, 240.0, 250.0],
  ];

  group('MLDataFeaturesExtractorImpl', () {
    test('should extract features according to passed colum read mask', () {
      final rowMask = <bool>[true, true, true, true];
      final columnsMask = <bool>[true, false, true, false, true];
      final encoders = <int, CategoricalDataEncoder>{};
      final labelIdx = 4;
      final valueConverter = MLDataValueConverterMock();

      final extractor = MLDataFeaturesExtractorImpl(rowMask, columnsMask, encoders, labelIdx, valueConverter);
      final features = extractor.extract(data, hasCategoricalData: false);

      expect(features, equals([
        [10.0, 30.0],
        [100.0, 300.0],
        [110.0, 130.0],
        [210.0, 230.0],
      ]));
    });

    test('should extract features according to passed row read mask', () {
      final rowMask = <bool>[true, false, false, true];
      final columnsMask = <bool>[true, true, true, true, true];
      final encoders = <int, CategoricalDataEncoder>{};
      final labelIdx = 4;
      final valueConverter = MLDataValueConverterMock();

      final extractor = MLDataFeaturesExtractorImpl(rowMask, columnsMask, encoders, labelIdx, valueConverter);
      final features = extractor.extract(data, hasCategoricalData: false);

      expect(features, equals([
        [10.0, 20.0, 30.0, 40.0],
        [210.0, 220.0, 230.0, 240.0],
      ]));
    });
  });
}