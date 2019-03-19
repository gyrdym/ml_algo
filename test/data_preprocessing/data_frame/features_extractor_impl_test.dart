import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/features_extractor/features_extractor_impl.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../test_utils/mocks.dart' as mocks;

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
      final valueConverter = mocks.MLDataValueConverterMockWithImpl();

      final extractor = DataFrameFeaturesExtractorImpl(data, rowMask,
          columnsMask, encoders, labelIdx, valueConverter);
      final features = extractor.extract();

      expect(
          features,
          equals([
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
      final valueConverter = mocks.MLDataValueConverterMockWithImpl();

      final extractor = DataFrameFeaturesExtractorImpl(data, rowMask,
          columnsMask, encoders, labelIdx, valueConverter);
      final features = extractor.extract();

      expect(
          features,
          equals([
            [10.0, 20.0, 30.0, 40.0],
            [210.0, 220.0, 230.0, 240.0],
          ]));
    });

    test('should consider index of a label column while extracting fratures',
        () {
      final rowMask = <bool>[true, true, true, true];
      final columnsMask = <bool>[true, true, true, true, true];
      final encoders = <int, CategoricalDataEncoder>{};
      final labelIdx = 1;
      final valueConverter = mocks.MLDataValueConverterMockWithImpl();

      final extractor = DataFrameFeaturesExtractorImpl(data, rowMask,
          columnsMask, encoders, labelIdx, valueConverter);
      final features = extractor.extract();

      expect(
          features,
          equals([
            [10.0, 30.0, 40.0, 50.0],
            [100.0, 300.0, 400.0, 500.0],
            [110.0, 130.0, 140.0, 150.0],
            [210.0, 230.0, 240.0, 250.0],
          ]));
    });

    test('should encode categorical features', () {
      final encoderMock = mocks.OneHotEncoderMock();
      when(encoderMock.encodeSingle(any)).thenReturn([1000.0, 2000.0]);

      final rowMask = <bool>[true, true, true, true];
      final columnsMask = <bool>[true, true, true, true, true];
      final encoders = <int, CategoricalDataEncoder>{
        2: encoderMock,
      };
      final labelIdx = 4;
      final valueConverter = mocks.MLDataValueConverterMockWithImpl();

      final extractor = DataFrameFeaturesExtractorImpl(data, rowMask,
          columnsMask, encoders, labelIdx, valueConverter);
      final features = extractor.extract();

      expect(
          features,
          equals([
            [10.0, 20.0, 1000.0, 2000.0, 40.0],
            [100.0, 200.0, 1000.0, 2000.0, 400.0],
            [110.0, 120.0, 1000.0, 2000.0, 140.0],
            [210.0, 220.0, 1000.0, 2000.0, 240.0],
          ]));
    });

    test(
        'should not throw an error if length of columns mask is less than number of elements in a feature row',
        () {
      final rowMask = <bool>[true, true, true, true];
      final columnsMask = <bool>[true, true, true];
      final encoders = <int, CategoricalDataEncoder>{};
      final labelIdx = 4;
      final valueConverter = mocks.MLDataValueConverterMockWithImpl();
      final extractor = DataFrameFeaturesExtractorImpl(data, rowMask,
          columnsMask, encoders, labelIdx, valueConverter);
      final actual = extractor.extract();

      expect(
          actual,
          equals([
            [10.0, 20.0, 30.0],
            [100.0, 200.0, 300.0],
            [110.0, 120.0, 130.0],
            [210.0, 220.0, 230.0],
          ]));
    });

    test(
        'should throw an error if length of columns mask is greater than number of elements in a feature row',
        () {
      final rowMask = <bool>[true, true, true, true];
      final columnsMask = <bool>[true, true, true, true, true, true];
      final encoders = <int, CategoricalDataEncoder>{};
      final labelIdx = 4;
      final valueConverter = mocks.MLDataValueConverterMockWithImpl();

      expect(
          () => DataFrameFeaturesExtractorImpl(data, rowMask, columnsMask,
              encoders, labelIdx, valueConverter),
          throwsException);
    });

    test('should not throw an error if length of rows mask is less than number '
        'of rows in dataset', () {
      final rowMask = <bool>[true, true, true];
      final columnsMask = <bool>[true, true, true, true, true];
      final encoders = <int, CategoricalDataEncoder>{};
      final labelIdx = 4;
      final valueConverter = mocks.MLDataValueConverterMockWithImpl();
      final extractor = DataFrameFeaturesExtractorImpl(data, rowMask,
          columnsMask, encoders, labelIdx, valueConverter);
      final actual = extractor.extract();

      expect(
          actual,
          equals([
            [10.0, 20.0, 30.0, 40.0],
            [100.0, 200.0, 300.0, 400.0],
            [110.0, 120.0, 130.0, 140.0],
          ]));
    });

    test('should throw an error if length of rows mask is greater than number '
        'of rows in dataset', () {
      final rowMask = <bool>[true, true, true, true, true, true];
      final columnsMask = <bool>[true, true, true, true, true];
      final encoders = <int, CategoricalDataEncoder>{};
      final labelIdx = 4;
      final valueConverter = mocks.MLDataValueConverterMockWithImpl();

      expect(
          () => DataFrameFeaturesExtractorImpl(data, rowMask, columnsMask,
              encoders, labelIdx, valueConverter),
          throwsException);
    });
  });
}
