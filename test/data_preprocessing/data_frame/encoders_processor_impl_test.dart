import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_type.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/encoders_processor/encoders_processor_impl.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../test_utils/mocks.dart';

void main() {
  final header = ['country', 'gender', 'age', 'martial_status', 'salary'];
  final records = [
    ['France', 'male', 35, 'married', 5000],
    ['Russia', 'male', 27, 'single', 2500],
    ['Spain', 'female', 21, 'single', 3000],
    ['Greece', 'female', 25, 'divorced', 2700],
  ];
  final predefinedCategories = <String, List<String>>{
    'gender': ['male', 'female'],
    'martial_status': ['married', 'single', 'divorced'],
    'age': ['21', '27', '25', '35'],
    'country': ['France', 'Russia', 'Spain', 'Greece'],
  };

  group('MLDataEncodersProcessorImpl', () {
    test(
        'should create fallback encoders for all columns if there are no specific encoders provided, predefined'
        ' categories are provided and the columns header is not empty', () {
      final encoderFactory = createCategoricalDataEncoderFactoryMock();
      final fallbackEncoderType = CategoricalDataEncoderType.oneHot;
      final fallbackEncoderMock = OneHotEncoderMock();
      final encoderProcessor = DataFrameEncodersProcessorImpl(
          records, header, encoderFactory, fallbackEncoderType);

      when(encoderFactory.fromType(any)).thenReturn(fallbackEncoderMock);

      encoderProcessor.createEncoders({}, {}, predefinedCategories);
      verify(encoderFactory.fromType(fallbackEncoderType)).called(4);
      verify(fallbackEncoderMock
          .setCategoryValues(argThat(equals(['male', 'female'])))).called(1);
      verify(fallbackEncoderMock.setCategoryValues(
          argThat(equals(['married', 'single', 'divorced'])))).called(1);
      verify(fallbackEncoderMock
          .setCategoryValues(argThat(equals(['21', '27', '25', '35'])))).called(1);
      verify(fallbackEncoderMock.setCategoryValues(
          argThat(equals(['France', 'Russia', 'Spain', 'Greece'])))).called(1);
    });

    test(
        'should create encoders from `name-to-encoder` map if the map and predefined categories are provided (the '
        'map has more priority) and the header is not empty', () {
      final encoderFactory = createCategoricalDataEncoderFactoryMock();
      final fallbackEncoderType = CategoricalDataEncoderType.oneHot;
      final encoderProcessor = DataFrameEncodersProcessorImpl(
          records, header, encoderFactory, fallbackEncoderType);
      final oneHotEncoderMock = OneHotEncoderMock();
      final ordinalEncoderMock = OrdinalEncoderMock();

      final nameToEncoder = <String, CategoricalDataEncoderType>{
        'gender': CategoricalDataEncoderType.oneHot,
        'martial_status': CategoricalDataEncoderType.oneHot,
        'age': CategoricalDataEncoderType.ordinal,
        'country': CategoricalDataEncoderType.ordinal,
      };

      when(encoderFactory.fromType(CategoricalDataEncoderType.ordinal))
          .thenReturn(ordinalEncoderMock);
      when(encoderFactory.fromType(CategoricalDataEncoderType.oneHot))
          .thenReturn(oneHotEncoderMock);

      encoderProcessor.createEncoders({}, nameToEncoder, predefinedCategories);

      verify(oneHotEncoderMock.setCategoryValues(
          argThat(equals(['male', 'male', 'female', 'female']))));
      verify(oneHotEncoderMock.setCategoryValues(
          argThat(equals(['married', 'single', 'single', 'divorced']))));
      verify(ordinalEncoderMock
          .setCategoryValues(argThat(equals(['35', '27', '21', '25']))));
      verify(ordinalEncoderMock.setCategoryValues(
          argThat(equals(['France', 'Russia', 'Spain', 'Greece']))));
    });

    test(
        'should create encoders from `index-to-encoder` map if the map, a `name-to-encoder` map and predefined '
        'categories are provided (`index-to-encoder` map has high priority)',
        () {
      final encoderFactory = createCategoricalDataEncoderFactoryMock();
      final fallbackEncoderType = CategoricalDataEncoderType.oneHot;
      final encoderProcessor = DataFrameEncodersProcessorImpl(
          records, header, encoderFactory, fallbackEncoderType);
      final oneHotEncoderMock = OneHotEncoderMock();
      final ordinalEncoderMock = OrdinalEncoderMock();

      final indexToEncoder = <int, CategoricalDataEncoderType>{
        0: CategoricalDataEncoderType.ordinal,
        1: CategoricalDataEncoderType.ordinal,
        2: CategoricalDataEncoderType.oneHot,
        3: CategoricalDataEncoderType.oneHot,
      };

      final nameToEncoder = <String, CategoricalDataEncoderType>{
        'country': CategoricalDataEncoderType.oneHot,
        'gender': CategoricalDataEncoderType.oneHot,
        'age': CategoricalDataEncoderType.ordinal,
        'martial_status': CategoricalDataEncoderType.ordinal,
      };

      when(encoderFactory.fromType(CategoricalDataEncoderType.ordinal))
          .thenReturn(ordinalEncoderMock);
      when(encoderFactory.fromType(CategoricalDataEncoderType.oneHot))
          .thenReturn(oneHotEncoderMock);

      encoderProcessor.createEncoders(
          indexToEncoder, nameToEncoder, predefinedCategories);

      verify(ordinalEncoderMock.setCategoryValues(
          argThat(equals(['France', 'Russia', 'Spain', 'Greece']))));
      verify(ordinalEncoderMock.setCategoryValues(
          argThat(equals(['male', 'male', 'female', 'female']))));
      verify(oneHotEncoderMock.setCategoryValues(
          argThat(equals(['married', 'single', 'single', 'divorced']))));
      verify(oneHotEncoderMock
          .setCategoryValues(argThat(equals(['35', '27', '21', '25']))));
    });

    test(
        'should throw a warning if predefined categories are provided, but columns header is not',
        () {
      final encoderFactory = createCategoricalDataEncoderFactoryMock();
      final fallbackEncoderType = CategoricalDataEncoderType.oneHot;
      final encoderProcessor = DataFrameEncodersProcessorImpl(
          records, [], encoderFactory, fallbackEncoderType);
      final oneHotEncoderMock = OneHotEncoderMock();

      when(encoderFactory.fromType(fallbackEncoderType))
          .thenReturn(oneHotEncoderMock);

      encoderProcessor.createEncoders({}, {}, predefinedCategories);

      verifyNever(oneHotEncoderMock
          .setCategoryValues(argThat(equals(['male', 'female']))));
      verifyNever(oneHotEncoderMock.setCategoryValues(
          argThat(equals(['married', 'single', 'divorced']))));
      verifyNever(oneHotEncoderMock
          .setCategoryValues(argThat(equals(['21', '27', '25', '35']))));
      verifyNever(oneHotEncoderMock.setCategoryValues(
          argThat(equals(['France', 'Russia', 'Spain', 'Greece']))));
    });

    test(
        'should throw a warning if column names to encoders map is provided, but columns header is not',
        () {
      final encoderFactory = createCategoricalDataEncoderFactoryMock();
      final fallbackEncoderType = CategoricalDataEncoderType.oneHot;
      final encoderProcessor = DataFrameEncodersProcessorImpl(
          records, [], encoderFactory, fallbackEncoderType);
      final oneHotEncoderMock = OneHotEncoderMock();
      final ordinalEncoderMock = OrdinalEncoderMock();

      final nameToEncoder = <String, CategoricalDataEncoderType>{
        'country': CategoricalDataEncoderType.oneHot,
        'gender': CategoricalDataEncoderType.oneHot,
        'age': CategoricalDataEncoderType.ordinal,
        'martial_status': CategoricalDataEncoderType.ordinal,
      };

      when(encoderFactory.fromType(fallbackEncoderType))
          .thenReturn(oneHotEncoderMock);

      encoderProcessor.createEncoders({}, nameToEncoder, predefinedCategories);

      verifyNever(ordinalEncoderMock.setCategoryValues(
          argThat(equals(['France', 'Russia', 'Spain', 'Greece']))));
      verifyNever(ordinalEncoderMock.setCategoryValues(
          argThat(equals(['male', 'male', 'female', 'female']))));
      verifyNever(oneHotEncoderMock.setCategoryValues(
          argThat(equals(['married', 'single', 'single', 'divorced']))));
      verifyNever(oneHotEncoderMock
          .setCategoryValues(argThat(equals(['35', '27', '21', '25']))));
    });

    test(
        'should return an empty encoders map if no categoriies data is provided',
        () {
      final encoderFactory = createCategoricalDataEncoderFactoryMock();
      final fallbackEncoderType = CategoricalDataEncoderType.oneHot;
      final encoderProcessor = DataFrameEncodersProcessorImpl(
          records, [], encoderFactory, fallbackEncoderType);
      final encoders = encoderProcessor.createEncoders({}, {}, {});

      expect(encoders, equals(<int, CategoricalDataEncoder>{}));
    });
  });
}
