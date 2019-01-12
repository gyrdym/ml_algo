import 'package:ml_algo/src/data_preprocessing/categorical_encoder/category_values_extractor.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_factory.dart';
import 'package:mockito/mockito.dart';

class OneHotEncoderMock extends Mock implements CategoricalDataEncoder {}
class OrdinalEncoderMock extends Mock implements CategoricalDataEncoder {}
class CategoricalDataEncoderFactoryMock extends Mock implements CategoricalDataEncoderFactory {}
class CategoryValuesExtractorMock extends Mock implements CategoryValuesExtractor<dynamic> {}

CategoricalDataEncoderFactory createCategoricalDataEncoderFactoryMock({
  CategoricalDataEncoder oneHotEncoderMock,
  CategoricalDataEncoder ordinalEncoderMock,
}) {
  oneHotEncoderMock ??= OneHotEncoderMock();
  ordinalEncoderMock ??= OrdinalEncoderMock();

  final factory = CategoricalDataEncoderFactoryMock();
  when(factory.oneHot(any)).thenReturn(oneHotEncoderMock);
  when(factory.ordinal(any)).thenReturn(ordinalEncoderMock);
  return factory;
}