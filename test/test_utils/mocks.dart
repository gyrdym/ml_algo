import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_factory.dart';
import 'package:mockito/mockito.dart';

class OneHotEncoderMock extends Mock implements CategoricalDataEncoder {}
class OrdinalEncoderMock extends Mock implements CategoricalDataEncoder {}
class CategoricalDataEncoderFactoryMock extends Mock implements CategoricalDataEncoderFactory {}

CategoricalDataEncoderFactory createCategoricalDataEncoderFactoryMock() {
  final factory = CategoricalDataEncoderFactoryMock();
  when(factory.oneHot(any, any)).thenReturn(OneHotEncoderMock());
  when(factory.ordinal(any, any)).thenReturn(OrdinalEncoderMock());
  return factory;
}