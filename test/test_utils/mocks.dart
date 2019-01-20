import 'package:logging/logging.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/category_values_extractor.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_factory.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/validator/ml_data_params_validator.dart';
import 'package:mockito/mockito.dart';

class EncoderMock extends Mock implements CategoricalDataEncoder {}
class OneHotEncoderMock extends Mock implements CategoricalDataEncoder {}
class OrdinalEncoderMock extends Mock implements CategoricalDataEncoder {}
class CategoricalDataEncoderFactoryMock extends Mock implements CategoricalDataEncoderFactory {}
class CategoryValuesExtractorMock extends Mock implements CategoryValuesExtractor<dynamic> {}
class MLDataParamsValidatorMock extends Mock implements MLDataParamsValidator {}
class LoggerMock extends Mock implements Logger {}

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

MLDataParamsValidator createMLDataParamsValidatorMock({bool validationShouldBeFailed}) {
  final validator = MLDataParamsValidatorMock();
  if (validationShouldBeFailed != null) {
    when(validator.validate(
      labelIdx: anyNamed('labelIdx'),
      rows: anyNamed('rows'),
      columns: anyNamed('columns'),
      headerExists: anyNamed('headerExists'),
      predefinedCategories: anyNamed('predefinedCategories'),
      namesToEncoders: anyNamed('namesToEncoders'),
      indexToEncoder: anyNamed('indexToEncoder'),
    )).thenReturn(validationShouldBeFailed ? 'error' : '');
  }
  return validator;
}
