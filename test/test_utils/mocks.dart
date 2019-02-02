import 'package:logging/logging.dart';
import 'package:ml_algo/src/classifier/labels_processor/labels_processor_factory.dart';
import 'package:ml_algo/src/cost_function/cost_function_factory.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/category_values_extractor.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_factory.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/validator/ml_data_params_validator.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/value_converter/value_converter.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_algo/src/math/randomizer/randomizer_factory.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_generator.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_generator_factory.dart';
import 'package:ml_algo/src/optimizer/learning_rate_generator/learning_rate_generator_factory.dart';
import 'package:mockito/mockito.dart';

class EncoderMock extends Mock implements CategoricalDataEncoder {}
class OneHotEncoderMock extends Mock implements CategoricalDataEncoder {}
class OrdinalEncoderMock extends Mock implements CategoricalDataEncoder {}
class CategoricalDataEncoderFactoryMock extends Mock implements CategoricalDataEncoderFactory {}
class CategoryValuesExtractorMock extends Mock implements CategoryValuesExtractor<dynamic> {}
class MLDataParamsValidatorMock extends Mock implements MLDataParamsValidator {}
class LoggerMock extends Mock implements Logger {}
class MLDataValueConverterMock extends Mock implements MLDataValueConverter {}
class RandomizerFactoryMock extends Mock implements RandomizerFactory {}
class CostFunctionFactoryMock extends Mock implements CostFunctionFactory {}
class LearningRateGeneratorFactoryMock extends Mock implements LearningRateGeneratorFactory {}
class InitialWeightsGeneratorFactoryMock extends Mock implements InitialWeightsGeneratorFactory {}
class InitialWeightsGeneratorMock extends Mock implements InitialWeightsGenerator {}
class LinkFunctionMock extends Mock implements LinkFunction {}
class LabelsProcessorFactoryMock extends Mock implements LabelsProcessorFactory {}

class MLDataValueConverterMockWithImpl extends Mock implements MLDataValueConverter {
  @override
  double convert(Object value, [double fallbackValue]) => value as double;
}

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
