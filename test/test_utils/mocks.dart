import 'package:logging/logging.dart';
import 'package:ml_algo/learning_rate_type.dart';
import 'package:ml_algo/src/classifier/labels_processor/labels_processor.dart';
import 'package:ml_algo/src/classifier/labels_processor/labels_processor_factory.dart';
import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/cost_function/cost_function_factory.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/category_values_extractor.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_factory.dart';
import 'package:ml_algo/src/data_preprocessing/intercept_preprocessor/intercept_preprocessor.dart';
import 'package:ml_algo/src/data_preprocessing/intercept_preprocessor/intercept_preprocessor_factory.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/validator/ml_data_params_validator.dart';
import 'package:ml_algo/src/data_preprocessing/ml_data/value_converter/value_converter.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_algo/src/link_function/link_function_factory.dart';
import 'package:ml_algo/src/link_function/link_function_type.dart';
import 'package:ml_algo/src/math/randomizer/randomizer.dart';
import 'package:ml_algo/src/math/randomizer/randomizer_factory.dart';
import 'package:ml_algo/src/optimizer/gradient/learning_rate_generator/learning_rate_generator.dart';
import 'package:ml_algo/src/optimizer/gradient/learning_rate_generator/learning_rate_generator_factory.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_generator.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_generator_factory.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_algo/src/optimizer/optimizer.dart';
import 'package:ml_algo/src/optimizer/optimizer_factory.dart';
import 'package:ml_algo/src/optimizer/optimizer_type.dart';
import 'package:mockito/mockito.dart';

class EncoderMock extends Mock implements CategoricalDataEncoder {}

class OneHotEncoderMock extends Mock implements CategoricalDataEncoder {}

class OrdinalEncoderMock extends Mock implements CategoricalDataEncoder {}

class CategoricalDataEncoderFactoryMock extends Mock
    implements CategoricalDataEncoderFactory {}

class CategoryValuesExtractorMock extends Mock
    implements CategoryValuesExtractor<dynamic> {}

class MLDataParamsValidatorMock extends Mock implements MLDataParamsValidator {}

class LoggerMock extends Mock implements Logger {}

class MLDataValueConverterMock extends Mock implements MLDataValueConverter {}

class RandomizerFactoryMock extends Mock implements RandomizerFactory {}

class RandomizerMock extends Mock implements Randomizer {}

class CostFunctionFactoryMock extends Mock implements CostFunctionFactory {}

class CostFunctionMock extends Mock implements CostFunction {}

class LearningRateGeneratorFactoryMock extends Mock
    implements LearningRateGeneratorFactory {}

class LearningRateGeneratorMock extends Mock implements LearningRateGenerator {}

class InitialWeightsGeneratorFactoryMock extends Mock
    implements InitialWeightsGeneratorFactory {}

class InitialWeightsGeneratorMock extends Mock
    implements InitialWeightsGenerator {}

class LinkFunctionMock extends Mock implements LinkFunction {}

class LinkFunctionFactoryMock extends Mock implements LinkFunctionFactory {}

class LabelsProcessorFactoryMock extends Mock
    implements LabelsProcessorFactory {}

class LabelsProcessorMock extends Mock implements LabelsProcessor {}

class InterceptPreprocessorFactoryMock extends Mock
    implements InterceptPreprocessorFactory {}

class InterceptPreprocessorMock extends Mock implements InterceptPreprocessor {}

class OptimizerFactoryMock extends Mock implements OptimizerFactory {}

class OptimizerMock extends Mock implements Optimizer {}

class MLDataValueConverterMockWithImpl extends Mock
    implements MLDataValueConverter {
  @override
  double convert(Object value, [double fallbackValue]) => value as double;
}

LearningRateGeneratorFactoryMock createLearningRateGeneratorFactoryMock({
  Map<LearningRateType, LearningRateGenerator> generators,
}) {
  final factory = LearningRateGeneratorFactoryMock();
  generators.forEach((LearningRateType type, LearningRateGenerator generator) {
    when(factory.fromType(type)).thenReturn(generator);
  });
  return factory;
}

CostFunctionFactoryMock createCostFunctionFactoryMock({
  Map<CostFunctionType, CostFunction> costFunctions,
}) {
  final factory = CostFunctionFactoryMock();
  costFunctions.forEach((CostFunctionType type, CostFunction fn) {
    when(factory.fromType(type)).thenReturn(fn);
  });
  return factory;
}

RandomizerFactoryMock createRandomizerFactoryMock({
  Map<int, RandomizerMock> randomizers,
}) {
  final factory = RandomizerFactoryMock();
  randomizers.forEach((int seed, Randomizer randomizer) {
    when(factory.create(seed)).thenReturn(randomizer);
  });
  return factory;
}

InitialWeightsGeneratorFactoryMock createInitialWeightsGeneratorFactoryMock({
  Map<InitialWeightsType, InitialWeightsGenerator> generators,
}) {
  final factory = InitialWeightsGeneratorFactoryMock();
  generators
      .forEach((InitialWeightsType type, InitialWeightsGenerator generator) {
    when(factory.fromType(type)).thenReturn(generator);
  });
  return factory;
}

LinkFunctionFactoryMock createLinkFunctionFactoryMock(
  Type dtype, {
  Map<LinkFunctionType, LinkFunction> linkFunctions,
}) {
  final factory = LinkFunctionFactoryMock();
  linkFunctions.forEach((LinkFunctionType type, LinkFunction fn) {
    when(factory.fromType(type, dtype)).thenReturn(fn);
  });
  return factory;
}

InterceptPreprocessorFactoryMock createInterceptPreprocessorFactoryMock({
  InterceptPreprocessor preprocessor,
}) {
  final factory = InterceptPreprocessorFactoryMock();
  when(factory.create(any, scale: anyNamed('scale'))).thenReturn(preprocessor);
  return factory;
}

LabelsProcessorFactoryMock createLabelsProcessorFactoryMock({
  Map<Type, LabelsProcessor> processors,
}) {
  final factory = LabelsProcessorFactoryMock();
  processors.forEach((Type type, LabelsProcessor processor) {
    when(factory.create(type)).thenReturn(processor);
  });
  return factory;
}

OptimizerFactoryMock createOptimizerFactoryMock({
  Map<OptimizerType, Optimizer> optimizers,
}) {
  final factory = OptimizerFactoryMock();

  optimizers.forEach((OptimizerType type, Optimizer optimizer) {
    when(factory.fromType(
      type,
      dtype: anyNamed('dtype'),
      randomizerFactory: anyNamed('randomizerFactory'),
      costFunctionFactory: anyNamed('costFunctionFactory'),
      learningRateGeneratorFactory: anyNamed('learningRateGeneratorFactory'),
      initialWeightsGeneratorFactory:
          anyNamed('initialWeightsGeneratorFactory'),
      costFunctionType: anyNamed('costFunctionType'),
      learningRateType: anyNamed('learningRateType'),
      initialWeightsType: anyNamed('initialWeightsType'),
      linkFunctionType: anyNamed('linkFunctionType'),
      initialLearningRate: anyNamed('initialLearningRate'),
      minCoefficientsUpdate: anyNamed('minCoefficientsUpdate'),
      iterationLimit: anyNamed('iterationLimit'),
      lambda: anyNamed('lambda'),
      batchSize: anyNamed('batchSize'),
      randomSeed: anyNamed('randomSeed'),
    )).thenReturn(optimizer);
  });

  return factory;
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

MLDataParamsValidator createMLDataParamsValidatorMock(
    {bool validationShouldBeFailed}) {
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
