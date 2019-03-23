import 'package:logging/logging.dart';
import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/cost_function/cost_function_factory.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_factory.dart';
import 'package:ml_algo/src/data_preprocessing/categorical_encoder/encoder_type.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/csv_codec_factory/csv_codec_factory.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/to_float_number_converter/to_float_number_converter.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/validator/params_validator.dart';
import 'package:ml_algo/src/data_preprocessing/intercept_preprocessor/intercept_preprocessor.dart';
import 'package:ml_algo/src/data_preprocessing/intercept_preprocessor/intercept_preprocessor_factory.dart';
import 'package:ml_algo/src/math/randomizer/randomizer.dart';
import 'package:ml_algo/src/math/randomizer/randomizer_factory.dart';
import 'package:ml_algo/src/optimizer/convergence_detector/convergence_detector.dart';
import 'package:ml_algo/src/optimizer/convergence_detector/convergence_detector_factory.dart';
import 'package:ml_algo/src/optimizer/gradient/learning_rate_generator/learning_rate_generator.dart';
import 'package:ml_algo/src/optimizer/gradient/learning_rate_generator/learning_rate_generator_factory.dart';
import 'package:ml_algo/src/optimizer/gradient/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_generator.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_generator_factory.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_algo/src/optimizer/optimizer.dart';
import 'package:ml_algo/src/optimizer/optimizer_factory.dart';
import 'package:ml_algo/src/optimizer/optimizer_type.dart';
import 'package:ml_algo/src/score_to_prob_mapper/score_to_prob_mapper.dart';
import 'package:ml_algo/src/score_to_prob_mapper/score_to_prob_mapper_factory.dart';
import 'package:ml_algo/src/score_to_prob_mapper/score_to_prob_mapper_type.dart';
import 'package:mockito/mockito.dart';

class EncoderMock extends Mock implements CategoricalDataEncoder {}

class OneHotEncoderMock extends Mock implements CategoricalDataEncoder {}

class OrdinalEncoderMock extends Mock implements CategoricalDataEncoder {}

class CategoricalDataEncoderFactoryMock extends Mock
    implements CategoricalDataEncoderFactory {}

class DataFrameParamsValidatorMock extends Mock implements
    DataFrameParamsValidator {}

class LoggerMock extends Mock implements Logger {}

class ToNumberConverterMock extends Mock implements
    ToFloatNumberConverter {}

class RandomizerFactoryMock extends Mock implements RandomizerFactory {}

class RandomizerMock extends Mock implements Randomizer {}

class CostFunctionFactoryMock extends Mock implements CostFunctionFactory {}

class CostFunctionMock extends Mock implements CostFunction {}

class LearningRateGeneratorFactoryMock extends Mock
    implements LearningRateGeneratorFactory {}

class LearningRateGeneratorMock extends Mock implements
    LearningRateGenerator {}

class InitialWeightsGeneratorFactoryMock extends Mock
    implements InitialWeightsGeneratorFactory {}

class InitialWeightsGeneratorMock extends Mock
    implements InitialWeightsGenerator {}

class ScoreToProbMapperMock extends Mock implements ScoreToProbMapper {}

class ScoreToProbMapperFactoryMock extends Mock
    implements ScoreToProbMapperFactory {}

class InterceptPreprocessorFactoryMock extends Mock
    implements InterceptPreprocessorFactory {}

class InterceptPreprocessorMock extends Mock implements InterceptPreprocessor {}

class OptimizerFactoryMock extends Mock implements OptimizerFactory {}

class OptimizerMock extends Mock implements Optimizer {}

class ConvergenceDetectorFactoryMock extends Mock
    implements ConvergenceDetectorFactory {}

class ConvergenceDetectorMock extends Mock implements ConvergenceDetector {}

class CategoricalDataEncoderMock extends Mock implements
    CategoricalDataEncoder {}

class CsvCodecFactoryMock extends Mock implements CsvCodecFactory {}

class MLDataValueConverterMockWithImpl extends Mock
    implements ToFloatNumberConverter {
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
    when(factory.fromType(type, any)).thenReturn(generator);
  });
  return factory;
}

ScoreToProbMapperFactoryMock createScoreToProbMapperFactoryMock(
  Type dtype, {
  Map<ScoreToProbMapperType, ScoreToProbMapper> mappers,
}) {
  final factory = ScoreToProbMapperFactoryMock();
  mappers.forEach((ScoreToProbMapperType type, ScoreToProbMapper fn) {
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
      scoreToProbMapperType: anyNamed('scoreToProbMapperType'),
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
  when(factory.oneHot()).thenReturn(oneHotEncoderMock);
  when(factory.ordinal()).thenReturn(ordinalEncoderMock);
  when(factory.fromType(CategoricalDataEncoderType.oneHot))
      .thenReturn(oneHotEncoderMock);
  when(factory.fromType(CategoricalDataEncoderType.ordinal))
      .thenReturn(ordinalEncoderMock);
  return factory;
}

DataFrameParamsValidator createDataFrameParamsValidatorMock(
    {bool validationShouldBeFailed}) {
  final validator = DataFrameParamsValidatorMock();
  if (validationShouldBeFailed != null) {
    when(validator.validate(
      labelIdx: anyNamed('labelIdx'),
      labelName: anyNamed('labelName'),
      rows: anyNamed('rows'),
      columns: anyNamed('columns'),
      headerExists: anyNamed('headerExists'),
      namesToEncoders: anyNamed('namesToEncoders'),
      indexToEncoder: anyNamed('indexToEncoder'),
    )).thenReturn(validationShouldBeFailed ? 'error' : '');
  }
  return validator;
}
