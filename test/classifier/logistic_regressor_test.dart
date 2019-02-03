import 'dart:typed_data';

import 'package:ml_algo/learning_rate_type.dart';
import 'package:ml_algo/src/classifier/logistic_regressor.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/link_function/link_function_type.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_type.dart';
import 'package:mockito/mockito.dart';
import 'package:test_api/test_api.dart';

import '../test_utils/mocks.dart';

LogisticRegressor createRegressor() {
  final classifier = LogisticRegressor();

  return classifier;
}

void main() {
  group('LogisticRegressor', () {
    test('should initialize properly', () {
      final labelsProcessorMock = LabelsProcessorMock();
      final labelsProcessorFactoryMock = createLabelsProcessorFactoryMock(processors: {Float32x4: labelsProcessorMock});

      final interceptPreprocessorMock = InterceptPreprocessorMock();
      final interceptPreprocessorFactoryMock = createInterceptPreprocessorFactoryMock(
          preprocessor: interceptPreprocessorMock);

      final probabilityCalculatorMock = LabelsProbabilityCalculatorMock();
      final probabilityCalculatorFactoryMock = createLabelsProbabilityCalculatorFactoryMock(
        linkType: LinkFunctionType.logit,
        dtype: Float32x4,
        calculator: probabilityCalculatorMock,
      );

      final optimizerMock = OptimizerMock();
      final optimizerFactoryMock = createOptimizerFactoryMock(gradient: optimizerMock);

      LogisticRegressor(
        learningRateType: LearningRateType.constant,
        initialWeightsType: InitialWeightsType.zeroes,
        iterationLimit: 100,
        learningRate: 0.01,
        minWeightsUpdate: 0.001,
        lambda: 0.1,
        labelsProcessorFactory: labelsProcessorFactoryMock,
        interceptPreprocessorFactory: interceptPreprocessorFactoryMock,
        linkFunctionType: LinkFunctionType.logit,
        probabilityCalculatorFactory: probabilityCalculatorFactoryMock,
        optimizerFactory: optimizerFactoryMock,
        batchSize: 1,
        randomSeed: 123,
      );

      verify(labelsProcessorFactoryMock.create(Float32x4)).called(1);
      verify(interceptPreprocessorFactoryMock.create(Float32x4, scale: 1.0)).called(1);
      verify(probabilityCalculatorFactoryMock.create(LinkFunctionType.logit, Float32x4)).called(1);
      verify(optimizerFactoryMock.gradient(
        costFnType: CostFunctionType.logLikelihood,
        learningRateType: LearningRateType.constant,
        initialWeightsType: InitialWeightsType.zeroes,
        linkFunctionType: LinkFunctionType.logit,
        initialLearningRate: 0.01,
        minCoefficientsUpdate: 0.001,
        iterationLimit: 100,
        lambda: 0.1,
        batchSize: 1,
        randomSeed: 123,
      )).called(1);
    });
  });
}
