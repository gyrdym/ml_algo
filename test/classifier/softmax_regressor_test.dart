import 'dart:typed_data';

import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_algo/src/score_to_prob_mapper/score_to_prob_mapper_type.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import 'classifier_common.dart';

void main() {
  final features = MLMatrix.from([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
  ]);

  final labels = MLMatrix.from([
    [10.0],
    [20.0],
  ]);

  group('SoftmaxRegressor', () {
    test('should initialize properly', () {
      final dtype = Float64x2;
      final encoder = CategoricalDataEncoderType.oneHot;

      setUpLabelsProcessorFactory();
      setUpInterceptPreprocessorFactory();
      setUpScoreToProbMapperFactory();
      setUpOptimizerFactory();
      setUpCategoricalDataEncoderFactory();

      createSoftmaxRegressor(dtype: dtype, encoder: encoder);

      verify(labelsProcessorFactoryMock.create(dtype)).called(1);
      verify(interceptPreprocessorFactoryMock.create(dtype, scale: 0.0))
          .called(1);
      verify(scoreToProbFactoryMock.fromType(
          ScoreToProbMapperType.logit, dtype))
          .called(1);
      verify(categoricalDataEncoderFactoryMock.fromType(encoder)).called(1);
      verify(optimizerFactoryMock.fromType(
        OptimizerType.gradientDescent,
        dtype: dtype,
        costFunctionType: CostFunctionType.logLikelihood,
        learningRateType: LearningRateType.constant,
        initialWeightsType: InitialWeightsType.zeroes,
        scoreToProbMapperType: ScoreToProbMapperType.logit,
        initialLearningRate: 0.01,
        minCoefficientsUpdate: 0.001,
        iterationLimit: 100,
        lambda: 0.1,
        batchSize: 1,
        randomSeed: 123,
      )).called(1);
    });

    test('should', () {
      final dtype = Float64x2;
      final encoder = CategoricalDataEncoderType.oneHot;

      setUpLabelsProcessorFactory();
      setUpInterceptPreprocessorFactory();
      setUpScoreToProbMapperFactory();
      setUpOptimizerFactory();
      setUpCategoricalDataEncoderFactory();

      final classifier = createSoftmaxRegressor(dtype: dtype, encoder: encoder);

      classifier.fit(features, labels);
    });
  });
}
