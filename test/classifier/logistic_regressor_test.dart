import 'dart:typed_data';

import 'package:ml_algo/learning_rate_type.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/link_function/link_function_type.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_algo/src/optimizer/optimizer_type.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:mockito/mockito.dart';
import 'package:test_api/test_api.dart';

import '../test_utils/helpers/floating_point_iterable_matchers.dart';
import 'logistic_regressor_common.dart';

void main() {
  group('LogisticRegressor', () {
    tearDown(resetMockitoState);

    test('should initialize properly', () {
      setUpLabelsProcessorFactory();
      setUpInterceptPreprocessorFactory();
      setUpProbabilityCalculatorFactory();
      setUpOptimizerFactory();

      createRegressor();

      verify(labelsProcessorFactoryMock.create(Float32x4)).called(1);
      verify(interceptPreprocessorFactoryMock.create(Float32x4, scale: 0.0)).called(1);
      verify(probabilityCalculatorFactoryMock.create(LinkFunctionType.logit, Float32x4)).called(1);
      verify(optimizerFactoryMock.fromType(OptimizerType.gradientDescent,
        dtype: Float32x4,
        costFunctionType: CostFunctionType.logLikelihood,
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

    test('should make appropriate method calls when `fit` is called', () {
      setUpLabelsProcessorFactory();
      setUpInterceptPreprocessorFactory();
      setUpProbabilityCalculatorFactory();
      setUpOptimizerFactory();

      final features = MLMatrix.from([
        [10.1, 10.2, 12.0, 13.4],
        [3.1, 5.2, 6.0, 77.4],
      ]);
      final origLabels = MLVector.from([1.0, 2.0]);

      when(interceptPreprocessorMock.addIntercept(argThat(matrixAlmostEqualTo([
        [10.1, 10.2, 12.0, 13.4],
        [3.1, 5.2, 6.0, 77.4],
      ])))).thenReturn(MLMatrix.from([
        [100.0, 200.0, 300.0, 400.0],
        [500.0, 600.0, 700.0, 800.0],
      ]));

      final initialWeights = MLVector.from([10.0, 20.0, 30.0, 40.0]);

      when(labelsProcessorMock.makeLabelsOneVsAll(argThat(equals([1.0, 2.0])), 1.0))
          .thenReturn(MLVector.from([1.0, 0.0]));
      when(labelsProcessorMock.makeLabelsOneVsAll(argThat(equals([1.0, 2.0])), 2.0))
          .thenReturn(MLVector.from([0.0, 1.0]));

      when(optimizerMock.findExtrema(
        argThat(matrixAlmostEqualTo([
          [100.0, 200.0, 300.0, 400.0],
          [500.0, 600.0, 700.0, 800.0],
        ])),
        argThat(equals([1.0, 0.0])),
        arePointsNormalized: true,
        initialWeights: initialWeights,
        isMinimizingObjective: false
      )).thenReturn(MLVector.from([333.0, 444.0]));

      when(optimizerMock.findExtrema(
          argThat(matrixAlmostEqualTo([
            [100.0, 200.0, 300.0, 400.0],
            [500.0, 600.0, 700.0, 800.0],
          ])),
          argThat(equals([0.0, 1.0])),
          arePointsNormalized: true,
          initialWeights: initialWeights,
          isMinimizingObjective: false
      )).thenReturn(MLVector.from([555.0, 666.0]));

      createRegressor()
        ..fit(features, origLabels, initialWeights: initialWeights, isDataNormalized: true);

      verify(interceptPreprocessorMock.addIntercept(argThat(matrixAlmostEqualTo([
        [10.1, 10.2, 12.0, 13.4],
        [3.1, 5.2, 6.0, 77.4],
      ])))).called(1);

      verify(optimizerMock.findExtrema(
        argThat(matrixAlmostEqualTo([
          [100.0, 200.0, 300.0, 400.0],
          [500.0, 600.0, 700.0, 800.0],
        ])),
        argThat(equals([1.0, 0.0])),
        initialWeights: initialWeights,
        arePointsNormalized: true,
        isMinimizingObjective: false
      )).called(1);

      verify(optimizerMock.findExtrema(
          argThat(matrixAlmostEqualTo([
            [100.0, 200.0, 300.0, 400.0],
            [500.0, 600.0, 700.0, 800.0],
          ])),
          argThat(equals([0.0, 1.0])),
          initialWeights: initialWeights,
          arePointsNormalized: true,
          isMinimizingObjective: false
      )).called(1);
    });
  });
}
