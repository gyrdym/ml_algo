import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_algo/src/optimizer/optimizer_type.dart';
import 'package:ml_algo/src/score_to_prob_mapper/score_to_prob_mapper_type.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../test_utils/helpers/floating_point_iterable_matchers.dart';
import 'classifier_common.dart';

void main() {
  group('LogisticRegressor', () {
    tearDown(resetMockitoState);

    test('should initialize properly', () {
      setUpInterceptPreprocessorFactory();
      setUpScoreToProbMapperFactory();

      final observations = Matrix.fromList([[1.0]]);
      final outcomes = Matrix.fromList([[0]]);

      setUpOptimizerFactory(observations, outcomes);
      createLogisticRegressor(observations, outcomes);

      verify(interceptPreprocessorFactoryMock.create(DType.float32, scale: 0.0))
          .called(1);
      verify(scoreToProbFactoryMock.fromType(
              ScoreToProbMapperType.logit, DType.float32))
          .called(1);
      verify(optimizerFactoryMock.fromType(
        OptimizerType.gradientDescent,
        argThat(equals([[1.0]])),
        argThat(equals([[0]])),
        dtype: DType.float32,
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

    test('should make calls of appropriate method when `fit` is called', () {
      final observations = Matrix.fromList([
        [10.1, 10.2, 12.0, 13.4],
        [3.1, 5.2, 6.0, 77.4],
      ]);
      final outcomes = Matrix.fromList([
        [1.0],
        [0.0],
      ]);

      setUpInterceptPreprocessorFactory();
      setUpScoreToProbMapperFactory();
      setUpOptimizerFactory(observations, outcomes);

      when(interceptPreprocessorMock.addIntercept(argThat(matrixAlmostEqualTo([
        [10.1, 10.2, 12.0, 13.4],
        [3.1, 5.2, 6.0, 77.4],
      ])))).thenReturn(Matrix.fromList([
        [100.0, 200.0, 300.0, 400.0],
        [500.0, 600.0, 700.0, 800.0],
      ]));

      final initialWeights = Matrix.fromList([
        [10.0],
        [20.0],
        [30.0],
        [40.0],
      ]);

      when(
          optimizerMock.findExtrema(
              initialWeights: argThat(equals(initialWeights),
                  named: 'initialWeights'),
              isMinimizingObjective: false)
      ).thenReturn(Matrix.fromRows([
        Vector.fromList([333.0, 444.0])
      ]));

      when(optimizerMock.findExtrema(
              initialWeights: argThat(
                  equals(initialWeights),
                  named: 'initialWeights'
              ),
              isMinimizingObjective: false))
          .thenReturn(Matrix.fromRows([Vector.fromList([555.0, 666.0])]));

      createLogisticRegressor(observations, outcomes)
        ..fit(initialWeights: initialWeights);

      verify(
          interceptPreprocessorMock.addIntercept(argThat(matrixAlmostEqualTo([
        [10.1, 10.2, 12.0, 13.4],
        [3.1, 5.2, 6.0, 77.4],
      ])))).called(1);

      verify(optimizerMock.findExtrema(
              initialWeights: argThat(
                  equals(initialWeights),
                  named: 'initialWeights'),
              isMinimizingObjective: false))
          .called(1);

      verify(optimizerMock.findExtrema(
              initialWeights: argThat(
                  equals(initialWeights),
                  named: 'initialWeights'),
              isMinimizingObjective: false))
          .called(1);
    });
  });
}
