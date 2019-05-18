import 'package:ml_algo/src/classifier/logistic_regressor.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/optimizer/gradient/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_algo/src/optimizer/optimizer_type.dart';
import 'package:ml_algo/src/score_to_prob_mapper/score_to_prob_mapper_type.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../test_utils/mocks.dart';

void main() {
  group('LogisticRegressor', () {
    test('should initialize properly', () {
      final scoreToProbFactoryMock = createScoreToProbMapperFactoryMock(
          DType.float32, mappers: {
            ScoreToProbMapperType.logit: ScoreToProbMapperMock(),
          },
      );
      final observations = Matrix.fromList([[1.0]]);
      final outcomes = Matrix.fromList([[0]]);
      final optimizerMock = OptimizerMock();
      final optimizerFactoryMock = createOptimizerFactoryMock(
        observations, outcomes, optimizers: {
          OptimizerType.gradient: optimizerMock
        },
      );

      LogisticRegressor(
        observations,
        outcomes,
        dtype: DType.float32,
        learningRateType: LearningRateType.constant,
        initialWeightsType: InitialWeightsType.zeroes,
        iterationsLimit: 100,
        initialLearningRate: 0.01,
        minWeightsUpdate: 0.001,
        lambda: 0.1,
        scoreToProbMapperFactory: scoreToProbFactoryMock,
        optimizer: OptimizerType.gradient,
        optimizerFactory: optimizerFactoryMock,
        randomSeed: 123,
      );

      verify(scoreToProbFactoryMock.fromType(
              ScoreToProbMapperType.logit, DType.float32))
          .called(1);
      verify(optimizerFactoryMock.fromType(
        OptimizerType.gradient,
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
      final scoreToProbFactoryMock =
          createScoreToProbMapperFactoryMock(DType.float32, mappers: {
            ScoreToProbMapperType.logit: ScoreToProbMapperMock(),
          });
      final optimizerMock = OptimizerMock();
      final optimizerFactoryMock = createOptimizerFactoryMock(
          observations, outcomes, optimizers: {
            OptimizerType.gradient: optimizerMock
          },
      );

      final initialWeights = Matrix.fromList([
        [10.0],
        [20.0],
        [30.0],
        [40.0],
      ]);

      LogisticRegressor(
        observations,
        outcomes,
        dtype: DType.float32,
        learningRateType: LearningRateType.constant,
        initialWeightsType: InitialWeightsType.zeroes,
        iterationsLimit: 100,
        initialLearningRate: 0.01,
        minWeightsUpdate: 0.001,
        lambda: 0.1,
        scoreToProbMapperFactory: scoreToProbFactoryMock,
        optimizer: OptimizerType.gradient,
        optimizerFactory: optimizerFactoryMock,
        initialWeights: initialWeights,
        randomSeed: 123,
      );

      verify(optimizerMock.findExtrema(
              initialWeights: argThat(
                  equals(initialWeights),
                  named: 'initialWeights'),
              isMinimizingObjective: false))
          .called(1);
    });
  });
}
