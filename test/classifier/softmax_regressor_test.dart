import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/gradient_softmax_regressor.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_algo/src/score_to_prob_mapper/score_to_prob_mapper_type.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../test_utils/helpers/floating_point_iterable_matchers.dart';
import '../test_utils/mocks.dart';

void main() {
  group('SoftmaxRegressor', () {
    final dtype = DType.float32;

    test('should initialize properly', () {
      final scoreToProbFactoryMock = createScoreToProbMapperFactoryMock(dtype,
        mappers: {
          ScoreToProbMapperType.logit: ScoreToProbMapperMock(),
        },
      );
      final observations = Matrix.fromList([[1.0]]);
      final outcomes = Matrix.fromList([[0]]);
      final optimizerMock = OptimizerMock();
      final optimizerFactoryMock = createGradientOptimizerFactoryMock(
        observations, outcomes, optimizerMock);

      GradientSoftmaxRegressor(
        observations, outcomes,
        dtype: dtype,
        learningRateType: LearningRateType.constant,
        initialWeightsType: InitialWeightsType.zeroes,
        iterationsLimit: 100,
        initialLearningRate: 0.01,
        minWeightsUpdate: 0.001,
        lambda: 0.1,
        scoreToProbMapperFactory: scoreToProbFactoryMock,
        optimizerFactory: optimizerFactoryMock,
        randomSeed: 123,
      );

      verify(scoreToProbFactoryMock
          .fromType(ScoreToProbMapperType.softmax, dtype))
          .called(1);
      verify(optimizerFactoryMock.gradient(
        observations,
        outcomes,
        dtype: dtype,
        costFunction: anyNamed('costFunction'),
        learningRateType: LearningRateType.constant,
        initialWeightsType: InitialWeightsType.zeroes,
        initialLearningRate: 0.01,
        minCoefficientsUpdate: 0.001,
        iterationLimit: 100,
        lambda: 0.1,
        batchSize: 1,
        randomSeed: 123,
      )).called(1);
    });

    test('should call optimizer\'s `findExtrema` method with proper '
        'parameters and consider intercept term', () {
      final scoreToProbFactoryMock = createScoreToProbMapperFactoryMock(dtype,
        mappers: {
          ScoreToProbMapperType.logit: ScoreToProbMapperMock(),
        },
      );

      final observations = Matrix.fromList([
        [10.1, 10.2, 12.0, 13.4],
        [13.1, 15.2, 61.0, 27.2],
        [30.1, 25.2, 62.0, 34.1],
        [32.1, 35.2, 36.0, 41.5],
        [35.1, 95.2, 56.0, 52.6],
        [90.1, 20.2, 10.0, 12.1],
      ]);

      final outcomes = Matrix.fromList([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
      ]);

      final initialWeights = Matrix.fromList([
        [1.0],
        [10.0],
        [20.0],
        [30.0],
        [40.0],
      ]);

      final optimizerMock = OptimizerMock();
      final optimizerFactoryMock = createGradientOptimizerFactoryMock(
        argThat(matrixAlmostEqualTo([
          [2.0, 10.1, 10.2, 12.0, 13.4],
          [2.0, 13.1, 15.2, 61.0, 27.2],
          [2.0, 30.1, 25.2, 62.0, 34.1],
          [2.0, 32.1, 35.2, 36.0, 41.5],
          [2.0, 35.1, 95.2, 56.0, 52.6],
          [2.0, 90.1, 20.2, 10.0, 12.1],
        ], 1e-2)), outcomes, optimizerMock,
      );

      GradientSoftmaxRegressor(
        observations,
        outcomes,
        dtype: dtype,
        learningRateType: LearningRateType.constant,
        initialWeightsType: InitialWeightsType.zeroes,
        iterationsLimit: 100,
        initialLearningRate: 0.01,
        minWeightsUpdate: 0.001,
        lambda: 0.1,
        fitIntercept: true,
        interceptScale: 2.0,
        scoreToProbMapperFactory: scoreToProbFactoryMock,
        optimizerFactory: optimizerFactoryMock,
        initialWeights: initialWeights,
        randomSeed: 123,
      );

      verify(optimizerFactoryMock.gradient(
        argThat(matrixAlmostEqualTo([
          [2.0, 10.1, 10.2, 12.0, 13.4],
          [2.0, 13.1, 15.2, 61.0, 27.2],
          [2.0, 30.1, 25.2, 62.0, 34.1],
          [2.0, 32.1, 35.2, 36.0, 41.5],
          [2.0, 35.1, 95.2, 56.0, 52.6],
          [2.0, 90.1, 20.2, 10.0, 12.1],
        ], 1e-2)),
        argThat(equals([
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 1.0, 0.0],
          [1.0, 0.0, 0.0],
          [1.0, 0.0, 0.0],
          [1.0, 0.0, 0.0],
        ])),
        dtype: dtype,
        costFunction: anyNamed('costFunction'),
        learningRateType: LearningRateType.constant,
        initialWeightsType: InitialWeightsType.zeroes,
        initialLearningRate: 0.01,
        minCoefficientsUpdate: 0.001,
        iterationLimit: 100,
        lambda: 0.1,
        batchSize: 1,
        randomSeed: 123,
      )).called(1);

      verify(optimizerMock.findExtrema(
        initialWeights: initialWeights,
        isMinimizingObjective: false,
      )).called(1);
    });
  });
}
