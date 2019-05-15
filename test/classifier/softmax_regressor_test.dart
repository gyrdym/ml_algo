import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/classifier/softmax_regressor.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/optimizer/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_algo/src/score_to_prob_mapper/score_to_prob_mapper_type.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../test_utils/helpers/floating_point_iterable_matchers.dart';
import '../test_utils/mocks.dart';
import 'classifier_common.dart';

void main() {
  group('SoftmaxRegressor', () {
    test('should initialize properly', () {
      final dtype = DType.float32;
      final scoreToProbFactoryMock = createScoreToProbMapperFactoryMock(dtype,
        mappers: {
          ScoreToProbMapperType.logit: ScoreToProbMapperMock(),
        },
      );
      final observations = Matrix.fromList([[1.0]]);
      final outcomes = Matrix.fromList([[0]]);
      final optimizerMock = OptimizerMock();
      final optimizerFactoryMock = createOptimizerFactoryMock(
        observations, outcomes, optimizers: {
          OptimizerType.gradientDescent: optimizerMock,
        },
      );

      SoftmaxRegressor(
        observations, outcomes,
        dtype: dtype,
        learningRateType: LearningRateType.constant,
        initialWeightsType: InitialWeightsType.zeroes,
        iterationsLimit: 100,
        initialLearningRate: 0.01,
        minWeightsUpdate: 0.001,
        lambda: 0.1,
        scoreToProbMapperFactory: scoreToProbFactoryMock,
        optimizer: OptimizerType.gradientDescent,
        optimizerFactory: optimizerFactoryMock,
        gradientType: GradientType.stochastic,
        randomSeed: 123,
      );

      verify(scoreToProbFactoryMock
          .fromType(ScoreToProbMapperType.softmax, dtype))
          .called(1);
      verify(optimizerFactoryMock.fromType(
        OptimizerType.gradientDescent,
        observations,
        outcomes,
        dtype: dtype,
        costFunctionType: CostFunctionType.logLikelihood,
        learningRateType: LearningRateType.constant,
        initialWeightsType: InitialWeightsType.zeroes,
        scoreToProbMapperType: ScoreToProbMapperType.softmax,
        initialLearningRate: 0.01,
        minCoefficientsUpdate: 0.001,
        iterationLimit: 100,
        lambda: 0.1,
        batchSize: 1,
        randomSeed: 123,
      )).called(1);
    });

    test('should call optimizer\'s `findExtrema` method with proper '
        'parameters', () {

      final features = Matrix.fromList([
        [10.1, 10.2, 12.0, 13.4],
        [13.1, 15.2, 61.0, 27.2],
        [30.1, 25.2, 62.0, 34.1],
        [32.1, 35.2, 36.0, 41.5],
        [35.1, 95.2, 56.0, 52.6],
        [90.1, 20.2, 10.0, 12.1],
      ]);

      final labels = Matrix.fromList([
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
      final optimizerFactoryMock = createOptimizerFactoryMock(
        features, labels, optimizers: {
          OptimizerType.gradientDescent: optimizerMock,
        },
      );

      when(optimizerMock.findExtrema(
          initialWeights: argThat(equals(initialWeights),
              named: 'initialWeights'),
          isMinimizingObjective: false))
          .thenReturn(Matrix.fromList([
            [100.0, 10.0, 1.0],
            [200.0, 20.0, 2.0],
            [300.0, 30.0, 3.0],
            [400.0, 40.0, 4.0],
            [500.0, 50.0, 5.0],
          ])
      );

      SoftmaxRegressor(
        features,
        labels,
        dtype: DType.float32,
        learningRateType: LearningRateType.constant,
        initialWeightsType: InitialWeightsType.zeroes,
        iterationsLimit: 100,
        initialLearningRate: 0.01,
        minWeightsUpdate: 0.001,
        lambda: 0.1,
        scoreToProbMapperFactory: scoreToProbFactoryMock,
        optimizer: OptimizerType.gradientDescent,
        optimizerFactory: optimizerFactoryMock,
        gradientType: GradientType.stochastic,
        randomSeed: 123,
      )..fit(initialWeights: initialWeights);

      verify(optimizerFactoryMock.fromType(
        OptimizerType.gradientDescent,
        argThat(matrixAlmostEqualTo([
          [1.0, 10.1, 10.2, 12.0, 13.4],
          [1.0, 13.1, 15.2, 61.0, 27.2],
          [1.0, 30.1, 25.2, 62.0, 34.1],
          [1.0, 32.1, 35.2, 36.0, 41.5],
          [1.0, 35.1, 95.2, 56.0, 52.6],
          [1.0, 90.1, 20.2, 10.0, 12.1],
        ], 1e-2)),
        argThat(equals([
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 1.0, 0.0],
          [1.0, 0.0, 0.0],
          [1.0, 0.0, 0.0],
          [1.0, 0.0, 0.0],
        ])),
      )).called(1);
    });
  });
}
