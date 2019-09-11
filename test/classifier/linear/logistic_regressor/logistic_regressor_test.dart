import 'package:ml_algo/src/classifier/linear/logistic_regressor/gradient_logistic_regressor.dart';
import 'package:ml_algo/src/solver/linear/gradient/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/solver/linear/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../../test_utils/mocks.dart';

void main() {
  group('LogisticRegressor', () {
    test('should initialize properly', () {
      final observations = Matrix.fromList([[1.0]]);
      final outcomes = Matrix.fromList([[0]]);
      final optimizerMock = OptimizerMock();
      final optimizerFactoryMock = createGradientOptimizerFactoryMock(
          observations, outcomes, optimizerMock);

      GradientLogisticRegressor(
        observations,
        outcomes,
        dtype: DType.float32,
        learningRateType: LearningRateType.constant,
        initialWeightsType: InitialWeightsType.zeroes,
        iterationsLimit: 100,
        initialLearningRate: 0.01,
        minWeightsUpdate: 0.001,
        lambda: 0.1,
        optimizerFactory: optimizerFactoryMock,
        randomSeed: 123,
      );

      verify(optimizerFactoryMock.gradient(
        argThat(equals([[1.0]])),
        argThat(equals([[0]])),
        dtype: DType.float32,
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

    test('should make calls of appropriate method when `fit` is called', () {
      final observations = Matrix.fromList([
        [10.1, 10.2, 12.0, 13.4],
        [3.1, 5.2, 6.0, 77.4],
      ]);
      final outcomes = Matrix.fromList([
        [1.0],
        [0.0],
      ]);
      final optimizerMock = OptimizerMock();
      final optimizerFactoryMock = createGradientOptimizerFactoryMock(
          observations, outcomes, optimizerMock);

      final initialWeights = Matrix.fromList([
        [10.0],
        [20.0],
        [30.0],
        [40.0],
      ]);

      GradientLogisticRegressor(
        observations,
        outcomes,
        dtype: DType.float32,
        learningRateType: LearningRateType.constant,
        initialWeightsType: InitialWeightsType.zeroes,
        iterationsLimit: 100,
        initialLearningRate: 0.01,
        minWeightsUpdate: 0.001,
        lambda: 0.1,
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
