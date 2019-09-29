import 'package:injector/injector.dart';
import 'package:ml_algo/src/classifier/linear/logistic_regressor/logistic_regressor.dart';
import 'package:ml_algo/src/cost_function/cost_function_factory.dart';
import 'package:ml_algo/src/di/injector.dart';
import 'package:ml_algo/src/link_function/link_function_factory.dart';
import 'package:ml_algo/src/linear_optimizer/gradient/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_factory.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../../test_utils/mocks.dart';

void main() {
  group('LogisticRegressor', () {
    final linkFunction = LinkFunctionMock();
    final linkFunctionFactoryMock = createLinkFunctionFactoryMock(
        linkFunction);

    final costFunction = CostFunctionMock();
    final costFunctionFactoryMock = createCostFunctionFactoryMock(costFunction);

    final optimizerMock = LinearOptimizerMock();
    final optimizerFactoryMock = createGradientOptimizerFactoryMock(
        optimizerMock);

    setUp(() => injector = Injector()
      ..registerSingleton<LinkFunctionFactory>(
              (_) => linkFunctionFactoryMock)
      ..registerDependency<CostFunctionFactory>(
              (_) => costFunctionFactoryMock)
      ..registerSingleton<LinearOptimizerFactory>(
              (_) => optimizerFactoryMock),
    );

    tearDownAll(() => injector.clearAll());

    test('should initialize properly', () {
      final observations = DataFrame([<num>[1.0, 0]], headerExists: false);

      LogisticRegressor(
        observations,
        'col_1',
        dtype: DType.float32,
        learningRateType: LearningRateType.constant,
        initialWeightsType: InitialWeightsType.zeroes,
        iterationsLimit: 100,
        initialLearningRate: 0.01,
        minWeightsUpdate: 0.001,
        lambda: 0.1,
        randomSeed: 123,
      );

      verify(optimizerFactoryMock.createByType(
        LinearOptimizerType.vanillaGD,
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

    test('should make calls of appropriate method while fitting '
        '(instantiating)', () {

      final observations = DataFrame([
        <num>[10.1, 10.2, 12.0, 13.4, 1.0],
        <num>[3.1, 5.2, 6.0, 77.4, 0.0],
      ], headerExists: false);

      final initialWeights = Matrix.fromList([
        [10.0],
        [20.0],
        [30.0],
        [40.0],
      ]);

      LogisticRegressor(
        observations,
        'col_4',
        dtype: DType.float32,
        learningRateType: LearningRateType.constant,
        initialWeightsType: InitialWeightsType.zeroes,
        iterationsLimit: 100,
        initialLearningRate: 0.01,
        minWeightsUpdate: 0.001,
        lambda: 0.1,
        initialWeights: initialWeights,
        randomSeed: 123,
      );

      verify(optimizerMock.findExtrema(
          initialWeights: argThat(
              equals(initialWeights),
              named: 'initialWeights'
          ),
          isMinimizingObjective: false
      )).called(1);
    });
  });
}
