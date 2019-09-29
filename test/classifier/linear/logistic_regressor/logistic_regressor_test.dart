import 'package:injector/injector.dart';
import 'package:ml_algo/src/classifier/linear/logistic_regressor/logistic_regressor.dart';
import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/cost_function/cost_function_factory.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/di/injector.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_algo/src/link_function/link_function_factory.dart';
import 'package:ml_algo/src/linear_optimizer/gradient/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/initial_weights_generator/initial_weights_type.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_factory.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';
import 'package:ml_algo/src/link_function/link_function_type.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_tech/unit_testing/matchers/iterable_2d_almost_equal_to.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../../test_utils/mocks.dart';

void main() {
  group('LogisticRegressor', () {
    final observations = DataFrame([
      <num>[10.1, 10.2, 12.0, 13.4, 1],
      <num>[ 3.1,  5.2,  6.0, 77.4, 0],
    ], headerExists: false);

    final initialWeights = Matrix.fromList([
      [10],
      [20],
      [30],
      [40],
      [50],
    ]);

    LinkFunction linkFunctionMock;
    LinkFunctionFactory linkFunctionFactoryMock;

    CostFunction costFunctionMock;
    CostFunctionFactory costFunctionFactoryMock;

    LinearOptimizer optimizerMock;
    LinearOptimizerFactory optimizerFactoryMock;

    setUp(() {
      linkFunctionMock = LinkFunctionMock();
      linkFunctionFactoryMock = createLinkFunctionFactoryMock(linkFunctionMock);

      costFunctionMock = CostFunctionMock();
      costFunctionFactoryMock = createCostFunctionFactoryMock(costFunctionMock);

      optimizerMock = LinearOptimizerMock();
      optimizerFactoryMock = createOptimizerFactoryMock(optimizerMock);

      injector = Injector()
        ..registerSingleton<LinkFunctionFactory>(
                (_) => linkFunctionFactoryMock)
        ..registerDependency<CostFunctionFactory>(
                (_) => costFunctionFactoryMock)
        ..registerSingleton<LinearOptimizerFactory>(
                (_) => optimizerFactoryMock);

      LogisticRegressor(
        observations,
        'col_4',
        dtype: DType.float32,
        learningRateType: LearningRateType.constant,
        initialWeightsType: InitialWeightsType.zeroes,
        iterationsLimit: 100,
        initialLearningRate: 0.01,
        minCoefficientsUpdate: 0.001,
        lambda: 0.1,
        initialWeights: initialWeights,
        randomSeed: 123,
        fitIntercept: true,
        interceptScale: 2.0,
      );
    });

    tearDown(() => injector.clearAll());

    test('should call link function factory twice', () {
      verify(linkFunctionFactoryMock.createByType(
        LinkFunctionType.inverseLogit,
        dtype: DType.float32,
      )).called(2);
    });

    test('should call cost function factory', () {
      verify(costFunctionFactoryMock.createByType(
        CostFunctionType.logLikelihood,
        linkFunction: linkFunctionMock,
      )).called(1);
    });

    test('should call linear optimizer factory and consider intercept term '
        'while calling the factory', () {
      verify(optimizerFactoryMock.createByType(
        LinearOptimizerType.vanillaGD,
        argThat(iterable2dAlmostEqualTo([
          [2.0, 10.1, 10.2, 12.0, 13.4],
          [2.0, 3.1, 5.2, 6.0, 77.4],
        ])),
        argThat(equals([
          [1.0],
          [0.0],
        ])),
        dtype: DType.float32,
        costFunction: costFunctionMock,
        learningRateType: LearningRateType.constant,
        initialWeightsType: InitialWeightsType.zeroes,
        initialLearningRate: 0.01,
        minCoefficientsUpdate: 0.001,
        iterationLimit: 100,
        lambda: 0.1,
        batchSize: 1,
        randomSeed: 123,
        isFittingDataNormalized: anyNamed('isFittingDataNormalized'),
      )).called(1);
    });

    test('should find the extrema for fitting observations while '
        'instantiating', () {
      verify(optimizerMock.findExtrema(
          initialCoefficients: argThat(
              equals(initialWeights),
              named: 'initialCoefficients'
          ),
            isMinimizingObjective: false
      )).called(1);
    });
  });
}
