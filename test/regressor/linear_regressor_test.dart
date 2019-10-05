import 'package:injector/injector.dart';
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/cost_function/cost_function_factory.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/di/injector.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_factory.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';
import 'package:ml_algo/src/linear_optimizer/regularization_type.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:ml_tech/unit_testing/matchers/iterable_2d_almost_equal_to.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../mocks.dart';

void main() {
  group('LinearRegressor', () {
    final initialCoefficients = Matrix.fromList([
      [1],
      [2],
      [3],
      [4],
      [5],
    ]);

    final learnedCoefficients = Matrix.fromColumns([
      Vector.fromList([55, 66, 77, 88, 99]),
    ]);

    final observations = DataFrame(
      [
        <num>[10, 20, 30, 40, 200],
        <num>[11, 22, 33, 44, 500],
      ],
      header: ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'target'],
      headerExists: false,
      dtype: DType.float32,
    );

    CostFunction costFunctionMock;
    CostFunctionFactory costFunctionFactoryMock;

    LinearOptimizer linearOptimizerMock;
    LinearOptimizerFactory linearOptimizerFactoryMock;

    setUp(() {
      costFunctionMock = CostFunctionMock();
      costFunctionFactoryMock = createCostFunctionFactoryMock(costFunctionMock);

      linearOptimizerMock = LinearOptimizerMock();
      linearOptimizerFactoryMock = createLinearOptimizerFactoryMock(
          linearOptimizerMock);

      injector = Injector()
        ..registerDependency<CostFunctionFactory>(
                (_) => costFunctionFactoryMock)
        ..registerDependency<LinearOptimizerFactory>(
                (_) => linearOptimizerFactoryMock);

      when(linearOptimizerMock.findExtrema(
        initialCoefficients: anyNamed('initialCoefficients'),
        isMinimizingObjective: anyNamed('isMinimizingObjective'),
      )).thenReturn(learnedCoefficients);
    });

    tearDownAll(() => injector = null);

    test('should throw an error if the target column does not exist', () {
      final targetColumn = 'absent_column';
      expect(
        () => LinearRegressor(observations, targetColumn),
        throwsException,
      );
    });

    test('should call cost function factory in order to create '
        'squared cost function instance', () {
      LinearRegressor(observations, 'target');

      verify(costFunctionFactoryMock.createByType(
        CostFunctionType.squared,
      )).called(1);
    });

    test('should call linear optimizer factory and consider intercept term '
        'while calling the factory', () {
      LinearRegressor(observations, 'target',
        optimizerType: LinearOptimizerType.vanillaCD,
        iterationsLimit: 1000,
        initialLearningRate: 5,
        minCoefficientsUpdate: 1000,
        lambda: 20.0,
        regularizationType: RegularizationType.L1,
        randomSeed: 200,
        batchSize: 100,
        fitIntercept: true,
        interceptScale: 3.0,
        learningRateType: LearningRateType.decreasing,
        initialCoefficientsType: InitialCoefficientsType.zeroes,
        initialCoefficients: initialCoefficients,
      );

      verify(linearOptimizerFactoryMock.createByType(
        LinearOptimizerType.vanillaCD,
        argThat(iterable2dAlmostEqualTo([
          [3.0, 10, 20, 30, 40],
          [3.0, 11, 22, 33, 44],
        ])),
        argThat(equals([
          [200],
          [500],
        ])),
        dtype: DType.float32,
        costFunction: costFunctionMock,
        learningRateType: LearningRateType.decreasing,
        initialCoefficientsType: InitialCoefficientsType.zeroes,
        initialLearningRate: 5,
        minCoefficientsUpdate: 1000,
        iterationLimit: 1000,
        lambda: 20.0,
        regularizationType: RegularizationType.L1,
        batchSize: 100,
        randomSeed: 200,
        isFittingDataNormalized: false,
      )).called(1);
    });

    test('should find the extrema for fitting observations while '
        'instantiating', () {
      LinearRegressor(observations, 'target',
        initialCoefficients: initialCoefficients,
      );

      verify(linearOptimizerMock.findExtrema(
        initialCoefficients: initialCoefficients,
        isMinimizingObjective: true,
      )).called(1);
    });

    test('should predict values basing on learned coefficients', () {
      final predictor = LinearRegressor(
        observations,
        'target',
        initialCoefficients: initialCoefficients,
        fitIntercept: true,
        interceptScale: 2.0,
      );

      final features = Matrix.fromList([
        [55, 44, 33, 22],
        [10, 88, 77, 11],
        [12, 22, 39, 13],
      ]);

      final featuresWithIntercept = Matrix.fromColumns([
        Vector.filled(3, 2),
        ...features.columns,
      ]);

      final prediction = predictor.predict(
        DataFrame.fromMatrix(features),
      );

      expect(prediction.header, equals(['target']));

      expect(prediction.toMatrix(), equals(
        featuresWithIntercept * learnedCoefficients,
      ));
    });
  });
}
