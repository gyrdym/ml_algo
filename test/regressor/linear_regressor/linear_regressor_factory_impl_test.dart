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
import 'package:ml_algo/src/regressor/linear_regressor/linear_regressor_factory_impl.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../helpers.dart';
import '../../mocks.dart';

void main() {
  group('LinearRegressorFactoryImpl', () {
    final initialCoefficients = Matrix.column([1, 2, 3, 4, 5]);
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
    );
    final costPerIteration = [1, 2, 3, 100];
    final factory = const LinearRegressorFactoryImpl();

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

      module
        ..registerSingleton<CostFunctionFactory>(costFunctionFactoryMock)
        ..registerSingleton<LinearOptimizerFactory>(linearOptimizerFactoryMock);

      when(linearOptimizerMock.findExtrema(
        initialCoefficients: anyNamed('initialCoefficients'),
        isMinimizingObjective: anyNamed('isMinimizingObjective'),
        collectLearningData: anyNamed('collectLearningData'),
      )).thenReturn(learnedCoefficients);
      when(linearOptimizerMock.costPerIteration).thenReturn(costPerIteration);
    });

    tearDown(() async {
      await module.reset();
    });

    test('should throw an error if the target column does not exist', () {
      final targetColumn = 'absent_column';
      expect(
        () => factory.create(
          fittingData: observations,
          targetName: targetColumn,
        ),
        throwsException,
      );
    });

    test('should call cost function factory in order to create '
        'squared cost function instance', () {
      factory.create(
        fittingData: observations,
        targetName: 'target',
      );

      verify(costFunctionFactoryMock.createByType(
        CostFunctionType.leastSquare,
      )).called(1);
    });

    test('should call linear optimizer factory and consider intercept term '
        'while calling the factory', () {
      factory.create(
        fittingData: observations,
        targetName: 'target',
        optimizerType: LinearOptimizerType.coordinate,
        iterationsLimit: 1000,
        initialLearningRate: 5,
        minCoefficientsUpdate: 1000,
        lambda: 20.0,
        regularizationType: RegularizationType.L1,
        randomSeed: 200,
        batchSize: 100,
        fitIntercept: true,
        interceptScale: 3.0,
        learningRateType: LearningRateType.decreasingAdaptive,
        initialCoefficientsType: InitialCoefficientsType.zeroes,
        initialCoefficients: initialCoefficients,
        dtype: DType.float32,
      );

      verify(linearOptimizerFactoryMock.createByType(
        LinearOptimizerType.coordinate,
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
        learningRateType: LearningRateType.decreasingAdaptive,
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
      factory.create(
        fittingData: observations,
        targetName: 'target',
        initialCoefficients: initialCoefficients,
      );

      verify(linearOptimizerMock.findExtrema(
        initialCoefficients: initialCoefficients,
        isMinimizingObjective: true,
      )).called(1);
    });

    test('should predict values basing on learned coefficients', () {
      final predictor = factory.create(
        fittingData: observations,
        targetName: 'target',
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

    test('should collect cost values per iteration if collectLearningData is '
        'true', () {
      final regressor = factory.create(
        fittingData: observations,
        targetName: 'target',
        initialCoefficients: initialCoefficients,
        collectLearningData: true,
      );

      expect(regressor.costPerIteration, same(costPerIteration));
      verify(linearOptimizerMock.findExtrema(
        initialCoefficients: anyNamed('initialCoefficients'),
        isMinimizingObjective: anyNamed('isMinimizingObjective'),
        collectLearningData: true,
      )).called(1);
    });

    test('should not collect cost values per iteration if collectLearningData is '
        'false', () {
      factory.create(
        fittingData: observations,
        targetName: 'target',
        initialCoefficients: initialCoefficients,
        collectLearningData: false,
      );

      verify(linearOptimizerMock.findExtrema(
        initialCoefficients: anyNamed('initialCoefficients'),
        isMinimizingObjective: anyNamed('isMinimizingObjective'),
        collectLearningData: false,
      )).called(1);
    });
  });
}
