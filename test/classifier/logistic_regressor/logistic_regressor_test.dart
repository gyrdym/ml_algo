import 'package:injector/injector.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/logistic_regressor_factory.dart';
import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/cost_function/cost_function_factory.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/di/injector.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_factory.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';
import 'package:ml_algo/src/linear_optimizer/regularization_type.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_algo/src/link_function/link_function_factory.dart';
import 'package:ml_algo/src/link_function/link_function_type.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:ml_tech/unit_testing/matchers/iterable_2d_almost_equal_to.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../mocks.dart';

void main() {
  group('LogisticRegressor', () {
    final observations = DataFrame([
      <num>[10.1, 10.2, 12.0, 13.4, 1],
      <num>[ 3.1,  5.2,  6.0, 77.4, 0],
    ], headerExists: false);

    final initialCoefficients = Vector.fromList([
      10,
      20,
      30,
      40,
      50,
    ]);

    final learnedCoefficients = Matrix.fromList([
      [100],
      [200],
      [300],
      [400],
      [500],
    ]);

    final negativeLabel = 100;
    final positiveLabel = 200;

    LinkFunction linkFunctionMock;
    LinkFunctionFactory linkFunctionFactoryMock;

    CostFunction costFunctionMock;
    CostFunctionFactory costFunctionFactoryMock;

    LinearOptimizer optimizerMock;
    LinearOptimizerFactory optimizerFactoryMock;

    LogisticRegressor logisticRegressorMock;
    LogisticRegressorFactory logisticRegressorFactoryMock;

    setUp(() {
      linkFunctionMock = LinkFunctionMock();
      linkFunctionFactoryMock = createLinkFunctionFactoryMock(linkFunctionMock);

      costFunctionMock = CostFunctionMock();
      costFunctionFactoryMock = createCostFunctionFactoryMock(costFunctionMock);

      optimizerMock = LinearOptimizerMock();
      optimizerFactoryMock = createLinearOptimizerFactoryMock(optimizerMock);

      logisticRegressorMock = LogisticRegressorMock();
      logisticRegressorFactoryMock = createLogisticRegressorFactoryMock(
          logisticRegressorMock);

      injector = Injector()
        ..registerSingleton<LinkFunctionFactory>(
                (_) => linkFunctionFactoryMock)
        ..registerDependency<CostFunctionFactory>(
                (_) => costFunctionFactoryMock)
        ..registerSingleton<LinearOptimizerFactory>(
                (_) => optimizerFactoryMock)
        ..registerSingleton<LogisticRegressorFactory>(
                (_) => logisticRegressorFactoryMock);

      when(optimizerMock.findExtrema(
        initialCoefficients: anyNamed('initialCoefficients'),
        isMinimizingObjective: anyNamed('isMinimizingObjective'),
      )).thenReturn(learnedCoefficients);
    });

    tearDownAll(() => injector = null);

    test('should throw an exception if a target column does not exist', () {
      final targetColumnName = 'col_10';

      final actual = () => LogisticRegressor(
        observations,
        targetColumnName,
      );

      expect(actual, throwsException);
    });

    test('should throw an exception if too few initial coefficients '
        'provided', () {

      final targetColumnName = 'col_4';

      final actual = () => LogisticRegressor(
        observations,
        targetColumnName,
        initialCoefficients: Vector.fromList([1, 2]),
      );

      expect(actual, throwsException);
    });

    test('should throw an exception if too many initial coefficients '
        'provided', () {

      final targetColumnName = 'col_4';

      final actual = () => LogisticRegressor(
        observations,
        targetColumnName,
        initialCoefficients: Vector.fromList([1, 2, 3, 4, 5, 6]),
      );

      expect(actual, throwsException);
    });

    test('should call link function factory twice', () {
      LogisticRegressor(
        observations,
        'col_4',
      );

      verify(linkFunctionFactoryMock.createByType(
        LinkFunctionType.inverseLogit,
        dtype: DType.float32,
      )).called(2);
    });

    test('should call cost function factory in order to create '
        'loglikelihood cost function', () {
      LogisticRegressor(
        observations,
        'col_4',
      );

      verify(costFunctionFactoryMock.createByType(
        CostFunctionType.logLikelihood,
        linkFunction: linkFunctionMock,
      )).called(1);
    });

    test('should call linear optimizer factory and consider intercept term', () {
      LogisticRegressor(
        observations,
        'col_4',
        learningRateType: LearningRateType.decreasingAdaptive,
        initialCoefficientsType: InitialCoefficientsType.zeroes,
        iterationsLimit: 1000,
        initialLearningRate: 0.01,
        minCoefficientsUpdate: 0.001,
        lambda: 0.1,
        regularizationType: RegularizationType.L2,
        initialCoefficients: initialCoefficients,
        randomSeed: 123,
        fitIntercept: true,
        interceptScale: 2.0,
        isFittingDataNormalized: true,
        positiveLabel: positiveLabel,
        negativeLabel: negativeLabel,
        dtype: DType.float32,
      );

      verify(optimizerFactoryMock.createByType(
        LinearOptimizerType.gradient,
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
        learningRateType: LearningRateType.decreasingAdaptive,
        initialCoefficientsType: InitialCoefficientsType.zeroes,
        initialLearningRate: 0.01,
        minCoefficientsUpdate: 0.001,
        iterationLimit: 1000,
        lambda: 0.1,
        regularizationType: RegularizationType.L2,
        batchSize: 1,
        randomSeed: 123,
        isFittingDataNormalized: true,
      )).called(1);
    });

    test('should find the extrema for fitting observations while '
        'instantiating', () {
      LogisticRegressor(
        observations,
        'col_4',
        initialCoefficients: initialCoefficients,
        fitIntercept: true,
      );

      verify(optimizerMock.findExtrema(
          initialCoefficients: argThat(
            equals(Matrix.fromColumns([initialCoefficients])),
            named: 'initialCoefficients',
          ),
          isMinimizingObjective: false,
      )).called(1);
    });

    test('should call logistic regressor factory in order to create the '
        'classifier instance', () {
      final targetName = 'col_4';
      final probabilityThreshold = 0.7;
      final fitIntercept = true;
      final interceptScale = -12.0;
      final dtype = DType.float32;

      final classifier = LogisticRegressor(
        observations,
        targetName,
        probabilityThreshold: probabilityThreshold,
        fitIntercept: fitIntercept,
        interceptScale: interceptScale,
        isFittingDataNormalized: true,
        positiveLabel: positiveLabel,
        negativeLabel: negativeLabel,
        dtype: dtype,
      );

      verify(logisticRegressorFactoryMock.create(
        targetName,
        linkFunctionMock,
        probabilityThreshold,
        fitIntercept,
        interceptScale,
        learnedCoefficients,
        negativeLabel,
        positiveLabel,
        dtype,
      )).called(1);

      expect(classifier, same(logisticRegressorMock));
    });
  });
}
