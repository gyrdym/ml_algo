import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/classifier/logistic_regressor/_injector.dart';
import 'package:ml_algo/src/common/exception/invalid_probability_threshold_exception.dart';
import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/cost_function/cost_function_factory.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/di/injector.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_factory.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';
import 'package:ml_algo/src/linear_optimizer/regularization_type.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../../helpers.dart';
import '../../../mocks.dart';
import '../../../mocks.mocks.dart';

void main() {
  group('LogisticRegressor', () {
    final negativeLabel = 100;
    final positiveLabel = 200;
    final observations = DataFrame([
      <num>[10.1, 10.2, 12.0, 13.4, positiveLabel],
      <num>[ 3.1,  5.2,  6.0, 77.4, negativeLabel],
    ], headerExists: false);

    final initialCoefficients = Vector.fromList([10, 20, 30, 40, 50]);
    final learnedCoefficients = Matrix.column([100, 200, 300, 400, 500]);
    final targetColumnName = 'col_4';
    final errors = <num>[];

    late LinkFunction linkFunctionMock;
    late CostFunction costFunctionMock;
    late CostFunctionFactory costFunctionFactoryMock;
    late MockLinearOptimizer optimizerMock;
    late MockLinearOptimizerFactory optimizerFactoryMock;

    setUp(() {
      injector.clearAll();
      logisticRegressorInjector.clearAll();

      linkFunctionMock = MockLinkFunction();
      costFunctionMock = MockCostFunction();
      costFunctionFactoryMock = createCostFunctionFactoryMock(costFunctionMock);
      optimizerMock = MockLinearOptimizer();
      optimizerFactoryMock = createLinearOptimizerFactoryMock(optimizerMock);

      injector
        ..registerDependency<CostFunctionFactory>(
                () => costFunctionFactoryMock)
        ..registerSingleton<LinearOptimizerFactory>(
                () => optimizerFactoryMock);

      logisticRegressorInjector
        .registerSingleton<LinkFunction>(() => linkFunctionMock);

      when(optimizerMock.findExtrema(
        initialCoefficients: anyNamed('initialCoefficients'),
        isMinimizingObjective: anyNamed('isMinimizingObjective'),
      )).thenReturn(learnedCoefficients);

      when(optimizerMock.costPerIteration).thenReturn(errors);
    });

    tearDown(() {
      injector.clearAll();
      logisticRegressorInjector.clearAll();
    });

    test('should throw an exception if a probability threshold is less than '
        '0.0', () {
      final probabilityThreshold = -0.01;
      final actual = () => LogisticRegressor(
        observations,
        targetColumnName,
        probabilityThreshold: probabilityThreshold,
        initialCoefficients: Vector.empty(),
      );

      expect(actual, throwsA(isA<InvalidProbabilityThresholdException>()));
    });

    test('should throw an exception if a probability threshold is equal to '
        '0.0', () {
      final probabilityThreshold = 0.0;
      final actual = () => LogisticRegressor(
        observations,
        targetColumnName,
        probabilityThreshold: probabilityThreshold,
        initialCoefficients: Vector.empty(),
      );

      expect(actual, throwsA(isA<InvalidProbabilityThresholdException>()));
    });

    test('should throw an exception if a probability threshold is equal to '
        '1.0', () {
      final probabilityThreshold = 1.0;
      final actual = () => LogisticRegressor(
        observations,
        targetColumnName,
        probabilityThreshold: probabilityThreshold,
        initialCoefficients: Vector.empty(),
      );

      expect(actual, throwsA(isA<InvalidProbabilityThresholdException>()));
    });

    test('should throw an exception if a probability threshold is greater than '
        '1.0', () {
      final probabilityThreshold = 1.01;
      final actual = () => LogisticRegressor(
        observations,
        targetColumnName,
        probabilityThreshold: probabilityThreshold,
        initialCoefficients: Vector.empty(),
      );

      expect(actual, throwsA(isA<InvalidProbabilityThresholdException>()));
    });

    test('should throw an exception if a target column does not exist', () {
      final targetColumnName = 'col_10';

      final actual = () => LogisticRegressor(
        observations,
        targetColumnName,
        initialCoefficients: Vector.empty(),
      );

      expect(actual, throwsException);
    });

    test('should throw an exception if too few initial coefficients '
        'provided', () {
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

    test('should call cost function factory in order to create '
        'loglikelihood cost function', () {
      LogisticRegressor(
        observations,
        'col_4',
        positiveLabel: positiveLabel,
        negativeLabel: negativeLabel,
        initialCoefficients: Vector.empty(),
      );

      verify(costFunctionFactoryMock.createByType(
        CostFunctionType.logLikelihood,
        linkFunction: linkFunctionMock,
        positiveLabel: positiveLabel,
        negativeLabel: negativeLabel,
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

    test('should find the extrema for provided observations while '
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

    test('should create a proper instance', () {
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
        initialCoefficients: Vector.empty(),
        interceptScale: interceptScale,
        isFittingDataNormalized: true,
        positiveLabel: positiveLabel,
        negativeLabel: negativeLabel,
        dtype: dtype,
      );

      expect(classifier.linkFunction, linkFunctionMock);
      expect(classifier.interceptScale, interceptScale);
      expect(classifier.fitIntercept, fitIntercept);
      expect(classifier.dtype, dtype);
      expect(classifier.coefficientsByClasses, learnedCoefficients);
      expect(classifier.targetNames, [targetName]);
    });
  });
}
