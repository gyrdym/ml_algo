import 'package:injector/injector.dart';
import 'package:ml_algo/src/classifier/logistic_regressor_classifier/logistic_regressor.dart';
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

import '../mocks.dart';

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

    setUp(() {
      linkFunctionMock = LinkFunctionMock();
      linkFunctionFactoryMock = createLinkFunctionFactoryMock(linkFunctionMock);

      costFunctionMock = CostFunctionMock();
      costFunctionFactoryMock = createCostFunctionFactoryMock(costFunctionMock);

      optimizerMock = LinearOptimizerMock();
      optimizerFactoryMock = createLinearOptimizerFactoryMock(optimizerMock);

      injector = Injector()
        ..registerSingleton<LinkFunctionFactory>(
                (_) => linkFunctionFactoryMock)
        ..registerDependency<CostFunctionFactory>(
                (_) => costFunctionFactoryMock)
        ..registerSingleton<LinearOptimizerFactory>(
                (_) => optimizerFactoryMock);

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

    test('should call link function factory twice in order to create inverse '
        'logit link function', () {
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

    test('should call linear optimizer factory and consider intercept term '
        'while calling the factory', () {
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
      );

      verify(optimizerMock.findExtrema(
          initialCoefficients: argThat(
            equals(Matrix.fromColumns([initialCoefficients])),
            named: 'initialCoefficients',
          ),
          isMinimizingObjective: false,
      )).called(1);
    });

    test('should predict classes basing on learned coefficients', () {
      final classifier = LogisticRegressor(
        observations,
        'col_4',
        initialCoefficients: initialCoefficients,
        fitIntercept: true,
        interceptScale: 2.0,
        positiveLabel: positiveLabel,
        negativeLabel: negativeLabel,
      );

      final probabilities = Matrix.fromList([
        [0.2],
        [0.3],
        [0.6],
      ]);

      when(linkFunctionMock.link(any)).thenReturn(probabilities);

      final features = Matrix.fromList([
        [55, 44, 33, 22],
        [10, 88, 77, 11],
        [12, 22, 39, 13],
      ]);

      final featuresWithIntercept = Matrix.fromColumns([
        Vector.filled(3, 2),
        ...features.columns,
      ]);

      final classes = classifier.predict(
        DataFrame.fromMatrix(features),
      );

      expect(classes.header, equals(['col_4']));

      expect(classes.toMatrix(), equals([
        [negativeLabel],
        [negativeLabel],
        [positiveLabel],
      ]));

      verify(linkFunctionMock.link(argThat(iterable2dAlmostEqualTo(
          featuresWithIntercept * learnedCoefficients
      )))).called(1);
    });

    test('should predict probabilities of classes basing on learned '
        'coefficients', () {
      final classifier = LogisticRegressor(
        observations,
        'col_4',
        initialCoefficients: initialCoefficients,
        fitIntercept: true,
        interceptScale: 2.0,
      );

      final probabilities = Matrix.fromList([
        [0.2],
        [0.3],
        [0.6],
      ]);

      when(linkFunctionMock.link(any)).thenReturn(probabilities);

      final features = Matrix.fromList([
        [55, 44, 33, 22],
        [10, 88, 77, 11],
        [12, 22, 39, 13],
      ]);

      final featuresWithIntercept = Matrix.fromColumns([
        Vector.filled(3, 2),
        ...features.columns,
      ]);

      final prediction = classifier.predictProbabilities(
        DataFrame.fromMatrix(features),
      );

      expect(prediction.header, equals(['col_4']));

      expect(prediction.toMatrix(), iterable2dAlmostEqualTo([
        [0.2],
        [0.3],
        [0.6],
      ]));

      verify(linkFunctionMock.link(argThat(iterable2dAlmostEqualTo(
          featuresWithIntercept * learnedCoefficients
      )))).called(1);
    });
  });
}
