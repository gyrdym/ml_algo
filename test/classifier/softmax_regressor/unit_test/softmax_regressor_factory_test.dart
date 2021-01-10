import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/_injector.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor_factory.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor_factory_impl.dart';
import 'package:ml_algo/src/cost_function/cost_function.dart';
import 'package:ml_algo/src/cost_function/cost_function_factory.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/di/injector.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_factory.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';
import 'package:ml_algo/src/linear_optimizer/regularization_type.dart';
import 'package:ml_algo/src/link_function/link_function.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../../helpers.dart';
import '../../../mocks.dart';

void main() {
  group('SoftmaxRegressorFactoryImpl', () {
    final negativeLabel = 10.0;
    final positiveLabel = 20.0;
    final features = Matrix.fromList([
      [10.1, 10.2, 12.0, 13.4],
      [13.1, 15.2, 61.0, 27.2],
      [30.1, 25.2, 62.0, 34.1],
      [32.1, 35.2, 36.0, 41.5],
      [35.1, 95.2, 56.0, 52.6],
      [90.1, 20.2, 10.0, 12.1],
    ]);
    final outcomes = Matrix.fromList([
      [positiveLabel, negativeLabel, negativeLabel],
      [negativeLabel, negativeLabel, positiveLabel],
      [negativeLabel, positiveLabel, negativeLabel],
      [positiveLabel, negativeLabel, negativeLabel],
      [negativeLabel, negativeLabel, positiveLabel],
      [positiveLabel, negativeLabel, negativeLabel],
    ]);
    final observations = DataFrame.fromMatrix(
      Matrix.fromColumns([
        ...features.columns,
        ...outcomes.columns,
      ], dtype: DType.float32),
      header: ['a', 'b', 'c', 'd', 'target_1', 'target_2', 'target_3'],
    );
    final initialCoefficients = Matrix.fromList([
      [1.0],
      [10.0],
      [20.0],
      [30.0],
      [40.0],
    ]);
    final learnedCoefficients = Matrix.fromList([
      [1, 2, 3],
      [1, 2, 3],
      [1, 2, 3],
      [1, 2, 3],
      [1, 2, 3],
    ]);
    final costPerIteration = [10, 0, -10, 33.1];

    LinkFunction linkFunctionMock;
    CostFunction costFunctionMock;
    CostFunctionFactory costFunctionFactoryMock;
    LinearOptimizer optimizerMock;
    LinearOptimizerFactory optimizerFactoryMock;
    SoftmaxRegressorFactory factory;

    setUp(() {
      linkFunctionMock = LinkFunctionMock();
      costFunctionMock = CostFunctionMock();
      costFunctionFactoryMock = createCostFunctionFactoryMock(costFunctionMock);
      optimizerMock = LinearOptimizerMock();
      optimizerFactoryMock = createLinearOptimizerFactoryMock(optimizerMock);

      injector
        ..clearAll()
        ..registerDependency<CostFunctionFactory>(
                () => costFunctionFactoryMock)
        ..registerSingleton<LinearOptimizerFactory>(() => optimizerFactoryMock);

      when(optimizerMock.findExtrema(
        initialCoefficients: anyNamed('initialCoefficients'),
        isMinimizingObjective: anyNamed('isMinimizingObjective'),
        collectLearningData: anyNamed('collectLearningData'),
      )).thenReturn(learnedCoefficients);
      when(optimizerMock.costPerIteration).thenReturn(costPerIteration);

      factory = SoftmaxRegressorFactoryImpl(linkFunctionMock);
    });

    tearDown(() {
      injector.clearAll();
      softmaxRegressorInjector.clearAll();
    });

    test('should throw an exception if some target columns do not exist', () {
      final targetColumnNames = ['target_1', 'some', 'unknown', 'columns'];

      final actual = () => factory.create(
        trainData: observations,
        targetNames: targetColumnNames,
      );

      expect(actual, throwsException);
    });

    test('should throw an exception if target columns number is less than '
        'two, since the SoftmaxRegressor supports only multiclass '
        'classification with one-hot (or other similar method) encoded '
        'features', () {

      final targetColumnNames =
        ['target_1'];

      final actual = () => factory.create(
        trainData: observations,
        targetNames: targetColumnNames,
      );

      expect(actual, throwsException);
    });

    test('should call cost function factory in order to create '
        'loglikelihood cost function', () {
      factory.create(
        trainData: observations,
        targetNames: ['target_1', 'target_2', 'target_3'],
        positiveLabel: positiveLabel,
        negativeLabel: negativeLabel,
        dtype: DType.float32,
      );

      verify(costFunctionFactoryMock.createByType(
        CostFunctionType.logLikelihood,
        linkFunction: linkFunctionMock,
        positiveLabel: positiveLabel,
        negativeLabel: negativeLabel,
      )).called(1);
    });

    test('should call linear optimizer factory and consider intercept term '
        'while calling the factory', () {
      factory.create(
        trainData: observations,
        targetNames: ['target_1', 'target_2', 'target_3'],
        optimizerType: LinearOptimizerType.gradient,
        learningRateType: LearningRateType.constant,
        initialCoefficientsType: InitialCoefficientsType.zeroes,
        iterationsLimit: 100,
        initialLearningRate: 0.01,
        minCoefficientsUpdate: 0.001,
        lambda: 0.1,
        regularizationType: RegularizationType.L2,
        fitIntercept: true,
        interceptScale: 2.0,
        initialCoefficients: initialCoefficients,
        randomSeed: 123,
        negativeLabel: negativeLabel,
        positiveLabel: positiveLabel,
        dtype: DType.float32,
      );

      verify(optimizerFactoryMock.createByType(
        LinearOptimizerType.gradient,
        argThat(iterable2dAlmostEqualTo([
          [2.0, 10.1, 10.2, 12.0, 13.4],
          [2.0, 13.1, 15.2, 61.0, 27.2],
          [2.0, 30.1, 25.2, 62.0, 34.1],
          [2.0, 32.1, 35.2, 36.0, 41.5],
          [2.0, 35.1, 95.2, 56.0, 52.6],
          [2.0, 90.1, 20.2, 10.0, 12.1],
        ], 1e-2)),
        argThat(equals([
          [1.0, 0.0, 0.0],
          [0.0, 0.0, 1.0],
          [0.0, 1.0, 0.0],
          [1.0, 0.0, 0.0],
          [0.0, 0.0, 1.0],
          [1.0, 0.0, 0.0],
        ])),
        dtype: DType.float32,
        costFunction: costFunctionMock,
        learningRateType: LearningRateType.constant,
        initialCoefficientsType: InitialCoefficientsType.zeroes,
        initialLearningRate: 0.01,
        minCoefficientsUpdate: 0.001,
        iterationLimit: 100,
        lambda: 0.1,
        regularizationType: RegularizationType.L2,
        batchSize: 1,
        randomSeed: 123,
      )).called(1);
    });

    test('should find the extrema for fitting observations while '
        'instantiating', () {
      factory.create(
        trainData: observations,
        targetNames: ['target_1', 'target_2', 'target_3'],
        initialCoefficients: initialCoefficients,
        dtype: DType.float32,
      );

      verify(optimizerMock.findExtrema(
        initialCoefficients: initialCoefficients,
        isMinimizingObjective: false,
      )).called(1);
    });

    test('should pass collectLearningData to the optimizer mock\'s findExtrema '
        'method, collectLearningData=true', () {
      factory.create(
        trainData: observations,
        targetNames: ['target_1', 'target_2', 'target_3'],
        collectLearningData: true,
        dtype: DType.float32,
      );

      verify(optimizerMock.findExtrema(
        initialCoefficients: anyNamed('initialCoefficients'),
        isMinimizingObjective: anyNamed('isMinimizingObjective'),
        collectLearningData: true,
      )).called(1);
    });

    test('should pass collectLearningData to the optimizer mock\'s findExtrema '
        'method, collectLearningData=false', () {
      factory.create(
        trainData: observations,
        targetNames: ['target_1', 'target_2', 'target_3'],
        collectLearningData: false,
        dtype: DType.float32,
      );

      verify(optimizerMock.findExtrema(
        initialCoefficients: anyNamed('initialCoefficients'),
        isMinimizingObjective: anyNamed('isMinimizingObjective'),
        collectLearningData: false,
      )).called(1);
    });
  });
}
