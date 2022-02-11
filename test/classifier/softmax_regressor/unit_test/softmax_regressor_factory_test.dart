import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/_injector.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor_factory.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor_factory_impl.dart';
import 'package:ml_algo/src/common/exception/unsupported_linear_optimizer_type_exception.dart';
import 'package:ml_algo/src/cost_function/cost_function_factory.dart';
import 'package:ml_algo/src/cost_function/cost_function_type.dart';
import 'package:ml_algo/src/di/injector.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_factory.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../../helpers.dart';
import '../../../mocks.dart';
import '../../../mocks.mocks.dart';

void main() {
  group('SoftmaxRegressorFactoryImpl', () {
    const defaultNegativeLabel = 10.0;
    const defaultPositiveLabel = 20.0;
    const defaultNormalizedFlag = false;
    const defaultDecay = 0.05;
    const defaultDropRate = 1;
    final features = Matrix.fromList([
      [10.1, 10.2, 12.0, 13.4],
      [13.1, 15.2, 61.0, 27.2],
      [30.1, 25.2, 62.0, 34.1],
      [32.1, 35.2, 36.0, 41.5],
      [35.1, 95.2, 56.0, 52.6],
      [90.1, 20.2, 10.0, 12.1],
    ]);
    final outcomes = Matrix.fromList([
      [defaultPositiveLabel, defaultNegativeLabel, defaultNegativeLabel],
      [defaultNegativeLabel, defaultNegativeLabel, defaultPositiveLabel],
      [defaultNegativeLabel, defaultPositiveLabel, defaultNegativeLabel],
      [defaultPositiveLabel, defaultNegativeLabel, defaultNegativeLabel],
      [defaultNegativeLabel, defaultNegativeLabel, defaultPositiveLabel],
      [defaultPositiveLabel, defaultNegativeLabel, defaultNegativeLabel],
    ]);
    final defaultTrainData = DataFrame.fromMatrix(
      Matrix.fromColumns([
        ...features.columns,
        ...outcomes.columns,
      ], dtype: DType.float32),
      header: ['a', 'b', 'c', 'd', 'target_1', 'target_2', 'target_3'],
    );
    final defaultInitialCoefficients = Matrix.fromList([
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

    late MockLinkFunction linkFunctionMock;
    late MockCostFunction costFunctionMock;
    late CostFunctionFactory costFunctionFactoryMock;
    late MockLinearOptimizer optimizerMock;
    late MockLinearOptimizerFactory optimizerFactoryMock;
    late SoftmaxRegressorFactory factory;

    final createRegressor = ({
      DataFrame? trainData,
      LinearOptimizerType? optimizerType,
      LearningRateType? learningRateType,
      InitialCoefficientsType? initialCoefficientsType,
      required Iterable<String> targetColumnNames,
      int iterationsLimit = 100,
      double initialLearningRate = 0.01,
      double decay = defaultDecay,
      int dropRate = defaultDropRate,
      double minCoefficientsUpdate = 0.001,
      double lambda = 0.1,
      RegularizationType? regularizationType,
      bool fitIntercept = false,
      double interceptScale = 1,
      Matrix? initialCoefficients,
      int? randomSeed,
      double? positiveLabel,
      double? negativeLabel,
      bool? isFittingDataNormalized,
      bool collectLearningData = false,
      DType dtype = DType.float32,
      int? batchSize,
    }) =>
        factory.create(
          trainData: trainData ?? defaultTrainData,
          targetNames: targetColumnNames,
          optimizerType: optimizerType ?? LinearOptimizerType.gradient,
          learningRateType: learningRateType ?? LearningRateType.constant,
          initialCoefficientsType:
              initialCoefficientsType ?? InitialCoefficientsType.zeroes,
          iterationsLimit: iterationsLimit,
          initialLearningRate: initialLearningRate,
          decay: decay,
          dropRate: dropRate,
          minCoefficientsUpdate: minCoefficientsUpdate,
          lambda: lambda,
          regularizationType: regularizationType,
          fitIntercept: fitIntercept,
          interceptScale: interceptScale,
          initialCoefficients:
              initialCoefficients ?? defaultInitialCoefficients,
          randomSeed: randomSeed,
          positiveLabel: positiveLabel ?? defaultPositiveLabel,
          negativeLabel: negativeLabel ?? defaultNegativeLabel,
          isFittingDataNormalized:
              isFittingDataNormalized ?? defaultNormalizedFlag,
          collectLearningData: collectLearningData,
          dtype: dtype,
          batchSize: batchSize ?? ((trainData ?? defaultTrainData).rows.length),
        );

    setUp(() {
      linkFunctionMock = MockLinkFunction();
      costFunctionMock = MockCostFunction();
      costFunctionFactoryMock = createCostFunctionFactoryMock(costFunctionMock);
      optimizerMock = MockLinearOptimizer();
      optimizerFactoryMock = createLinearOptimizerFactoryMock(optimizerMock);

      injector
        ..clearAll()
        ..registerDependency<CostFunctionFactory>(() => costFunctionFactoryMock)
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

    test(
        'should throw an exception if optimizer type is LinearOptimizerType.closedForm',
        () {
      final targetColumnNames = ['target_1'];

      final actual = () => createRegressor(
          targetColumnNames: targetColumnNames,
          optimizerType: LinearOptimizerType.closedForm);

      expect(actual, throwsA(isA<UnsupportedLinearOptimizerTypeException>()));
    });

    test('should throw an exception if some target columns do not exist', () {
      final targetColumnNames = ['target_1', 'some', 'unknown', 'columns'];

      final actual =
          () => createRegressor(targetColumnNames: targetColumnNames);

      expect(actual, throwsException);
    });

    test(
        'should throw an exception if target columns number is less than '
        'two, since the SoftmaxRegressor supports only multiclass '
        'classification with one-hot (or other similar method) encoded '
        'features', () {
      final targetColumnNames = ['target_1'];

      final actual = () => createRegressor(
            targetColumnNames: targetColumnNames,
          );

      expect(actual, throwsException);
    });

    test(
        'should call cost function factory in order to create '
        'loglikelihood cost function', () {
      createRegressor(
        targetColumnNames: ['target_1', 'target_2', 'target_3'],
      );

      verify(costFunctionFactoryMock.createByType(
        CostFunctionType.logLikelihood,
        linkFunction: linkFunctionMock,
        positiveLabel: defaultPositiveLabel,
        negativeLabel: defaultNegativeLabel,
      )).called(1);
    });

    test(
        'should call linear optimizer factory and consider intercept term '
        'while calling the factory', () {
      createRegressor(
        targetColumnNames: ['target_1', 'target_2', 'target_3'],
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
        initialCoefficients: defaultInitialCoefficients,
        randomSeed: 123,
        negativeLabel: defaultNegativeLabel,
        positiveLabel: defaultPositiveLabel,
        dtype: DType.float32,
        batchSize: 1,
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
        decay: defaultDecay,
        dropRate: defaultDropRate,
        minCoefficientsUpdate: 0.001,
        iterationLimit: 100,
        lambda: 0.1,
        regularizationType: RegularizationType.L2,
        batchSize: 1,
        randomSeed: 123,
        isFittingDataNormalized: defaultNormalizedFlag,
      )).called(1);
    });

    test(
        'should find the extrema for fitting observations while '
        'instantiating', () {
      createRegressor(
        targetColumnNames: ['target_1', 'target_2', 'target_3'],
        initialCoefficients: defaultInitialCoefficients,
        dtype: DType.float32,
      );

      verify(optimizerMock.findExtrema(
        initialCoefficients: defaultInitialCoefficients,
        isMinimizingObjective: false,
      )).called(1);
    });

    test(
        'should pass collectLearningData to the optimizer mock\'s findExtrema '
        'method, collectLearningData=true', () {
      createRegressor(
        targetColumnNames: ['target_1', 'target_2', 'target_3'],
        collectLearningData: true,
        dtype: DType.float32,
      );

      verify(optimizerMock.findExtrema(
        initialCoefficients: anyNamed('initialCoefficients'),
        isMinimizingObjective: anyNamed('isMinimizingObjective'),
        collectLearningData: true,
      )).called(1);
    });

    test(
        'should pass collectLearningData to the optimizer mock\'s findExtrema '
        'method, collectLearningData=false', () {
      createRegressor(
        targetColumnNames: ['target_1', 'target_2', 'target_3'],
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
