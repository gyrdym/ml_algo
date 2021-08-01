import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/_injector.dart';
import 'package:ml_algo/src/classifier/softmax_regressor/softmax_regressor_factory.dart';
import 'package:ml_algo/src/di/injector.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../../mocks.mocks.dart';

void main() {
  group('SoftmaxRegressorFactoryImpl', () {
    final positiveLabel = 1.0;
    final negativeLabel = -1.0;
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
    final featureNames = ['a', 'b', 'c', 'd'];
    final targetNames = ['target_1', 'target_2', 'target_3'];
    final observations = DataFrame.fromMatrix(
      Matrix.fromColumns([
        ...features.columns,
        ...outcomes.columns,
      ], dtype: DType.float32),
      header: [...featureNames, ...targetNames],
    );
    final optimizerType = LinearOptimizerType.gradient;
    final iterationsLimit = 3;
    final initialLearningRate = 0.75;
    final minCoefficientsUpdate = 0.3;
    final lambda = 12.5;
    final regularizationType = RegularizationType.L2;
    final randomSeed = 144;
    final batchSize = 2;
    final isFittingDataNormalized = true;
    final learningRateType = LearningRateType.decreasingAdaptive;
    final initialCoefficientsType = InitialCoefficientsType.zeroes;
    final initialCoefficients = Matrix.fromList([
      [13, 43, 55]
    ]);
    final fitIntercept = false;
    final interceptScale = 1.0;
    final dtype = DType.float32;
    final collectLearningData = true;
    final factoryMock = MockSoftmaxRegressorFactory();
    final regressorMock = MockSoftmaxRegressor();
    final createRegressor = () => SoftmaxRegressor(
          observations,
          targetNames,
          optimizerType: optimizerType,
          iterationsLimit: iterationsLimit,
          initialLearningRate: initialLearningRate,
          minCoefficientsUpdate: minCoefficientsUpdate,
          lambda: lambda,
          regularizationType: regularizationType,
          randomSeed: randomSeed,
          batchSize: batchSize,
          fitIntercept: fitIntercept,
          interceptScale: interceptScale,
          learningRateType: learningRateType,
          isFittingDataNormalized: isFittingDataNormalized,
          initialCoefficientsType: initialCoefficientsType,
          initialCoefficients: initialCoefficients,
          positiveLabel: positiveLabel,
          negativeLabel: negativeLabel,
          collectLearningData: collectLearningData,
          dtype: dtype,
        );

    setUp(() {
      softmaxRegressorInjector
          .registerSingleton<SoftmaxRegressorFactory>(() => factoryMock);

      when(factoryMock.create(
        trainData: observations,
        targetNames: targetNames,
        optimizerType: optimizerType,
        iterationsLimit: iterationsLimit,
        initialLearningRate: initialLearningRate,
        minCoefficientsUpdate: minCoefficientsUpdate,
        lambda: lambda,
        regularizationType: regularizationType,
        randomSeed: randomSeed,
        batchSize: batchSize,
        fitIntercept: fitIntercept,
        interceptScale: interceptScale,
        learningRateType: learningRateType,
        isFittingDataNormalized: isFittingDataNormalized,
        initialCoefficientsType: initialCoefficientsType,
        initialCoefficients: initialCoefficients,
        positiveLabel: positiveLabel,
        negativeLabel: negativeLabel,
        collectLearningData: collectLearningData,
        dtype: dtype,
      )).thenReturn(regressorMock);
    });

    tearDown(() {
      injector.clearAll();
      softmaxRegressorInjector.clearAll();
    });

    test('should pass all the arguments to the softmax regressor factory', () {
      createRegressor();

      verify(factoryMock.create(
        trainData: observations,
        targetNames: targetNames,
        optimizerType: optimizerType,
        iterationsLimit: iterationsLimit,
        initialLearningRate: initialLearningRate,
        minCoefficientsUpdate: minCoefficientsUpdate,
        lambda: lambda,
        regularizationType: regularizationType,
        randomSeed: randomSeed,
        batchSize: batchSize,
        fitIntercept: fitIntercept,
        interceptScale: interceptScale,
        learningRateType: learningRateType,
        isFittingDataNormalized: isFittingDataNormalized,
        initialCoefficientsType: initialCoefficientsType,
        initialCoefficients: initialCoefficients,
        positiveLabel: positiveLabel,
        negativeLabel: negativeLabel,
        collectLearningData: true,
        dtype: dtype,
      )).called(1);
    });

    test('should return an instance created by softmax regressor factory', () {
      expect(createRegressor(), regressorMock);
    });
  });
}
