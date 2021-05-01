import 'package:ml_algo/ml_algo.dart';
import 'package:ml_algo/src/di/injector.dart';
import 'package:ml_algo/src/linear_optimizer/gradient_optimizer/learning_rate_generator/learning_rate_type.dart';
import 'package:ml_algo/src/linear_optimizer/initial_coefficients_generator/initial_coefficients_type.dart';
import 'package:ml_algo/src/linear_optimizer/linear_optimizer_type.dart';
import 'package:ml_algo/src/regressor/linear_regressor/_injector.dart';
import 'package:ml_algo/src/regressor/linear_regressor/linear_regressor_constants.dart';
import 'package:ml_algo/src/regressor/linear_regressor/linear_regressor_factory.dart';
import 'package:ml_algo/src/regressor/linear_regressor/linear_regressor_impl.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:mockito/mockito.dart';
import 'package:test/test.dart';

import '../../mocks.dart';
import '../../mocks.mocks.dart';

void main() {
  group('LinearRegressorImpl', () {
    late LinearRegressorFactory regressorFactory;

    final coefficients = Vector.fromList([1, 2, 3, 4, 5]);
    final targetName = 'Target';
    final optimizerType = LinearOptimizerType.coordinate;
    final iterationsLimit = 233;
    final learningRateType = LearningRateType.constant;
    final initialCoefficientsType = InitialCoefficientsType.zeroes;
    final initialLearningRate = 7.89065;
    final minCoefficientsUpdate = 345.998;
    final lambda = 19823.876;
    final regularizationType = RegularizationType.L1;
    final randomSeed = 9001;
    final batchSize = 234;
    final initialCoefficients = Matrix.column([23, 44, 22.12]);
    final isFittingDataNormalized = false;
    final fitIntercept = true;
    final interceptScale = 109.23;
    final costPerIteration = [23, 34, 12];
    final dtype = DType.float32;
    final retrainingData = DataFrame([[12, 34, -45.66]]);
    final retrainedModelMock = MockLinearRegressor();
    final createRegressor = ({
      int schemaVersion = linearRegressorJsonSchemaVersion,
    }) => LinearRegressorImpl(
      coefficients,
      targetName,
      optimizerType: optimizerType,
      iterationsLimit: iterationsLimit,
      learningRateType: learningRateType,
      initialCoefficientsType: initialCoefficientsType,
      initialLearningRate: initialLearningRate,
      minCoefficientsUpdate: minCoefficientsUpdate,
      lambda: lambda,
      regularizationType: regularizationType,
      randomSeed: randomSeed,
      batchSize: batchSize,
      initialCoefficients: initialCoefficients,
      isFittingDataNormalized: isFittingDataNormalized,
      fitIntercept: fitIntercept,
      interceptScale: interceptScale,
      costPerIteration: costPerIteration,
      dtype: dtype,
      schemaVersion: schemaVersion,
    );

    setUp(() {
      regressorFactory = createLinearRegressorFactoryMock(retrainedModelMock);
      linearRegressorInjector
          .registerSingleton<LinearRegressorFactory>(() => regressorFactory);
    });

    tearDown(() {
      reset(regressorFactory);
      reset(retrainedModelMock);

      injector.clearAll();
      linearRegressorInjector.clearAll();
    });

    test('should call a factory while retraining the model', () {
      createRegressor().retrain(retrainingData);

      verify(regressorFactory.create(
        fittingData: retrainingData,
        targetName: targetName,
        optimizerType: optimizerType,
        iterationsLimit: iterationsLimit,
        learningRateType: learningRateType,
        initialCoefficientsType: initialCoefficientsType,
        initialLearningRate: initialLearningRate,
        minCoefficientsUpdate: minCoefficientsUpdate,
        lambda: lambda,
        regularizationType: regularizationType,
        fitIntercept: fitIntercept,
        interceptScale: interceptScale,
        randomSeed: randomSeed,
        batchSize: batchSize,
        initialCoefficients: initialCoefficients,
        isFittingDataNormalized: isFittingDataNormalized,
        collectLearningData: false,
        dtype: dtype,
      )).called(1);
    });

    test('should return a new instance as a retrained model', () {
      final regressor = createRegressor();
      final retrainedModel = regressor.retrain(retrainingData);

      expect(retrainedModel, same(retrainedModelMock));
      expect(retrainedModel, isNot(same(regressor)));
    });

    test('should return a proper schema version', () {
      final regressor = createRegressor();

      expect(regressor.schemaVersion, 2);
    });
  });
}
